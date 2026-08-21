"""Run projection → real Cellpose → whole-movie soma traces on DANDI:000350."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import zarr

import neuroflow
from benchmarks.benchmark_projection import _tree_size
from examples.dandi_fish_projection import (
    ASSET_ID,
    ASSET_PATH,
    DANDISET,
    MOVIE_SHAPE,
    OBJECT_NAME,
    FishProjectionConfig,
    run_example,
)
from neuroflow.benchmarking import benchmark_record, write_benchmark_record
from neuroflow.provenance import capture_environment
from neuroflow.selection import NWBQuery
from neuroflow_cellpose import (
    DEVICE_CHOICES,
    CellposeDevice,
    resolve_cellpose_device,
)

CELLPOSE_EVAL_SETTINGS: dict[str, object] = {
    "batch_size": 1,
    "channels": None,
    "channel_axis": None,
    "z_axis": None,
    "normalize": True,
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "do_3D": False,
    "anisotropy": None,
    "min_size": 15,
    "tile_overlap": 0.1,
}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _integer_metric(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _sum_known(values: list[int | None]) -> int | None:
    return sum(cast(list[int], values)) if all(v is not None for v in values) else None


def _direct_cellpose_equivalence(
    projection: neuroflow.NeuroArray,
    labels: neuroflow.NeuroArray,
    *,
    model_name: str,
    device: CellposeDevice,
) -> dict[str, object]:
    """Run Cellpose directly on every persisted projection plane.

    ``device`` is the same resolved device the NeuroFlow-mediated run used.
    Comparing masks produced on different devices would test nothing useful,
    because CPU and GPU kernels are not required to be bit-identical.
    """
    models = importlib.import_module("cellpose.models")
    model_type = getattr(models, "CellposeModel", None)
    if model_type is None:
        raise RuntimeError("installed Cellpose has no CellposeModel API")
    model = model_type(
        gpu=device.gpu,
        pretrained_model=model_name,
        use_bfloat16=False,
    )
    projection_data = projection.selection.as_dask_array()
    mediated_data = labels.selection.as_dask_array()
    if projection.axes != ("y", "x", "z") or labels.axes != projection.axes:
        raise ValueError("the fish Cellpose comparison expects y/x/z projections")
    mismatch_count = 0
    direct_object_count = 0
    started = time.perf_counter()
    for plane in range(projection.shape[2]):
        model_input = np.asarray(
            projection_data[:, :, plane].compute(scheduler="threads", num_workers=1)
        )
        evaluation = model.eval(model_input, **CELLPOSE_EVAL_SETTINGS)
        if not isinstance(evaluation, tuple) or not evaluation:
            raise TypeError("CellposeModel.eval() returned an unsupported value")
        direct = np.asarray(evaluation[0])
        mediated_global = np.asarray(
            mediated_data[:, :, plane].compute(scheduler="threads", num_workers=1)
        )
        namespace = np.uint64(plane + 1) << np.uint64(32)
        mediated = np.where(
            mediated_global != 0,
            mediated_global - namespace,
            0,
        ).astype(direct.dtype)
        mismatch_count += int(np.count_nonzero(mediated != direct))
        direct_object_count += int(np.count_nonzero(np.unique(direct)))
    return {
        "valid": mismatch_count == 0,
        "exact": True,
        "mismatched_voxels": mismatch_count,
        "direct_object_count": direct_object_count,
        "wall_time_seconds": time.perf_counter() - started,
        "cellpose_version": importlib.metadata.version("cellpose"),
        "model": model_name,
        "settings": CELLPOSE_EVAL_SETTINGS,
        "device": device.to_dict(),
    }


def _direct_numpy_trace_validation(
    movie: neuroflow.NeuroArray,
    labels: neuroflow.NeuroArray,
    traces: neuroflow.NeuroArray,
    *,
    trace_output: Path,
    frames: int,
) -> dict[str, object]:
    """Compare leading fish frames with a direct plane-wise NumPy reduction."""
    if movie.axes != ("time", "y", "x", "z"):
        raise ValueError("the fish NumPy reference expects time/y/x/z movie axes")
    if labels.axes != ("y", "x", "z"):
        raise ValueError("the fish NumPy reference expects y/x/z labels")
    movie_data = movie.selection.as_dask_array(chunks="native")
    label_data = labels.selection.as_dask_array()
    group = zarr.open_group(str(trace_output), mode="r")
    cell_ids = np.asarray(group["cell_ids"], dtype=np.uint64)
    id_to_column = {int(value): index for index, value in enumerate(cell_ids)}
    expected = np.full((frames, len(cell_ids)), np.nan, dtype=np.float32)
    started = time.perf_counter()
    for plane in range(labels.shape[2]):
        global_labels = np.asarray(
            label_data[:, :, plane].compute(scheduler="threads", num_workers=1),
            dtype=np.uint64,
        )
        namespace = np.uint64(plane + 1) << np.uint64(32)
        local_labels = np.where(
            global_labels != 0,
            global_labels - namespace,
            0,
        ).astype(np.int64)
        flat_labels = local_labels.reshape(-1)
        local_ids = np.unique(flat_labels)
        local_ids = local_ids[local_ids != 0]
        if not len(local_ids):
            continue
        counts = np.bincount(flat_labels)
        block = np.asarray(
            movie_data[:frames, :, :, plane].compute(
                scheduler="threads", num_workers=1
            ),
            dtype=np.float32,
        ).reshape(frames, -1)
        columns = [
            id_to_column[int(namespace + np.uint64(label_id))] for label_id in local_ids
        ]
        for frame in range(frames):
            sums = np.bincount(flat_labels, weights=block[frame])
            expected[frame, columns] = (sums[local_ids] / counts[local_ids]).astype(
                np.float32
            )
    actual = np.asarray(
        traces.selection.as_dask_array()[:frames].compute(
            scheduler="threads", num_workers=1
        ),
        dtype=np.float32,
    )
    if np.isnan(expected).any():
        raise RuntimeError("direct NumPy validation did not cover every trace")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    maximum_absolute_error = float(np.max(difference, initial=0.0))
    maximum_relative_error = float(np.max(difference / denominator, initial=0.0))
    atol = 1e-5
    rtol = 1e-5
    return {
        "valid": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape_agreement": actual.shape == expected.shape,
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "maximum_absolute_error": maximum_absolute_error,
        "maximum_relative_error": maximum_relative_error,
        "atol": atol,
        "rtol": rtol,
        "direct_checksum": hashlib.sha256(expected.tobytes()).hexdigest(),
        "wall_time_seconds": time.perf_counter() - started,
        "method": "direct plane-wise NumPy mean on the leading movie frames",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("publication/runs"))
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("auto", "lindi", "remfile"), default="auto"
    )
    parser.add_argument("--memory-limit", default="2 GiB")
    # Explicit transport configuration, applied to every stage and recorded
    # numerically. The projection stage previously ran through the example
    # default (256 KiB blocks) while trace extraction opened with the library
    # default (1 MiB blocks), so transfer figures mixed two configurations and
    # were not comparable with the manual baseline. remfile only; LINDI
    # manages its own remote access and ignores these.
    parser.add_argument("--block-size", type=int, default=1_048_576)
    parser.add_argument("--cache-size-mib", type=int, default=64)
    parser.add_argument("--projection-frames", type=int, default=50)
    parser.add_argument("--validation-frames", type=int, default=1)
    parser.add_argument("--cellpose-model", default="cpsam")
    parser.add_argument(
        "--cellpose-device",
        choices=DEVICE_CHOICES,
        default="auto",
        help=(
            "device for both the NeuroFlow-mediated and the direct Cellpose "
            "run; 'auto' uses CUDA when available and CPU otherwise"
        ),
    )
    parser.add_argument(
        "--classification", choices=("current", "publication"), default="current"
    )
    args = parser.parse_args()
    environment = capture_environment()
    git = cast(dict[str, object], environment["git"])
    if args.classification == "publication" and git.get("dirty") is not False:
        parser.error("publication classification requires a clean Git tree")
    if not 1 <= args.projection_frames <= 50:
        parser.error("--projection-frames must be between 1 and 50")
    if not 1 <= args.validation_frames <= MOVIE_SHAPE[0]:
        parser.error("--validation-frames is outside the movie")
    # Resolve the device before any download or segmentation so an unsatisfiable
    # request fails in a second rather than after hours of remote reads.
    try:
        cellpose_device = resolve_cellpose_device(args.cellpose_device)
    except RuntimeError as exc:
        parser.error(str(exc))

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    projection_path = root / "fish-projection.zarr"
    preview_path = root / "fish-projection-z14.png"
    segmentation_path = root / "fish-cellpose.zarr"
    traces_path = root / "fish-traces.zarr"
    pipeline_started = time.perf_counter()
    projection_summary = run_example(
        FishProjectionConfig(
            frames=args.projection_frames,
            backend=cast(Literal["auto", "lindi", "remfile"], args.backend),
            block_size=args.block_size,
            cache_size=args.cache_size_mib * 1024 * 1024,
            output=projection_path,
            preview=preview_path,
        )
    )

    projection_source: Any | None = None
    movie_source: Any | None = None
    projection: neuroflow.NeuroArray | None = None
    labels: neuroflow.NeuroArray | None = None
    movie: neuroflow.NeuroArray | None = None
    traces: neuroflow.NeuroArray | None = None
    resumed: neuroflow.NeuroArray | None = None
    try:
        projection_source, projection_selection = neuroflow.open_array(projection_path)
        projection = neuroflow.NeuroArray(projection_source, projection_selection)
        cellpose_started = time.perf_counter()
        labels = projection.cellpose(
            pretrained_model=args.cellpose_model,
            output=segmentation_path,
            memory_limit=args.memory_limit,
            max_workers=1,
            gpu=cellpose_device.gpu,
            use_bfloat16=False,
            batch_size=1,
        )
        segmentation = labels.workflow
        assert segmentation is not None
        if not segmentation.verify().valid:
            raise RuntimeError("Cellpose segmentation failed integrity verification")
        cellpose_seconds = time.perf_counter() - cellpose_started
        object_table = segmentation.tables["objects"].as_dask_dataframe().compute()
        segmentation_provenance = segmentation.provenance or {}
        direct_cellpose = _direct_cellpose_equivalence(
            projection,
            labels,
            model_name=args.cellpose_model,
            device=cellpose_device,
        )
        if not direct_cellpose["valid"]:
            raise RuntimeError("NeuroFlow-mediated Cellpose labels differ from direct")

        movie_source = neuroflow.open_dandi(
            DANDISET,
            backend=cast(Literal["auto", "lindi", "remfile"], args.backend),
            storage_options=(
                None
                if args.backend == "lindi"
                else {
                    "block_size": args.block_size,
                    "cache_size": args.cache_size_mib * 1024 * 1024,
                }
            ),
        )
        movie_selection = movie_source.select(
            NWBQuery(asset=ASSET_ID, name=OBJECT_NAME)
        )
        movie = neuroflow.NeuroArray(movie_source, movie_selection)
        trace_plan = movie.plan_traces(labels, memory_limit=args.memory_limit)
        traces = movie.extract_traces(
            labels,
            output=traces_path,
            memory_limit=args.memory_limit,
        )
        trace_result = neuroflow.open_result(traces_path)
        verification = trace_result.verify()
        if not verification.valid:
            raise RuntimeError("trace output failed integrity verification")
        first_trace_provenance = trace_result.provenance
        first_trace_metrics = first_trace_provenance.get("execution_metrics", {})
        if not isinstance(first_trace_metrics, dict):
            first_trace_metrics = {}

        before_validation = movie_source.io_stats()
        trace_validation = _direct_numpy_trace_validation(
            movie,
            labels,
            traces,
            trace_output=traces_path,
            frames=args.validation_frames,
        )
        after_validation = movie_source.io_stats()
        validation_before = _integer_metric(
            before_validation.get("response_content_bytes")
        )
        validation_after = _integer_metric(
            after_validation.get("response_content_bytes")
        )
        validation_bytes = (
            max(0, validation_after - validation_before)
            if validation_before is not None and validation_after is not None
            else None
        )
        if not trace_validation["valid"]:
            raise RuntimeError("fish trace subset differs from direct NumPy")

        # A second identical call verifies that all durable time partitions are
        # reused. Per-attempt metrics preserve the original full-run evidence.
        resumed = movie.extract_traces(
            labels,
            output=traces_path,
            memory_limit=args.memory_limit,
        )
        resumed_provenance = neuroflow.open_result(traces_path).provenance
        resume_metrics = resumed_provenance.get("execution_metrics", {})
        if not isinstance(resume_metrics, dict):
            resume_metrics = {}
        if resume_metrics.get("computed_task_count") != 0:
            raise RuntimeError("completed trace resume unexpectedly recomputed work")

        trace_checksum = first_trace_provenance.get("result_checksum")
        if not isinstance(trace_checksum, str):
            raise RuntimeError("trace provenance has no result checksum")
        projection_result = neuroflow.open_result(projection_path)
        projection_provenance = projection_result.provenance
        projection_metrics = projection_provenance.get("execution_metrics", {})
        if not isinstance(projection_metrics, dict):
            projection_metrics = {}
        projection_remote = projection_summary.get("remote_io", {})
        if not isinstance(projection_remote, dict):
            projection_remote = {}
        total_remote_bytes = _sum_known(
            [
                _integer_metric(projection_remote.get("response_content_bytes")),
                _integer_metric(first_trace_metrics.get("bytes_read")),
                validation_bytes,
                _integer_metric(resume_metrics.get("bytes_read")),
            ]
        )
        source_attributes = movie.selection.metadata.attributes or {}
        projection_task_count = projection_summary.get("task_count")
        if not isinstance(projection_task_count, int):
            raise TypeError("projection summary has no integer task count")
        task_count = (
            projection_task_count + segmentation.plan.task_count + trace_plan.task_count
        )
        resume_record = {
            "exercised": True,
            "computed_task_count": resume_metrics.get("computed_task_count"),
            "resumed_task_count": resume_metrics.get("resumed_task_count"),
            "bytes_read": resume_metrics.get("bytes_read"),
            "integrity_verified": neuroflow.open_result(traces_path).verify().valid,
            "corruption_repair": (
                "covered by benchmarks.benchmark_resume_integrity on deterministic "
                "local data; archive output was not deliberately corrupted"
            ),
        }
        record = benchmark_record(
            name="dandi-fish-cellpose-soma-traces",
            classification=args.classification,
            backend=f"dandi-nwb-hdf5-{source_attributes.get('transport')}",
            source={
                "dataset_identifier": DANDISET.split("@", 1)[0],
                "dataset_version": DANDISET.split("@", 1)[1],
                "asset": ASSET_PATH,
                "asset_id": ASSET_ID,
                "path": f"/acquisition/{OBJECT_NAME}",
                "shape": list(MOVIE_SHAPE),
                "dtype": movie.dtype.str,
                "physical_chunk_shape": list(
                    movie.selection.metadata.native_chunks or ()
                ),
                "total_logical_bytes": movie.nbytes,
                "selected_bytes": movie.nbytes,
            },
            execution={
                "partition_configuration": {
                    "projection_tasks": projection_summary["task_count"],
                    "cellpose_tasks": segmentation.plan.task_count,
                    "trace_plan": trace_plan.to_dict(),
                },
                "memory_budget": args.memory_limit,
                "task_count": task_count,
                "bytes_read": total_remote_bytes,
                "peak_rss_bytes": _peak_rss_bytes(),
                "wall_time_seconds": time.perf_counter() - pipeline_started,
                "cache_state": (
                    "backend-managed; LINDI counters unavailable through the "
                    "current bridge"
                    if args.backend == "lindi"
                    else (
                        f"bounded remfile: block {args.block_size} B, cache "
                        f"{args.cache_size_mib} MiB, identical across every "
                        "stage"
                    )
                ),
                "transport_configuration": (
                    {"managed_by": "lindi"}
                    if args.backend == "lindi"
                    else {
                        "block_size": args.block_size,
                        "cache_size_bytes": args.cache_size_mib * 1024 * 1024,
                        "applies_to": "projection, trace extraction, resume",
                    }
                ),
                "network_context": "public DANDI HTTPS; runner location is external",
            },
            result={
                "checksum": trace_checksum,
                "numerical_validation": trace_validation,
                "integrity_verified": bool(verification.valid),
                "resume": resume_record,
                "output_bytes": _tree_size(traces_path),
            },
            notes=[
                "No complete source download is performed by this workflow.",
                "Cellpose transport equivalence is distinct from biological accuracy.",
                "A publication-classified run is rejected when Git is dirty.",
                "Trace arrays are stored as (time, cell).",
            ],
        )
        record.update(
            {
                "pipeline_schema_version": "1",
                "policy": {
                    "memory_limit": args.memory_limit,
                    "normal_user_overrides": [
                        "backend",
                        "memory_limit",
                        "cellpose_device",
                    ],
                },
                "compute_device": {
                    "cellpose": cellpose_device.to_dict(),
                    "same_device_for_direct_comparison": (
                        direct_cellpose["device"] == cellpose_device.to_dict()
                    ),
                },
                "stages": {
                    "projection": {
                        "summary": projection_summary,
                        "identity": projection_result.array_source_identity(
                            verify_checksums=False
                        ),
                        "provenance": projection_provenance,
                        "execution_metrics": projection_metrics,
                    },
                    "cellpose": {
                        "input_projection_identity": (
                            projection_result.array_source_identity(
                                verify_checksums=False
                            )
                        ),
                        "cellpose_version": importlib.metadata.version("cellpose"),
                        "model": args.cellpose_model,
                        "device": cellpose_device.to_dict(),
                        "wall_time_seconds": cellpose_seconds,
                        "parameters": segmentation_provenance.get("parameters"),
                        "object_count": int(len(object_table)),
                        "output_checksum": segmentation_provenance.get(
                            "result_checksum"
                        ),
                        "direct_equivalence": direct_cellpose,
                        "provenance": segmentation_provenance,
                    },
                    "traces": {
                        "shape": list(traces.shape),
                        "axes": list(traces.axes),
                        "dtype": traces.dtype.str,
                        "timepoints_processed": movie.shape[0],
                        "soma_count": trace_plan.cell_count,
                        "source_chunks_touched": (
                            trace_plan.estimated_source_chunks_touched
                        ),
                        "preflight": trace_plan.to_dict(),
                        "checksum": trace_checksum,
                        "integrity_verified": verification.valid,
                        "first_execution_metrics": first_trace_metrics,
                        "resume": resume_record,
                        "provenance": first_trace_provenance,
                    },
                },
                "validation": {
                    "cellpose_direct_equivalence": direct_cellpose,
                    "trace_direct_numpy": trace_validation,
                    "trace_validation_remote_bytes": validation_bytes,
                    "biological_accuracy": None,
                },
            }
        )
        write_benchmark_record(args.record, record)
        print(json.dumps(record, indent=2, sort_keys=True))
    finally:
        for value in (resumed, traces, labels, projection, movie):
            if value is not None:
                value.close()
        # NeuroArray.close() owns these sources; these guards cover failures
        # before the arrays themselves were constructed.
        if projection is None and projection_source is not None:
            projection_source.close()
        if movie is None and movie_source is not None:
            movie_source.close()


if __name__ == "__main__":
    main()
