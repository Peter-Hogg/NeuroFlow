"""Check that NeuroFlow's computation semantics do not depend on transport.

The claim under test is narrow. Both runs open the same DANDI asset, select the
same frames and z-planes, and persist the same ordinary NumPy temporal median.
Only the remote HDF5 transport differs, so any numerical disagreement between
the two persisted Zarr outputs is a transport defect rather than an analysis
difference.

Each backend runs in a fresh subprocess. Peak RSS is a process high-water mark
that cannot be reset, so two sequential backends in one interpreter would make
the second silently inherit the first's peak. The parent then reopens both
outputs with checksum verification and compares them elementwise.

The slice is deliberately tiny. This measures equivalence, not throughput, and
the asset's native chunk is a complete image plane, so a handful of frames
across two planes already exercises the whole read path for a few tens of MB
per backend instead of the 230 GB the full-movie benchmark transfers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

import neuroflow
from benchmarks.benchmark_projection import _tree_size
from examples.dandi_fish_projection import (
    ASSET_ID,
    ASSET_PATH,
    DANDISET,
    MOVIE_SHAPE,
    OBJECT_NAME,
    build_projection,
)
from neuroflow.benchmarking import (
    benchmark_record,
    peak_rss_bytes,
    write_benchmark_record,
)
from neuroflow.provenance import capture_environment
from neuroflow.selection import NWBQuery

BACKENDS: tuple[str, ...] = ("remfile", "lindi")

# Fields that must agree for the two runs to be the same computation rather
# than two computations that happen to produce similar numbers.
PLAN_COMPARISON_KEYS = (
    "selected_shape",
    "output_shape",
    "output_axes",
    "dtype",
    "native_chunks",
    "processing_partition_shape",
    "task_count",
)


def output_path(
    root: Path, backend: str, frames: int, z_start: int, planes: int
) -> Path:
    return root / f"lindi-equivalence-{backend}-t{frames}-z{z_start}n{planes}.zarr"


def run_backend(
    *,
    backend: str,
    frames: int,
    planes: int,
    z_start: int,
    output: Path,
    tile_y: int,
    tile_x: int,
    block_size: int,
    cache_size: int,
    memory_limit: str,
) -> dict[str, object]:
    """Persist the projection through one transport and report its metrics."""
    # LINDI manages its own remote access and refuses remfile cache options.
    storage_options: dict[str, object] | None = (
        None
        if backend == "lindi"
        else {"block_size": block_size, "cache_size": cache_size}
    )
    started = time.perf_counter()
    source = neuroflow.open_dandi(
        DANDISET, backend=cast(Any, backend), storage_options=storage_options
    )
    try:
        selected = source.select(NWBQuery(asset=ASSET_ID, name=OBJECT_NAME))
        bounded = selected.isel(
            time=slice(0, frames), z=slice(z_start, z_start + planes)
        )
        open_seconds = time.perf_counter() - started
        movie = neuroflow.NeuroArray(source, bounded)
        persist_started = time.perf_counter()
        persisted = build_projection(movie).persist(
            output,
            chunks=(tile_y, tile_x, 1),
            max_workers=1,
            memory_limit=memory_limit,
        )
        persist_seconds = time.perf_counter() - persist_started
        workflow = persisted.workflow
        assert workflow is not None
        plan = workflow.plan
        verified = bool(workflow.verify().valid)
        metrics = (workflow.provenance or {}).get("execution_metrics", {})
        attributes = bounded.metadata.attributes or {}
        selected_dtype = np.dtype(bounded.metadata.dtype)
        stats = source.io_stats()
        persisted.close()
        return {
            "backend": backend,
            "transport": attributes.get("transport"),
            "status": "measured",
            "workflow_id": plan.workflow_id,
            "selected_shape": list(bounded.metadata.shape),
            "selected_dtype": str(selected_dtype),
            "selected_bytes": int(np.prod(bounded.metadata.shape))
            * selected_dtype.itemsize,
            "native_chunks": (
                list(plan.native_chunks) if plan.native_chunks is not None else None
            ),
            "output_shape": list(plan.output_shape),
            "output_axes": list(plan.output_axes),
            "output_chunks": [tile_y, tile_x, 1],
            "dtype": plan.dtype,
            "processing_partition_shape": list(plan.processing_partition_shape),
            "task_count": plan.task_count,
            "estimated_total_bytes_read": plan.estimated_total_bytes_read,
            "open_seconds": open_seconds,
            "persist_seconds": persist_seconds,
            "wall_time_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes(),
            "integrity_verified": verified,
            "output": str(output),
            "output_bytes": _tree_size(output),
            "execution_metrics": metrics if isinstance(metrics, dict) else {},
            "io_stats": stats,
            # remfile hooks a response counter onto its requests session; LINDI
            # exposes no equivalent, so this stays null instead of a false zero.
            "bytes_read": stats.get("response_content_bytes"),
            "http_responses": stats.get("http_responses"),
            "bytes_read_status": (
                "counted from HTTP Content-Length headers"
                if isinstance(stats.get("response_content_bytes"), int)
                else "unknown: this transport exposes no byte counter"
            ),
        }
    finally:
        source.close()


def _spawn(backend: str, arguments: argparse.Namespace, output: Path) -> dict[str, Any]:
    """Run one backend in a fresh interpreter and return its JSON payload."""
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.benchmark_lindi_equivalence",
            "--output-root",
            str(arguments.output_root),
            "--backend",
            backend,
            "--frames",
            str(arguments.frames),
            "--planes",
            str(arguments.planes),
            "--z-start",
            str(arguments.z_start),
            "--tile-y",
            str(arguments.tile_y),
            "--tile-x",
            str(arguments.tile_x),
            "--block-size",
            str(arguments.block_size),
            "--cache-size-mib",
            str(arguments.cache_size_mib),
            "--memory-limit",
            arguments.memory_limit,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    payload: dict[str, Any] | None = None
    for line in process.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                payload = cast(dict[str, Any], json.loads(stripped))
            except json.JSONDecodeError:
                continue
    if process.returncode != 0 or payload is None:
        return {
            "backend": backend,
            "status": "failed",
            "returncode": process.returncode,
            "output": str(output),
            "stderr": process.stderr[-4000:],
        }
    return payload


def _read_output(path: Path) -> np.ndarray[Any, np.dtype[Any]]:
    """Reopen a persisted result with partition-checksum verification."""
    source, selection = neuroflow.open_array(path)
    try:
        return np.asarray(selection.as_dask_array().compute())
    finally:
        source.close()


def _relative_error(left: Any, right: Any) -> float:
    """Largest |left - right| / |right| over elements with a non-zero reference."""
    reference = np.abs(right)
    nonzero = reference > 0.0
    if not bool(np.any(nonzero)):
        return 0.0
    return float(np.max(np.abs(left - right)[nonzero] / reference[nonzero]))


def compare_outputs(results: list[dict[str, Any]]) -> dict[str, object]:
    """Compare every persisted output against the first backend's output."""
    arrays: dict[str, np.ndarray[Any, np.dtype[Any]]] = {}
    checksums: dict[str, object] = {}
    for result in results:
        backend = str(result["backend"])
        if result.get("status") != "measured":
            checksums[backend] = None
            continue
        array = _read_output(Path(str(result["output"])))
        arrays[backend] = array
        checksums[backend] = hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest()
    names = list(arrays)
    if len(names) < 2:
        return {
            "valid": None,
            "status": "fewer than two backends produced an output",
            "checksums": checksums,
            "atol": 0.0,
            "rtol": 0.0,
        }
    reference_name = names[0]
    reference = arrays[reference_name]
    pairs: list[dict[str, object]] = []
    for name in names[1:]:
        candidate = arrays[name]
        shape_equal = candidate.shape == reference.shape
        left = candidate.astype(np.float64) if shape_equal else np.array([np.inf])
        right = reference.astype(np.float64) if shape_equal else np.array([0.0])
        pairs.append(
            {
                "left": name,
                "right": reference_name,
                "shape_agreement": shape_equal,
                "dtype_agreement": candidate.dtype == reference.dtype,
                "elementwise_equal": shape_equal
                and bool(np.array_equal(candidate, reference)),
                "checksums_equal": checksums[name] == checksums[reference_name],
                "maximum_absolute_error": float(np.max(np.abs(left - right))),
                "maximum_relative_error": _relative_error(left, right),
            }
        )
    return {
        "valid": all(
            bool(pair["elementwise_equal"]) and bool(pair["checksums_equal"])
            for pair in pairs
        ),
        "comparison": "identical slice, identical expression, differing transport",
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "checksums": checksums,
        "pairs": pairs,
        # Exact equality is the assertion, so the tolerances are zero.
        "atol": 0.0,
        "rtol": 0.0,
        "repeatability": "median of integers is deterministic; single run per backend",
    }


def _plan_agreement(results: list[dict[str, Any]]) -> dict[str, object]:
    """Report whether both transports produced the same partitioning decisions."""
    measured = [item for item in results if item.get("status") == "measured"]
    agreement: dict[str, object] = {}
    for key in PLAN_COMPARISON_KEYS:
        values = [json.dumps(item.get(key), sort_keys=True) for item in measured]
        agreement[key] = len(set(values)) == 1 if values else None
    return agreement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        help="single transport; used for subprocess dispatch",
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--planes", type=int, default=2)
    parser.add_argument("--z-start", type=int, default=14)
    parser.add_argument("--tile-y", type=int, default=256)
    parser.add_argument("--tile-x", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=262_144)
    parser.add_argument("--cache-size-mib", type=int, default=64)
    parser.add_argument("--memory-limit", default="2 GiB")
    parser.add_argument(
        "--classification", choices=("current", "publication"), default="current"
    )
    arguments = parser.parse_args()
    if not 1 <= arguments.frames <= 32:
        parser.error("--frames must be between 1 and 32; this is an equivalence check")
    if not 1 <= arguments.planes <= 4:
        parser.error("--planes must be between 1 and 4; this is an equivalence check")
    if not 0 <= arguments.z_start <= MOVIE_SHAPE[3] - arguments.planes:
        parser.error("--z-start plus --planes must stay inside the z axis")
    arguments.output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        backend: output_path(
            arguments.output_root,
            backend,
            arguments.frames,
            arguments.z_start,
            arguments.planes,
        )
        for backend in BACKENDS
    }

    if arguments.backend is not None:
        print(
            json.dumps(
                run_backend(
                    backend=arguments.backend,
                    frames=arguments.frames,
                    planes=arguments.planes,
                    z_start=arguments.z_start,
                    output=paths[arguments.backend],
                    tile_y=arguments.tile_y,
                    tile_x=arguments.tile_x,
                    block_size=arguments.block_size,
                    cache_size=arguments.cache_size_mib * 1024 * 1024,
                    memory_limit=arguments.memory_limit,
                ),
                sort_keys=True,
            )
        )
        return

    environment = capture_environment()
    git = cast(dict[str, object], environment["git"])
    if arguments.classification == "publication" and git.get("dirty") is not False:
        parser.error("publication classification requires a clean Git tree")
    if arguments.record is None:
        parser.error("--record is required when comparing every backend")
    for backend, path in paths.items():
        if path.exists():
            parser.error(
                f"{path} already exists; a resumed {backend} run would report a "
                "wall time that no longer measures the transport"
            )

    started = time.perf_counter()
    results = [_spawn(backend, arguments, paths[backend]) for backend in BACKENDS]
    validation = compare_outputs(results)
    wall_time = time.perf_counter() - started
    failures = [item for item in results if item.get("status") != "measured"]
    measured = {str(item["backend"]): item for item in results if item not in failures}
    reference = measured.get("remfile") or (
        next(iter(measured.values())) if measured else {}
    )

    record = benchmark_record(
        name="dandi-fish-transport-equivalence",
        classification=arguments.classification,
        backend="nwb-hdf5 remfile vs lindi",
        source={
            "dataset_identifier": DANDISET.split("@", 1)[0],
            "dataset_version": DANDISET.split("@", 1)[1],
            "asset": ASSET_PATH,
            "asset_id": ASSET_ID,
            "path": f"/acquisition/{OBJECT_NAME}",
            "shape": list(MOVIE_SHAPE),
            "dtype": reference.get("selected_dtype"),
            "physical_chunk_shape": [1, MOVIE_SHAPE[1], MOVIE_SHAPE[2], 1],
            "total_logical_bytes": int(np.prod(MOVIE_SHAPE)) * 2,
            "selected_bytes": reference.get("selected_bytes"),
            "selected_shape": reference.get("selected_shape"),
            "selection": {
                "frames": arguments.frames,
                "z_start": arguments.z_start,
                "planes": arguments.planes,
            },
        },
        execution={
            "partition_configuration": {
                "processing_shape": reference.get("processing_partition_shape"),
                "output_chunks": reference.get("output_chunks"),
                "identical_across_backends": _plan_agreement(results),
            },
            "memory_budget": arguments.memory_limit,
            "task_count": reference.get("task_count"),
            # No aggregate is defensible: only one of the two transports can be
            # measured, so the honest total is unknown. See per_backend below.
            "bytes_read": None,
            "peak_rss_bytes": max(
                (int(item["peak_rss_bytes"]) for item in measured.values()),
                default=peak_rss_bytes(),
            ),
            "wall_time_seconds": wall_time,
            "cache_state": (
                f"cold per backend, fresh subprocess each; remfile block "
                f"{arguments.block_size} B, cache {arguments.cache_size_mib} MiB; "
                "LINDI manages its own remote access"
            ),
            "network_context": "public DANDI HTTPS; runner location is external",
            "independent_process_per_backend": True,
            "per_backend": results,
        },
        result={
            "checksum": validation.get("checksums", {}),
            "numerical_validation": validation,
            "integrity_verified": bool(measured)
            and all(bool(item["integrity_verified"]) for item in measured.values()),
            "resume": {"supported": True, "exercised": False},
            "output_bytes": sum(
                int(item["output_bytes"]) for item in measured.values()
            ),
        },
        notes=[
            "Only the HDF5 transport differs; the selection, the NumPy "
            "expression, the output chunking, and the memory budget are shared.",
            "Outputs are compared after reopening them with partition-checksum "
            "verification, not from in-process arrays.",
            "bytes_read is measured for remfile from HTTP Content-Length headers "
            "and is null for LINDI, which exposes no transport counter through "
            "LindiH5pyFile; a zero there would be a false measurement.",
            "Peak RSS is a whole-process high-water mark for each backend's own "
            "subprocess, so the two values are independent.",
            "Wall time here is dominated by remote latency on a few native "
            "chunks and is not a throughput comparison.",
            "The outputs live under a scratch directory; publication run stores "
            "are never written by this benchmark.",
        ],
    )
    write_benchmark_record(arguments.record, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
