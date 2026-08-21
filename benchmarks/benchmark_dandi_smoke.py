"""Smoke-validate the generic engine on an unfamiliar real DANDI dataset.

The claim under test is generality, not throughput: the execution engine is
dataset-independent by construction, and this benchmark converts that into a
measurement by running the ordinary public workflow -- discover an object with
``NWBQuery(neurodata_type=...)``, inspect what NeuroFlow inferred, preflight a
plan, persist a bounded temporal reduction, verify the output, and compare it
against a direct independent computation -- on a dataset the repository has
never touched.

Nothing here is specific to any dataset. Every identifier arrives through the
command line, so the same harness exercises any NWB-HDF5 asset on DANDI. The
selection is deliberately a bounded number of leading frames: enough to touch
many native chunks through the remote chunked read path, small enough that the
whole run transfers a few hundred MB rather than re-validating archive-scale
behaviour the fish benchmark already covers.

The direct reference deliberately bypasses NeuroFlow: it reads the same frames
through plain h5py over remfile and reduces them with plain NumPy, so agreement
demonstrates the persisted result rather than the engine agreeing with itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import remfile

import neuroflow
from benchmarks.benchmark_projection import _tree_size
from neuroflow.benchmarking import (
    benchmark_record,
    peak_rss_bytes,
    write_benchmark_record,
)
from neuroflow.exceptions import AmbiguousSelectionError
from neuroflow.provenance import capture_environment
from neuroflow.selection import NWBQuery

DOWNLOAD_URL = "https://api.dandiarchive.org/api/assets/{asset_id}/download/"


def _direct_reference(
    asset_id: str, dataset_path: str, frames: int, output_dtype: np.dtype[Any]
) -> np.ndarray[Any, np.dtype[Any]]:
    """Compute the same temporal mean without NeuroFlow.

    Plain h5py over remfile and plain NumPy, sharing nothing with the engine
    but the transport library. ``np.mean`` accumulates integer input in
    float64, which is also what the engine's staged mean uses, so the cast at
    the end makes exact agreement possible rather than merely close.
    """
    remote = h5py.File(
        remfile.File(DOWNLOAD_URL.format(asset_id=asset_id), verbose=False), "r"
    )
    try:
        dataset = remote[f"{dataset_path}/data"]
        assert isinstance(dataset, h5py.Dataset)
        block = np.asarray(dataset[:frames])
    finally:
        remote.close()
    return np.mean(block, axis=0).astype(output_dtype)


def _list_assets(source: Any, limit: int = 10) -> str:
    lines = [
        f"  {asset.asset_id}  {asset.path}" for asset in source.assets()[:limit]
    ]
    if len(source.assets()) > limit:
        lines.append(f"  ... and {len(source.assets()) - limit} more")
    return "\n".join(lines)


def _run_repetitions(args: argparse.Namespace) -> None:
    """Repeat the single-run benchmark in fresh subprocesses and aggregate.

    Each repetition gets its own interpreter (peak RSS is a non-resettable
    high-water mark) and its own output root (rerunning into an existing
    store would measure a zero-compute resume, not a repetition).
    """
    per_run: list[dict[str, Any]] = []
    for index in range(args.repetitions):
        rep_record = args.record.parent / f"{args.record.stem}-rep{index}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.benchmark_dandi_smoke",
            "--dandiset",
            args.dandiset,
            "--neurodata-type",
            args.neurodata_type,
            "--backend",
            args.backend,
            "--frames",
            str(args.frames),
            "--memory-limit",
            args.memory_limit,
            "--output-root",
            str(args.output_root / f"rep{index}"),
            "--record",
            str(rep_record),
            "--repetitions",
            "1",
            "--classification",
            args.classification,
        ]
        if args.asset is not None:
            command.extend(["--asset", args.asset])
        if args.name is not None:
            command.extend(["--name", args.name])
        if args.expect_axes is not None:
            command.extend(["--expect-axes", args.expect_axes])
        process = subprocess.run(
            command,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        if process.returncode != 0:
            raise SystemExit(
                f"repetition {index} failed with exit code {process.returncode}"
            )
        per_run.append(json.loads(rep_record.read_text()))

    def spread(values: list[float]) -> dict[str, float]:
        return {
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    checksums = {run["result"]["checksum"] for run in per_run}
    aggregate = {
        "aggregate_schema_version": "1",
        "benchmark_name": "dandi-second-dataset-smoke-repetitions",
        "classification": args.classification,
        "repetition_count": args.repetitions,
        "environment": capture_environment(),
        # One checksum across every repetition is itself evidence: the
        # computation is deterministic under repeated cold execution.
        "checksums_identical_across_repetitions": len(checksums) == 1,
        "checksums": sorted(checksums),
        "summary": {
            "engine_phase_peak_rss_bytes": spread(
                [
                    float(run["execution"]["engine_phase_peak_rss_bytes"])
                    for run in per_run
                ]
            ),
            "process_peak_rss_bytes": spread(
                [float(run["execution"]["peak_rss_bytes"]) for run in per_run]
            ),
            "wall_time_seconds": spread(
                [float(run["execution"]["wall_time_seconds"]) for run in per_run]
            ),
            "bytes_read": spread(
                [
                    float(run["execution"]["bytes_read"])
                    for run in per_run
                    if run["execution"]["bytes_read"] is not None
                ]
            )
            if any(run["execution"]["bytes_read"] is not None for run in per_run)
            else None,
        },
        "runs": per_run,
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    summary = aggregate["summary"]
    assert isinstance(summary, dict)
    process_peak = summary["process_peak_rss_bytes"]
    print(
        f"repetitions={args.repetitions} "
        f"checksums_identical={aggregate['checksums_identical_across_repetitions']} "
        f"process_peak_median={process_peak['median']:.0f} "
        f"range=[{process_peak['min']:.0f}, {process_peak['max']:.0f}] "
        f"record={args.record}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dandiset",
        required=True,
        help="versioned identifier, e.g. 'DANDI:000223@0.260528.0906'",
    )
    parser.add_argument(
        "--asset",
        help="asset id or path; required when the Dandiset holds several assets",
    )
    parser.add_argument("--neurodata-type", default="TwoPhotonSeries")
    parser.add_argument("--name", help="optional object-name filter")
    parser.add_argument(
        "--backend", choices=("auto", "remfile", "lindi"), default="remfile"
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--memory-limit", default="2 GiB")
    parser.add_argument(
        "--expect-axes",
        help=(
            "comma-separated axis names the inference must produce, e.g. "
            "'time,y,x'; a mismatch fails the run instead of hiding a semantic "
            "surprise"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help=(
            "independent repeats, each in a fresh process with a fresh output "
            "root so peak RSS is a true high-water mark and nothing resumes; "
            "the record then reports median and min-max range plus a check "
            "that every repetition produced the identical checksum"
        ),
    )
    parser.add_argument(
        "--classification", choices=("current", "publication"), default="current"
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    environment = capture_environment()
    git = cast(dict[str, object], environment["git"])
    if args.classification == "publication" and git.get("dirty") is not False:
        parser.error("publication classification requires a clean Git tree")
    if not 1 <= args.frames <= 1024:
        parser.error("--frames must stay within 1..1024; this is a smoke test")
    if args.repetitions > 1:
        _run_repetitions(args)
        return

    started = time.perf_counter()
    source = neuroflow.open_dandi(args.dandiset, backend=args.backend)
    try:
        asset_count = len(source.assets())
        query = NWBQuery(
            asset=args.asset,
            neurodata_type=args.neurodata_type,
            name=args.name,
        )
        try:
            selected = source.select(query)
        except AmbiguousSelectionError as error:
            parser.error(f"{error}; available assets:\n{_list_assets(source)}")
        metadata = selected.metadata
        asset_path = next(
            (
                asset.path
                for asset in source.assets()
                if asset.asset_id == metadata.source.asset_id
            ),
            None,
        )
        axes = tuple(metadata.axes)
        if args.expect_axes is not None:
            expected = tuple(args.expect_axes.split(","))
            if axes != expected:
                raise SystemExit(
                    f"inferred axes {axes} do not match the expected {expected}; "
                    "this is a real semantic finding about the axis inference, "
                    "not a harness failure -- document it rather than adjusting "
                    "the expectation"
                )
        if "time" not in axes:
            parser.error("the selected object has no inferred time axis")
        time_size = metadata.shape[axes.index("time")]
        if args.frames > time_size:
            parser.error(f"--frames exceeds the {time_size}-frame time axis")

        bounded = selected.isel(time=slice(0, args.frames))
        movie = neuroflow.NeuroArray(source, bounded)
        projection = np.mean(movie, axis="time").astype(  # type: ignore[call-overload]
            np.float32
        )

        args.output_root.mkdir(parents=True, exist_ok=True)
        safe_name = args.dandiset.replace(":", "-").replace("@", "-")
        output = args.output_root / f"dandi-smoke-{safe_name}-t{args.frames}.zarr"
        if output.exists():
            parser.error(
                f"{output} already exists; a resumed run would report a wall "
                "time that no longer measures this dataset"
            )

        # Metadata-only preflight, retained separately from execution so the
        # record shows what was promised before any data moved.
        plan = projection.plan(output, memory_limit=args.memory_limit)

        persist_started = time.perf_counter()
        persisted = projection.persist(output, memory_limit=args.memory_limit)
        persist_seconds = time.perf_counter() - persist_started
        workflow = persisted.workflow
        assert workflow is not None
        verified = bool(workflow.verify().valid)
        metrics = (workflow.provenance or {}).get("execution_metrics", {})
        stats = source.io_stats()
        # Snapshot before the independent reference computation below, which
        # deliberately loads the raw frames into memory and would otherwise be
        # indistinguishable from engine residency in the process high-water
        # mark.
        engine_phase_peak_rss = peak_rss_bytes()
        persisted.close()
    finally:
        source.close()

    result_source, result_selection = neuroflow.open_array(output)
    try:
        persisted_values = np.asarray(result_selection.as_dask_array().compute())
    finally:
        result_source.close()

    reference = _direct_reference(
        str(metadata.source.asset_id),
        metadata.path,
        args.frames,
        persisted_values.dtype,
    )
    exact = bool(np.array_equal(persisted_values, reference))
    absolute = float(np.max(np.abs(persisted_values - reference)))
    scale = np.abs(reference)
    nonzero = scale > 0.0
    relative = (
        float(
            np.max(
                np.abs(persisted_values - reference)[nonzero] / scale[nonzero]
            )
        )
        if bool(np.any(nonzero))
        else 0.0
    )
    tolerance = 1e-6
    selected_dtype = np.dtype(metadata.dtype)
    record = benchmark_record(
        name="dandi-second-dataset-smoke",
        classification=args.classification,
        backend=f"nwb-hdf5 {args.backend}",
        source={
            "dataset_identifier": args.dandiset.split("@", 1)[0],
            "dataset_version": args.dandiset.split("@", 1)[1]
            if "@" in args.dandiset
            else None,
            "asset": asset_path,
            "asset_id": metadata.source.asset_id,
            "asset_count_in_dandiset": asset_count,
            "path": metadata.path,
            "object_name": metadata.name,
            "neurodata_type": metadata.neurodata_type,
            "discovery": {
                "mechanism": "NWBQuery through the ordinary public select()",
                "neurodata_type": args.neurodata_type,
                "name": args.name,
                "asset": args.asset,
            },
            "inferred_axes": list(axes),
            "expected_axes": (
                args.expect_axes.split(",") if args.expect_axes else None
            ),
            "shape": list(metadata.shape),
            "dtype": str(selected_dtype),
            "physical_chunk_shape": (
                list(metadata.native_chunks)
                if metadata.native_chunks is not None
                else None
            ),
            "total_logical_bytes": int(np.prod(metadata.shape))
            * selected_dtype.itemsize,
            "selected_shape": list(bounded.metadata.shape),
            "selected_bytes": int(np.prod(bounded.metadata.shape))
            * selected_dtype.itemsize,
            "selection": {"frames": args.frames},
        },
        execution={
            "memory_budget": args.memory_limit,
            "plan": plan.to_dict(),
            "task_count": plan.task_count,
            "partition_configuration": {
                "processing_shape": list(plan.processing_partition_shape),
                "chosen_by": (
                    "planner, from the memory target alone; the command line "
                    "sets no tile, chunk, block, cache, or worker parameters"
                ),
            },
            "estimated_total_bytes_read": plan.estimated_total_bytes_read,
            "bytes_read": stats.get("response_content_bytes"),
            "bytes_read_status": (
                "counted from HTTP Content-Length headers"
                if isinstance(stats.get("response_content_bytes"), int)
                else "unknown: this transport exposes no byte counter"
            ),
            "http_responses": stats.get("http_responses"),
            # Engine phase only: sampled after persist+verify but before the
            # direct reference loads raw frames into this same process.
            "engine_phase_peak_rss_bytes": engine_phase_peak_rss,
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_rss_note": (
                "whole-process high-water mark including the harness's own "
                "direct-reference computation; engine_phase_peak_rss_bytes is "
                "the engine-attributable figure"
            ),
            "wall_time_seconds": time.perf_counter() - started,
            "persist_seconds": persist_seconds,
            "execution_metrics": metrics if isinstance(metrics, dict) else {},
            "cache_state": "cold; fresh process, no prior reads of this asset",
            "network_context": "public DANDI HTTPS; runner location is external",
        },
        result={
            "output": str(output),
            "output_shape": list(persisted_values.shape),
            "output_dtype": str(persisted_values.dtype),
            "output_bytes": _tree_size(output),
            "checksum": hashlib.sha256(
                np.ascontiguousarray(persisted_values).tobytes()
            ).hexdigest(),
            "integrity_verified": verified,
            "resume": {"supported": True, "exercised": False},
            "numerical_validation": {
                "valid": bool(absolute <= tolerance and relative <= tolerance),
                "comparison": (
                    "plain h5py+remfile read of the same frames reduced with "
                    "plain NumPy, sharing only the transport library"
                ),
                "elementwise_equal": exact,
                "maximum_absolute_error": absolute,
                "maximum_relative_error": relative,
                "atol": tolerance,
                "rtol": tolerance,
            },
        },
        notes=[
            "This is a generality smoke test: an unfamiliar real dataset with "
            "a different dimensionality, dtype, and native chunk geometry than "
            "the retained fish evidence, driven entirely through the public "
            "discovery and persistence API with no dataset-specific code.",
            "Output chunking, partitioning, and concurrency are chosen by the "
            "planner from the memory target alone; the command line sets no "
            "tile, chunk, block, cache, or worker parameters.",
            "The direct reference reads the same frames without NeuroFlow, so "
            "the comparison does not test the engine against itself.",
            "Wall time includes remote metadata reads and is not a throughput "
            "measurement.",
        ],
    )
    write_benchmark_record(args.record, record)
    axes_text = ",".join(axes)
    print(
        f"axes={axes_text} verified={verified} exact={exact} "
        f"max_abs={absolute} max_rel={relative} "
        f"engine_peak_rss={engine_phase_peak_rss} "
        f"process_peak_rss={peak_rss_bytes()} record={args.record}"
    )


if __name__ == "__main__":
    main()
