"""Reproducible local projection benchmark; prints one JSON record."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import numpy as np
from hdmf_zarr import NWBZarrIO, ZarrDataIO
from pynwb import NWBFile, TimeSeries

import neuroflow
from neuroflow.benchmarking import (
    benchmark_record,
    peak_rss_bytes,
    write_benchmark_record,
)
from neuroflow.provenance import capture_environment
from neuroflow.selection import NWBQuery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--classification",
        choices=("current", "publication"),
        default="current",
    )
    args = parser.parse_args()
    environment = capture_environment()
    git = environment.get("git", {})
    if (
        args.classification == "publication"
        and isinstance(git, dict)
        and git.get("dirty") is not False
    ):
        parser.error("publication classification requires a clean Git tree")
    if min(args.frames, args.height, args.width) < 1:
        parser.error("all dimensions must be positive")
    rng = np.random.default_rng(args.seed)
    data = rng.normal(size=(args.frames, args.height, args.width)).astype("float32")
    with tempfile.TemporaryDirectory(prefix="neuroflow-benchmark-") as directory:
        root = Path(directory)
        source_path = root / "benchmark.nwb.zarr"
        result_path = root / "projection.zarr"
        _write_source(source_path, data)

        direct_start = time.perf_counter()
        expected = np.median(data, axis=0)
        direct_seconds = time.perf_counter() - direct_start

        dask_source = neuroflow.open_source(source_path)
        dask_selection = dask_source.select(NWBQuery(name="movie"))
        dask_start = time.perf_counter()
        dask_actual = da.median(
            dask_selection.as_dask_array(chunks="native"), axis=0
        ).compute()
        direct_dask_seconds = time.perf_counter() - dask_start
        dask_source.close()

        start = time.perf_counter()
        movie = neuroflow.load(source_path, name="movie")
        projection = movie.median(
            "time",
            output=result_path,
            chunks=(128, 128),
            max_workers=1,
            memory_limit="1 GiB",
        )
        actual = projection.compute()
        neuroflow_seconds = time.perf_counter() - start
        valid = bool(projection.workflow.verify().valid)
        plan = projection.workflow.plan
        provenance = projection.workflow.provenance or {}
        projection.close()
        movie.close()
        output_bytes = _tree_size(result_path)
        maximum_error = float(np.max(np.abs(actual - expected)))
        dask_error = float(np.max(np.abs(dask_actual - expected)))
        peak_rss = peak_rss_bytes()
        execution_metrics = provenance.get("execution_metrics", {})
        if not isinstance(execution_metrics, dict):
            execution_metrics = {}
        record = benchmark_record(
            name="local-projection-correctness",
            classification=args.classification,
            backend="nwb-zarr",
            source={
                "dataset_identifier": "synthetic:normal-projection",
                "dataset_version": "1",
                "asset": str(source_path.name),
                "path": "/acquisition/movie",
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "physical_chunk_shape": [1, args.height, args.width],
                "total_logical_bytes": int(data.nbytes),
                "selected_bytes": int(data.nbytes),
            },
            execution={
                "partition_configuration": {
                    "processing_shape": list(plan.processing_partition_shape),
                    "output_chunks": [128, 128],
                },
                "memory_budget": "1 GiB",
                "task_count": plan.task_count,
                "bytes_read": None,
                "peak_rss_bytes": peak_rss,
                "wall_time_seconds": neuroflow_seconds,
                "cache_state": "local-temporary-source",
                "network_context": None,
            },
            result={
                "checksum": hashlib.sha256(actual.tobytes()).hexdigest(),
                "numerical_validation": {
                    "valid": maximum_error == 0,
                    "maximum_absolute_error": maximum_error,
                    "maximum_relative_error": 0.0,
                    "atol": 0.0,
                    "rtol": 0.0,
                    "repeatability": "deterministic seed; single run in this record",
                },
                "integrity_verified": valid,
                "resume": {
                    "exercised": False,
                    "resumed_partitions": execution_metrics.get(
                        "resumed_task_count", 0
                    ),
                },
                "output_bytes": output_bytes,
            },
            baselines=[
                {
                    "name": "direct-numpy",
                    "version": np.__version__,
                    "wall_time_seconds": direct_seconds,
                    "maximum_absolute_error": 0.0,
                    "cache_state": "in-memory-generated-input",
                },
                {
                    "name": "direct-dask",
                    "version": da.__version__ if hasattr(da, "__version__") else None,
                    "wall_time_seconds": direct_dask_seconds,
                    "maximum_absolute_error": dask_error,
                    "cache_state": "local-temporary-source",
                },
            ],
            notes=[
                "Peak RSS is the process high-water mark across all three methods.",
                "This small local run assesses correctness and overhead, not "
                "remote I/O.",
            ],
        )
        # Retain version-1 flat timing keys for the existing repetition summarizer.
        record.update(
            {
                "schema_version": "2",
                "command": {
                    "frames": args.frames,
                    "height": args.height,
                    "width": args.width,
                    "seed": args.seed,
                },
                "direct_numpy_seconds": direct_seconds,
                "direct_dask_seconds": direct_dask_seconds,
                "neuroflow_seconds": neuroflow_seconds,
                "maximum_absolute_error": maximum_error,
                "verified": valid,
            }
        )
    payload = json.dumps(record, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        write_benchmark_record(args.output, record)


def _write_source(path: Path, data: np.ndarray) -> None:
    nwb = NWBFile(
        session_description="NeuroFlow benchmark",
        identifier="benchmark",
        session_start_time=datetime.now(timezone.utc),
    )
    nwb.add_acquisition(
        TimeSeries(
            name="movie",
            data=ZarrDataIO(data, chunks=(1, data.shape[1], data.shape[2])),
            unit="a.u.",
            rate=2.0,
        )
    )
    with NWBZarrIO(path, mode="w") as io:
        io.write(nwb)


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


if __name__ == "__main__":
    main()
