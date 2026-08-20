"""Current-engine DANDI fish benchmark with publication-schema output."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

import neuroflow
from benchmarks.benchmark_projection import _tree_size
from examples.dandi_fish_projection import (
    ASSET_PATH,
    DANDISET,
    MOVIE_SHAPE,
    OBJECT_NAME,
    FishProjectionConfig,
    run_example,
)
from neuroflow.benchmarking import benchmark_record, write_benchmark_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--tile-y", type=int, default=256)
    parser.add_argument("--tile-x", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--cache-size-mib", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=262_144)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    config = FishProjectionConfig(
        frames=args.frames,
        tile_y=args.tile_y,
        tile_x=args.tile_x,
        max_workers=args.max_workers,
        cache_size=args.cache_size_mib * 1024 * 1024,
        block_size=args.block_size,
        output=args.result,
        preview=args.preview,
    )
    started = time.perf_counter()
    summary = run_example(config)
    wall_time = time.perf_counter() - started
    persisted = neuroflow.open_result(args.result)
    checksum = persisted.array_source_identity(verify_checksums=False)
    provenance = persisted.provenance
    metrics = provenance.get("execution_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    remote_io = summary.get("remote_io", {})
    if not isinstance(remote_io, dict):
        remote_io = {}
    dtype = np.dtype(str(summary["input_dtype"]))
    selected_shape_value = summary["input_shape"]
    native_chunks_value = summary["native_chunks"]
    output_chunks_value = summary["output_chunks"]
    if not isinstance(selected_shape_value, (list, tuple)):
        raise TypeError("archive summary input_shape must be a sequence")
    if not isinstance(native_chunks_value, (list, tuple)):
        raise TypeError("archive summary native_chunks must be a sequence")
    if not isinstance(output_chunks_value, (list, tuple)):
        raise TypeError("archive summary output_chunks must be a sequence")
    selected_shape = tuple(int(item) for item in selected_shape_value)
    record = benchmark_record(
        name="dandi-fish-temporal-median",
        classification="publication",
        backend="dandi-nwb-hdf5-remfile",
        source={
            "dataset_identifier": DANDISET.split("@", 1)[0],
            "dataset_version": DANDISET.split("@", 1)[1],
            "asset": ASSET_PATH,
            "path": f"/acquisition/{OBJECT_NAME}",
            "shape": list(selected_shape),
            "dtype": str(dtype),
            "physical_chunk_shape": list(native_chunks_value),
            "total_logical_bytes": int(np.prod(MOVIE_SHAPE)) * dtype.itemsize,
            "selected_bytes": int(np.prod(selected_shape)) * dtype.itemsize,
        },
        execution={
            "partition_configuration": {
                "output_chunks": list(output_chunks_value),
                "native_chunks_touched": summary["native_chunks_touched"],
            },
            "memory_budget": "2 GiB",
            "task_count": summary["task_count"],
            "bytes_read": remote_io.get("response_content_bytes"),
            "peak_rss_bytes": metrics.get("peak_rss_bytes")
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "wall_time_seconds": wall_time,
            "cache_state": (
                "bounded-remfile-cache; freshness must be recorded by runner"
            ),
            "network_context": "public DANDI HTTPS; record location/link separately",
        },
        result={
            "checksum": checksum,
            "numerical_validation": {
                "valid": None,
                "status": "requires independent retained reference comparison",
                "atol": None,
                "rtol": None,
            },
            "integrity_verified": bool(summary["verified"]),
            "resume": {
                "resumed_partitions": metrics.get("resumed_task_count"),
            },
            "output_bytes": _tree_size(args.result),
        },
        notes=[
            "This script uses the current NumPy-expression engine.",
            "Run once with a fresh output and retain it; do not repeatedly load the "
            "public archive for favorable timings.",
            "Scientific interpretation is not established by this software benchmark.",
        ],
    )
    write_benchmark_record(args.record, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
