"""Reproducible local projection benchmark; prints one JSON record."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import dask.array as da
import numpy as np
from hdmf_zarr import NWBZarrIO, ZarrDataIO
from pynwb import NWBFile, TimeSeries

import neuroflow
from neuroflow.selection import NWBQuery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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
        projection.close()
        movie.close()
        record = {
            "schema_version": "1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "command": {
                "frames": args.frames,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
            },
            "environment": {
                "machine": platform.machine(),
                "operating_system": platform.platform(),
                "python": sys.version.split()[0],
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("dask", "hdmf-zarr", "neuroflow", "numpy", "pynwb")
                },
            },
            "seed": args.seed,
            "shape": list(data.shape),
            "direct_numpy_seconds": direct_seconds,
            "direct_dask_seconds": direct_dask_seconds,
            "neuroflow_seconds": neuroflow_seconds,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "source_bytes": _tree_size(source_path),
            "result_bytes": _tree_size(result_path),
            "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
            "direct_dask_maximum_absolute_error": float(
                np.max(np.abs(dask_actual - expected))
            ),
            "verified": valid,
            "numerical_tolerance": {"atol": 0.0, "rtol": 0.0},
        }
    payload = json.dumps(record, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")


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
