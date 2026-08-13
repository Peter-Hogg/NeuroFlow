"""Run deterministic local benchmark repetitions and write a JSON summary."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/local-summary.json")
    )
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("use at least three repetitions")
    records: list[dict[str, object]] = []
    for _ in range(args.repetitions):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.benchmark_projection",
                "--frames",
                str(args.frames),
                "--height",
                str(args.height),
                "--width",
                str(args.width),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        records.append(json.loads(completed.stdout))
    timing_keys = (
        "direct_numpy_seconds",
        "direct_dask_seconds",
        "neuroflow_seconds",
    )
    summary = {
        "schema_version": "1",
        "repetitions": args.repetitions,
        "records": records,
        "median": {
            key: statistics.median(float(record[key]) for record in records)
            for key in timing_keys
        },
        "all_verified": all(bool(record["verified"]) for record in records),
        "maximum_absolute_error": max(
            float(record["maximum_absolute_error"]) for record in records
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
