"""Run isolated synthetic projection sizes and retain structured JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from neuroflow.benchmarking import validate_benchmark_record
from neuroflow.provenance import capture_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="128,256,512")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/synthetic-scaling.json"),
    )
    args = parser.parse_args()
    try:
        sizes = [int(item) for item in args.sizes.split(",")]
    except ValueError as exc:
        parser.error(f"--sizes must be comma-separated integers: {exc}")
    if not sizes or any(size < 8 for size in sizes):
        parser.error("all sizes must be at least 8")
    if args.frames < 2 or args.repetitions < 1:
        parser.error("frames must be at least 2 and repetitions must be positive")

    records: list[dict[str, object]] = []
    for size in sizes:
        for _ in range(args.repetitions):
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.benchmark_projection",
                    "--frames",
                    str(args.frames),
                    "--height",
                    str(size),
                    "--width",
                    str(size),
                    "--classification",
                    "publication",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            record = json.loads(completed.stdout)
            validate_benchmark_record(record)
            records.append(record)
    suite = {
        "suite_schema_version": "1",
        "suite_name": "synthetic-bounded-memory-scaling",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": capture_environment(),
        "independent_process_per_record": True,
        "records": records,
        "notes": [
            "Peak RSS is a process high-water mark and includes Python/import "
            "overhead.",
            "Compare peak RSS against selected bytes and partition configuration; "
            "do not infer a hard operating-system memory cap.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    print(json.dumps(suite, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
