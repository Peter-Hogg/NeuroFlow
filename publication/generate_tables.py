"""Generate paper-facing Markdown and CSV tables from retained benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neuroflow.benchmarking import validate_benchmark_record


def load_records(
    results: Path, *, include_current: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for path in sorted(results.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            candidates = payload.get("records", [payload])
            if not isinstance(candidates, list):
                raise ValueError("records must be a list")
            for candidate in candidates:
                validate_benchmark_record(candidate)
                classification = candidate["classification"]
                if classification != "publication" and not (
                    include_current and classification == "current"
                ):
                    skipped.append(f"{path}: {classification}")
                    continue
                records.append(candidate)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            skipped.append(f"{path}: {exc}")
    return records, skipped


def rows_for(records: Iterable[dict[str, Any]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        source = record["source"]
        execution = record["execution"]
        result = record["result"]
        validation = result["numerical_validation"]
        resume = result["resume"]
        rows.append(
            {
                "benchmark": record["benchmark_name"],
                "classification": record["classification"],
                "backend": record["backend"],
                "shape": "×".join(str(value) for value in source["shape"]),
                "selected_bytes": source["selected_bytes"],
                "tasks": execution["task_count"],
                "peak_rss_bytes": execution["peak_rss_bytes"],
                "wall_time_seconds": execution["wall_time_seconds"],
                "numerically_valid": validation.get("valid"),
                "integrity_verified": result["integrity_verified"],
                "resumed_partitions": resume.get(
                    "resumed_partitions",
                    resume.get("completed_before_interruption"),
                ),
                "git_sha": record["environment"]["git"].get("commit"),
            }
        )
    return rows


def write_tables(output: Path, rows: list[dict[str, object]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["benchmark"]
    with (output / "benchmark-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Benchmark summary", ""]
    if not rows:
        lines.append("No current publication-schema benchmark records were found.")
    else:
        lines.extend(
            [
                "| " + " | ".join(fields) + " |",
                "| " + " | ".join("---" for _ in fields) + " |",
            ]
        )
        for row in rows:
            lines.append(
                "| " + " | ".join(_display(row[field]) for field in fields) + " |"
            )
    (output / "benchmark-summary.md").write_text("\n".join(lines) + "\n")


def _display(value: object) -> str:
    return "unknown" if value is None else str(value).replace("|", "\\|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--output", type=Path, default=Path("publication/tables"))
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="include development records; never use this for final paper tables",
    )
    args = parser.parse_args()
    records, skipped = load_records(
        args.results, include_current=args.include_current
    )
    write_tables(args.output, rows_for(records))
    for message in skipped:
        print(f"SKIP {message}")
    print(f"WROTE {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
