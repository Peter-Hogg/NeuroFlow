"""Publication benchmark record construction and validation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from neuroflow.provenance import capture_environment

BENCHMARK_SCHEMA_VERSION = "1"
BENCHMARK_CLASSIFICATIONS = {"current", "historical", "publication"}


def benchmark_record(
    *,
    name: str,
    classification: str,
    backend: str,
    source: dict[str, object],
    execution: dict[str, object],
    result: dict[str, object],
    baselines: list[dict[str, object]] | None = None,
    notes: list[str] | None = None,
) -> dict[str, object]:
    """Create one complete record, retaining null for unavailable metrics."""
    record: dict[str, object] = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark_name": name,
        "classification": classification,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": capture_environment(),
        "backend": backend,
        "source": source,
        "execution": execution,
        "result": result,
        "baselines": baselines or [],
        "notes": notes or [],
    }
    validate_benchmark_record(record)
    return record


def validate_benchmark_record(record: object) -> None:
    """Raise ``ValueError`` when a publication record is incomplete."""
    if not isinstance(record, dict):
        raise ValueError("benchmark record must be a JSON object")
    required = {
        "benchmark_schema_version",
        "benchmark_name",
        "classification",
        "timestamp",
        "environment",
        "backend",
        "source",
        "execution",
        "result",
        "baselines",
        "notes",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError("benchmark record is missing: " + ", ".join(missing))
    if record.get("benchmark_schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark schema version")
    if record.get("classification") not in BENCHMARK_CLASSIFICATIONS:
        raise ValueError("benchmark classification is invalid")
    for key in ("benchmark_name", "timestamp", "backend"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"benchmark {key} must be a non-empty string")
    environment = _mapping(record["environment"], "environment")
    if "neuroflow_version" not in environment or "git" not in environment:
        raise ValueError("benchmark environment lacks software/Git identity")
    source = _mapping(record["source"], "source")
    for key in (
        "dataset_identifier",
        "dataset_version",
        "asset",
        "path",
        "shape",
        "dtype",
        "physical_chunk_shape",
        "total_logical_bytes",
        "selected_bytes",
    ):
        if key not in source:
            raise ValueError(f"benchmark source lacks {key}")
    execution = _mapping(record["execution"], "execution")
    for key in (
        "partition_configuration",
        "memory_budget",
        "task_count",
        "bytes_read",
        "peak_rss_bytes",
        "wall_time_seconds",
        "cache_state",
        "network_context",
    ):
        if key not in execution:
            raise ValueError(f"benchmark execution lacks {key}")
    result = _mapping(record["result"], "result")
    for key in (
        "checksum",
        "numerical_validation",
        "integrity_verified",
        "resume",
        "output_bytes",
    ):
        if key not in result:
            raise ValueError(f"benchmark result lacks {key}")
    if not isinstance(record["baselines"], list) or not isinstance(
        record["notes"], list
    ):
        raise ValueError("benchmark baselines and notes must be lists")
    try:
        json.dumps(record, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"benchmark record is not strict JSON: {exc}") from exc


def write_benchmark_record(path: Path, record: dict[str, object]) -> None:
    validate_benchmark_record(record)
    if path.is_symlink():
        raise ValueError("refusing to write a benchmark record through a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"benchmark {name} must be a JSON object")
    return value
