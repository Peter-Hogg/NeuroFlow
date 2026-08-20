from __future__ import annotations

import json
from pathlib import Path

from neuroflow.benchmarking import benchmark_record
from publication.generate_tables import load_records


def _record(classification: str) -> dict[str, object]:
    return benchmark_record(
        name=f"test-{classification}",
        classification=classification,
        backend="local",
        source={
            "dataset_identifier": "synthetic:test",
            "dataset_version": "1",
            "asset": "array",
            "path": "/array",
            "shape": [1],
            "dtype": "float32",
            "physical_chunk_shape": [1],
            "total_logical_bytes": 4,
            "selected_bytes": 4,
        },
        execution={
            "partition_configuration": {},
            "memory_budget": None,
            "task_count": 1,
            "bytes_read": None,
            "peak_rss_bytes": 1,
            "wall_time_seconds": 0.0,
            "cache_state": "local",
            "network_context": None,
        },
        result={
            "checksum": "test",
            "numerical_validation": {"valid": True},
            "integrity_verified": True,
            "resume": {},
            "output_bytes": 4,
        },
    )


def test_paper_tables_exclude_current_development_records_by_default(
    tmp_path: Path,
) -> None:
    for classification in ("current", "publication"):
        path = tmp_path / f"{classification}.json"
        path.write_text(json.dumps(_record(classification)))

    records, skipped = load_records(tmp_path)

    assert [item["classification"] for item in records] == ["publication"]
    assert any(message.endswith(": current") for message in skipped)
    included, _ = load_records(tmp_path, include_current=True)
    assert {item["classification"] for item in included} == {
        "current",
        "publication",
    }
