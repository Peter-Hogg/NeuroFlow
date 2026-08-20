import json

import pytest

from neuroflow.benchmarking import benchmark_record, validate_benchmark_record


def _record() -> dict[str, object]:
    return benchmark_record(
        name="deterministic-test",
        classification="current",
        backend="nwb-zarr",
        source={
            "dataset_identifier": "synthetic:test",
            "dataset_version": "1",
            "asset": None,
            "path": "/acquisition/movie",
            "shape": [4, 3, 2],
            "dtype": "float32",
            "physical_chunk_shape": [1, 3, 2],
            "total_logical_bytes": 96,
            "selected_bytes": 96,
        },
        execution={
            "partition_configuration": {"shape": [1, 3, 2]},
            "memory_budget": "64 MiB",
            "task_count": 4,
            "bytes_read": None,
            "peak_rss_bytes": 1,
            "wall_time_seconds": 0.1,
            "cache_state": "not-applicable",
            "network_context": None,
        },
        result={
            "checksum": "0" * 64,
            "numerical_validation": {"valid": True, "atol": 0, "rtol": 0},
            "integrity_verified": True,
            "resume": {"resumed_partitions": 0},
            "output_bytes": 24,
        },
    )


def test_benchmark_schema_is_strict_json_and_accepts_unknown_measurements() -> None:
    record = _record()
    validate_benchmark_record(json.loads(json.dumps(record, allow_nan=False)))


def test_benchmark_schema_rejects_missing_evidence() -> None:
    record = _record()
    del record["result"]
    with pytest.raises(ValueError, match="missing"):
        validate_benchmark_record(record)
