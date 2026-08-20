"""Deterministic interruption, resume, corruption, and repair benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import zarr

import neuroflow
from benchmarks.benchmark_projection import _tree_size, _write_source
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.benchmarking import benchmark_record, write_benchmark_record
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ZarrOutput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/resume-integrity.json"),
    )
    args = parser.parse_args()
    data = np.arange(12 * 8 * 8, dtype=np.float32).reshape(12, 8, 8)
    with tempfile.TemporaryDirectory(prefix="neuroflow-resume-benchmark-") as tmp:
        root = Path(tmp)
        source_path = root / "source.nwb.zarr"
        output_path = root / "result.zarr"
        _write_source(source_path, data)
        source = neuroflow.open_source(source_path)
        selection = source.select(NWBQuery(name="movie"))
        state = {"calls": 0, "interrupt": True}

        def interrupt_once(value: np.ndarray) -> np.ndarray:
            state["calls"] += 1
            if state["interrupt"] and state["calls"] == 2:
                raise RuntimeError("intentional benchmark interruption")
            return value * np.float32(2)

        result = neuroflow.run(
            source=source,
            selection=selection,
            adapter=FunctionAdapter(
                function=interrupt_once,
                input_kind="array",
                output=ArrayOutput("float32"),
                name="resume-integrity-transform",
                version="1",
                splittable_axes=("time",),
            ),
            partition=TimeWindowPlan(size=4),
            output=ZarrOutput(str(output_path)),
            max_workers=1,
        )
        started = time.perf_counter()
        try:
            result.execute()
        except RuntimeError as exc:
            if "intentional benchmark interruption" not in str(exc):
                raise
        completed_before_resume = len(result.status.completed_partitions)
        state["interrupt"] = False
        calls_before_resume = state["calls"]
        result.resume()
        calls_during_resume = state["calls"] - calls_before_resume
        expected = data * np.float32(2)
        actual = result.arrays["result"].as_dask_array().compute()
        np.testing.assert_array_equal(actual, expected)
        valid_after_resume = result.verify().valid

        first_partition = result.plan.partitions[0]
        stored = zarr.open_group(str(output_path), mode="a")
        stored["result"][first_partition.output_slices] = np.float32(-1)
        corruption_detected = not result.verify().valid
        calls_before_repair = state["calls"]
        result.resume()
        repair_calls = state["calls"] - calls_before_repair
        repaired = result.arrays["result"].as_dask_array().compute()
        valid_after_repair = result.verify().valid
        wall_time = time.perf_counter() - started
        provenance = result.provenance or {}
        metrics = provenance.get("execution_metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        source.close()

        maximum_error = float(np.max(np.abs(repaired - expected)))
        record = benchmark_record(
            name="resume-integrity",
            classification="publication",
            backend="nwb-zarr",
            source={
                "dataset_identifier": "synthetic:resume-integrity",
                "dataset_version": "1",
                "asset": source_path.name,
                "path": "/acquisition/movie",
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "physical_chunk_shape": [1, 8, 8],
                "total_logical_bytes": int(data.nbytes),
                "selected_bytes": int(data.nbytes),
            },
            execution={
                "partition_configuration": {"time_window": 4},
                "memory_budget": None,
                "task_count": result.plan.task_count,
                "bytes_read": None,
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                "wall_time_seconds": wall_time,
                "cache_state": "local-temporary-source",
                "network_context": None,
            },
            result={
                "checksum": hashlib.sha256(repaired.tobytes()).hexdigest(),
                "numerical_validation": {
                    "valid": maximum_error == 0,
                    "maximum_absolute_error": maximum_error,
                    "atol": 0.0,
                    "rtol": 0.0,
                },
                "integrity_verified": valid_after_repair,
                "resume": {
                    "completed_before_interruption": completed_before_resume,
                    "calls_during_resume": calls_during_resume,
                    "valid_after_resume": valid_after_resume,
                    "corruption_detected": corruption_detected,
                    "repair_calls": repair_calls,
                    "computed_task_count_after_repair": metrics.get(
                        "computed_task_count"
                    ),
                },
                "output_bytes": _tree_size(output_path),
            },
            notes=[
                "The adapter intentionally raises on its second bounded call.",
                "One persisted partition is then modified and checksum repair is run.",
            ],
        )
    write_benchmark_record(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
