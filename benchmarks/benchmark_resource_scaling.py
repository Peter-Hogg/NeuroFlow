"""Measure planner behaviour and real RSS across memory/worker configurations.

The point of this benchmark is to check the claim that a user can state the
resources they have and get sensible partitioning and concurrency back, without
touching low-level knobs. For every configuration it records what the planner
predicted *and* what the process actually did, so the two can be compared
rather than conflated.

A local synthetic movie is used deliberately. Re-reading a 323 GB archive to
tune a scaling curve would cost hours per point and add network variance to a
measurement about memory. Plane geometry matches the fish asset
(888x2048 int16, one plane per source chunk), so the per-frame arithmetic that
drives window selection is the same arithmetic the archive run uses.

Each configuration runs in a fresh subprocess: peak RSS is a high-water mark
and cannot be reset within a process, so sequential configurations in one
interpreter would each inherit the largest earlier peak.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr

import neuroflow
from neuroflow.provenance import capture_environment
from neuroflow.source.array import ArraySource

# One (y, x) plane per source chunk, as in DANDI:000350.
PLANE_HEIGHT = 888
PLANE_WIDTH = 2048
DEFAULT_FRAMES = 384
DEFAULT_CELLS = 512

# (memory_limit, workers) pairs from the publication scaling matrix.
DEFAULT_CONFIGURATIONS: tuple[tuple[str, int], ...] = (
    ("2 GiB", 1),
    ("4 GiB", 1),
    ("4 GiB", 2),
    ("8 GiB", 2),
    ("8 GiB", 4),
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def build_fixture(
    root: Path, *, frames: int, cells: int
) -> tuple[Path, Path]:
    """Create (or reuse) a fish-shaped synthetic movie and label volume."""
    movie_path = root / f"scaling-movie-t{frames}.zarr"
    labels_path = root / f"scaling-labels-c{cells}.zarr"
    if not (movie_path / ".zgroup").exists():
        root.mkdir(parents=True, exist_ok=True)
        group = zarr.open_group(str(movie_path), mode="w")
        array = group.create_dataset(
            "movie",
            shape=(frames, PLANE_HEIGHT, PLANE_WIDTH, 1),
            chunks=(1, PLANE_HEIGHT, PLANE_WIDTH, 1),
            dtype="int16",
        )
        rng = np.random.default_rng(0)
        # Written frame by frame so building the fixture never needs the whole
        # movie resident; this harness must not itself be the memory hog.
        for index in range(frames):
            array[index, :, :, 0] = rng.integers(
                0, 4096, size=(PLANE_HEIGHT, PLANE_WIDTH), dtype=np.int16
            )
    if not (labels_path / ".zgroup").exists():
        labels = np.zeros((PLANE_HEIGHT, PLANE_WIDTH, 1), dtype=np.uint64)
        # Compact square ROIs on a grid, one label per cell.
        side = 8
        per_row = PLANE_WIDTH // (side * 2)
        for cell in range(cells):
            row = (cell // per_row) * side * 2
            column = (cell % per_row) * side * 2
            if row + side > PLANE_HEIGHT:
                break
            labels[row : row + side, column : column + side, 0] = cell + 1
        group = zarr.open_group(str(labels_path), mode="w")
        group.create_dataset(
            "labels", data=labels, chunks=(PLANE_HEIGHT, PLANE_WIDTH, 1)
        )
    return movie_path, labels_path


def _open(path: Path, component: str, axes: tuple[str, ...]) -> neuroflow.NeuroArray:
    source = ArraySource(path, component=component, axes=axes)
    return neuroflow.NeuroArray(source, source.select())


def run_configuration(
    *,
    movie_path: Path,
    labels_path: Path,
    output_root: Path,
    memory_limit: str,
    workers: int,
) -> dict[str, object]:
    """Run one (memory_limit, workers) point and report planned vs measured."""
    movie = _open(movie_path, "movie", ("time", "y", "x", "z"))
    labels = _open(labels_path, "labels", ("y", "x", "z"))
    record: dict[str, object] = {
        "memory_limit": memory_limit,
        "workers": workers,
    }
    try:
        plan = movie.plan_traces(labels, memory_limit=memory_limit)
        budget = plan.memory_budget
        record["trace_plan"] = {
            "time_window": plan.time_chunk,
            "automatic_time_chunk": plan.automatic_time_chunk,
            "task_count": plan.task_count,
            "cell_count": plan.cell_count,
            "planned_task_working_bytes": plan.estimated_memory_per_task,
            "planned_process_peak_bytes": (
                budget.reserved_bytes + plan.estimated_memory_per_task
            ),
            "budget": budget.to_dict(),
            "estimated_total_bytes_read": plan.estimated_total_bytes_read,
        }

        # Projection carries the worker dimension: trace extraction is
        # deliberately one partition at a time, so concurrency has to be
        # exercised through the graph-executed path to mean anything.
        tag = f"projection-{memory_limit}-{workers}".replace(" ", "")
        projection_output = output_root / tag
        projection_started = time.perf_counter()
        projection = np.median(movie, axis="time").astype(  # type: ignore[call-overload]
            np.float32
        )
        projection_result = projection.persist(
            projection_output,
            chunks=(256, 256, 1),
            max_workers=workers,
            memory_limit=memory_limit,
            mode="overwrite",
        )
        projection_seconds = time.perf_counter() - projection_started
        projection_plan = projection_result.workflow
        record["projection"] = {
            "requested_workers": workers,
            "granted_workers": (
                projection_plan.max_workers if projection_plan is not None else None
            ),
            "task_count": (
                projection_plan.plan.task_count if projection_plan is not None else None
            ),
            "planned_memory_per_task": (
                projection_plan.plan.memory_per_task
                if projection_plan is not None
                else None
            ),
            "wall_time_seconds": projection_seconds,
        }
        projection_result.close()

        trace_output = output_root / f"traces-{memory_limit}-{workers}".replace(" ", "")
        trace_started = time.perf_counter()
        traces = movie.extract_traces(
            labels, output=trace_output, memory_limit=memory_limit
        )
        trace_seconds = time.perf_counter() - trace_started
        metrics = neuroflow.open_result(trace_output).provenance.get(
            "execution_metrics", {}
        )
        metrics = metrics if isinstance(metrics, dict) else {}
        record["traces"] = {
            "wall_time_seconds": trace_seconds,
            "computed_task_count": metrics.get("computed_task_count"),
            "bytes_read": metrics.get("bytes_read"),
            "output_bytes": metrics.get("output_bytes"),
            "memory": metrics.get("memory"),
        }
        traces.close()
        record["measured_process_peak_rss_bytes"] = _peak_rss_bytes()
        record["status"] = "measured"
    finally:
        labels.close()
        movie.close()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--configuration",
        help="single 'memory_limit:workers' point; used for subprocess dispatch",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help=(
            "independent repeats of every configuration, each in a fresh "
            "process with a fresh output root; the record then reports "
            "median and min-max range instead of a single observation"
        ),
    )
    parser.add_argument(
        "--classification", choices=("current", "publication"), default="current"
    )
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    environment = capture_environment()
    git = cast(dict[str, object], environment["git"])
    if arguments.classification == "publication" and git.get("dirty") is not False:
        parser.error("publication classification requires a clean Git tree")

    movie_path, labels_path = build_fixture(
        arguments.fixture_root, frames=arguments.frames, cells=arguments.cells
    )
    arguments.output_root.mkdir(parents=True, exist_ok=True)

    if arguments.configuration is not None:
        limit, _, worker_text = arguments.configuration.partition(":")
        print(
            json.dumps(
                run_configuration(
                    movie_path=movie_path,
                    labels_path=labels_path,
                    output_root=arguments.output_root,
                    memory_limit=limit,
                    workers=int(worker_text),
                ),
                sort_keys=True,
            )
        )
        return

    def run_one(limit: str, workers: int, output_root: Path) -> dict[str, Any]:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.benchmark_resource_scaling",
                "--fixture-root",
                str(arguments.fixture_root),
                "--output-root",
                str(output_root),
                "--frames",
                str(arguments.frames),
                "--cells",
                str(arguments.cells),
                "--configuration",
                f"{limit}:{workers}",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        if process.returncode != 0:
            return {
                "memory_limit": limit,
                "workers": workers,
                "status": "failed",
                "returncode": process.returncode,
                "stderr": process.stderr[-3000:],
            }
        payload = None
        for line in process.stdout.splitlines():
            if line.strip().startswith("{"):
                payload = json.loads(line)
        return (
            payload
            if payload is not None
            else {
                "memory_limit": limit,
                "workers": workers,
                "status": "failed",
                "stderr": "no JSON line produced",
            }
        )

    results: list[dict[str, Any]] = []
    for limit, workers in DEFAULT_CONFIGURATIONS:
        if arguments.repetitions == 1:
            results.append(run_one(limit, workers, arguments.output_root))
            continue
        # Peak RSS is a high-water mark and resume would skip recomputation,
        # so every repetition gets a fresh process *and* a fresh output root:
        # rerunning into an existing store would measure a zero-compute
        # resume, not a repetition.
        runs = [
            run_one(limit, workers, arguments.output_root / f"rep{index}")
            for index in range(arguments.repetitions)
        ]
        measured = [run for run in runs if run.get("status") == "measured"]

        def spread(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            return {
                "median": float(statistics.median(values)),
                "min": float(min(values)),
                "max": float(max(values)),
            }

        results.append(
            {
                "memory_limit": limit,
                "workers": workers,
                "status": "measured" if len(measured) == len(runs) else "failed",
                "repetition_count": len(runs),
                "repetitions": runs,
                "summary": {
                    "measured_process_peak_rss_bytes": spread(
                        [
                            float(run["measured_process_peak_rss_bytes"])
                            for run in measured
                        ]
                    ),
                    "wall_time_seconds": spread(
                        [
                            float(run["projection"]["wall_time_seconds"])
                            + float(run["traces"]["wall_time_seconds"])
                            for run in measured
                        ]
                    ),
                },
            }
        )

    suite = {
        "suite_schema_version": "1",
        "suite_name": "resource-scaling",
        "classification": arguments.classification,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "independent_process_per_configuration": True,
        "repetitions_per_configuration": arguments.repetitions,
        "fixture": {
            "kind": "local synthetic",
            "reason": (
                "isolates memory behaviour from network variance and avoids "
                "re-reading a 323 GB archive per scaling point"
            ),
            "movie_shape": [arguments.frames, PLANE_HEIGHT, PLANE_WIDTH, 1],
            "source_chunk_shape": [1, PLANE_HEIGHT, PLANE_WIDTH, 1],
            "dtype": "int16",
            "requested_cells": arguments.cells,
        },
        "configurations": results,
        "notes": [
            "memory_limit is an approximate total process-memory target, not an "
            "enforced ceiling; no OS-level cap is installed.",
            "Trace extraction processes one time partition at a time by design, "
            "so the worker dimension is exercised through the graph-executed "
            "projection path.",
            "Peak RSS is a whole-process high-water mark for the configuration's "
            "own subprocess.",
        ],
    }
    if arguments.record is not None:
        arguments.record.parent.mkdir(parents=True, exist_ok=True)
        arguments.record.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
    print(json.dumps(suite, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
