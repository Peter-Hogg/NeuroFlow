"""Attribute whole-process peak RSS to named components.

Each component runs in a fresh subprocess so that its resident-set cost is
measured against a clean interpreter rather than against whatever the previous
step left allocated. ``--component`` runs exactly one probe and prints a JSON
line; without it this module drives every probe and writes an attribution
record.

The probes deliberately mirror the fish pipeline: the same plane geometry, the
same remfile cache size, the same Cellpose model, and the same trace window
arithmetic. That makes the resulting table comparable with
``benchmark_fish_pipeline`` peak RSS rather than an abstract microbenchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Fish geometry: one source spatial chunk is a full (y, x) plane of one z index.
PLANE_SHAPE = (888, 2048)
PLANE_ELEMENTS = math.prod(PLANE_SHAPE)
PLANES = 29
CELL_COUNT = 5557
TIME_WINDOW = 106
REMFILE_CACHE_BYTES = 67_108_864
REMOTE_URL = (
    "https://api.dandiarchive.org/api/assets/"
    "4f898ff7-6084-4e84-a449-f05811c1d951/download/"
)


# Probes append here so their allocations stay reachable while resident set
# size is sampled. ``run_component`` clears it between probes.
KEEPALIVE: list[object] = []


def current_rss_bytes() -> int:
    """Read the live resident set size from ``/proc`` when available."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return peak_rss_bytes()


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _probe_baseline() -> dict[str, object]:
    """Bare CPython plus this module's own numpy import."""
    return {"detail": "interpreter, numpy, stdlib only"}


def _probe_neuroflow_import() -> dict[str, object]:
    import neuroflow  # noqa: F401

    return {"detail": "neuroflow package import chain (dask, zarr, h5py, pynwb)"}


def _probe_dask_runtime() -> dict[str, object]:
    import dask.array as da

    lazy = da.zeros(
        (TIME_WINDOW, *PLANE_SHAPE), chunks=(1, *PLANE_SHAPE), dtype="int16"
    )
    graph_keys = len(lazy.__dask_graph__())
    total = da.stack([lazy for _ in range(4)]).sum()
    KEEPALIVE.extend((lazy, total))
    return {
        "detail": "dask.array import plus a 4x stacked lazy graph, never computed",
        "graph_keys": graph_keys,
        "stacked_keys": len(total.__dask_graph__()),
    }


def _probe_remfile_cache() -> dict[str, object]:
    """Open the real asset and read HDF5 metadata with the pipeline cache size."""
    import h5py
    import remfile

    remote = remfile.File(
        REMOTE_URL,
        _min_chunk_size=262_144,
        _max_cache_size=REMFILE_CACHE_BYTES,
    )
    handle = h5py.File(remote, "r")
    dataset = handle["/acquisition/NeuronOnePhotonSeries/data"]
    shape = tuple(int(value) for value in dataset.shape)  # type: ignore[union-attr]
    chunks = getattr(dataset, "chunks", None)
    # One native chunk read exercises the cache on a realistic access.
    block = np.asarray(dataset[0:1, :, :, 0:1])  # type: ignore[index]
    KEEPALIVE.extend((remote, handle, block))
    return {
        "detail": "remfile.File + h5py metadata + one native chunk read",
        "dataset_shape": list(shape),
        "native_chunks": list(chunks) if chunks else None,
        "cache_size_bytes": REMFILE_CACHE_BYTES,
        "block_bytes": int(block.nbytes),
    }


def _probe_source_partition() -> dict[str, object]:
    """One materialised int16 source window, as a trace task loads it.

    ``np.full`` rather than ``np.zeros`` so every page is really faulted in;
    a lazily zero-filled allocation would under-report resident set size.
    """
    block = np.full((TIME_WINDOW, *PLANE_SHAPE), 7, dtype=np.int16)
    KEEPALIVE.append(block)
    return {
        "detail": (
            f"one int16 ({TIME_WINDOW}, {PLANE_SHAPE[0]}, {PLANE_SHAPE[1]}) window"
        ),
        "array_bytes": int(block.nbytes),
    }


def _probe_temporary_arrays() -> dict[str, object]:
    """The float32 cast and the moveaxis/reshape copy that follow the read."""
    block = np.full((TIME_WINDOW, *PLANE_SHAPE), 7, dtype=np.int16)
    as_float = np.asarray(block, dtype=np.float32)
    flattened = np.moveaxis(as_float, 0, 0).reshape(TIME_WINDOW, -1)
    # ``reshape`` on a contiguous array is a view; force the worst case that the
    # planner budgets for by copying, matching a non-leading time axis.
    copied = np.ascontiguousarray(flattened)
    copied[0, 0] = 1.0
    KEEPALIVE.extend((block, as_float, copied))
    return {
        "detail": "int16 window + float32 cast + contiguous reshape copy",
        "source_bytes": int(block.nbytes),
        "float_bytes": int(as_float.nbytes),
        "copy_bytes": int(copied.nbytes),
    }


def _probe_roi_index() -> dict[str, object]:
    """The label plane, its np.unique workspace, and the Python ROI mapping."""
    labels = np.zeros(PLANE_SHAPE, dtype=np.uint64)
    labels.reshape(-1)[: CELL_COUNT * 20] = np.repeat(
        np.arange(1, CELL_COUNT + 1, dtype=np.uint64), 20
    )
    unique, counts = np.unique(labels, return_counts=True)
    counts_by_id = {int(key): int(value) for key, value in zip(unique, counts)}
    roi_chunks = [
        ((slice(0, PLANE_SHAPE[0]), slice(0, PLANE_SHAPE[1]), slice(index, index + 1)),
         tuple(range(1, CELL_COUNT // PLANES + 1)))
        for index in range(PLANES)
    ]
    KEEPALIVE.extend((labels, counts_by_id, roi_chunks))
    return {
        "detail": "uint64 label plane, np.unique workspace, ROI chunk index",
        "label_bytes": int(labels.nbytes),
        "distinct_labels": len(counts_by_id),
        "roi_chunk_count": len(roi_chunks),
    }


def _probe_accumulators() -> dict[str, object]:
    """float64 sums plus the float32 output block for one time window."""
    sums = np.full((TIME_WINDOW, CELL_COUNT), 1.0, dtype=np.float64)
    values = (sums / np.ones(CELL_COUNT, dtype=np.int64)[None, :]).astype(np.float32)
    KEEPALIVE.extend((sums, values))
    return {
        "detail": "float64 accumulator + float32 output block for one window",
        "accumulator_bytes": int(sums.nbytes),
        "output_bytes": int(values.nbytes),
    }


def _probe_output_buffer() -> dict[str, object]:
    """A zarr store plus one written window, matching the trace output."""
    import tempfile

    import zarr

    with tempfile.TemporaryDirectory() as directory:
        root = zarr.open_group(directory, mode="a")
        traces = root.create_dataset(
            "traces",
            shape=(3065, CELL_COUNT),
            chunks=(TIME_WINDOW, min(1024, CELL_COUNT)),
            dtype="float32",
            fill_value=np.nan,
        )
        window = np.ones((TIME_WINDOW, CELL_COUNT), dtype=np.float32)
        traces[0:TIME_WINDOW, :] = window
        read_back = np.asarray(traces[0:TIME_WINDOW, :])
    return {
        "detail": "zarr group, one written window, one verification read",
        "written_bytes": int(window.nbytes),
        "read_bytes": int(read_back.nbytes),
    }


def _probe_torch_import() -> dict[str, object]:
    import torch

    KEEPALIVE.append(torch)
    return {
        "detail": "torch import only",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _probe_cellpose_model(device: str) -> dict[str, object]:
    import importlib
    import importlib.metadata

    import torch

    models = importlib.import_module("cellpose.models")
    use_gpu = device == "cuda"
    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model="cpsam",
        use_bfloat16=False,
    )
    parameter_bytes = sum(
        value.numel() * value.element_size() for value in model.net.parameters()
    )
    KEEPALIVE.append(model)
    return {
        "detail": f"cellpose import + CellposeModel(cpsam) on {device}",
        "cellpose_version": importlib.metadata.version("cellpose"),
        "device": device,
        "model_parameter_bytes": int(parameter_bytes),
        "torch_cuda_allocated_bytes": (
            int(torch.cuda.memory_allocated()) if use_gpu else 0
        ),
    }


def _probe_cellpose_eval(device: str) -> dict[str, object]:
    import importlib
    import time

    import torch

    models = importlib.import_module("cellpose.models")
    use_gpu = device == "cuda"
    model = models.CellposeModel(
        gpu=use_gpu,
        pretrained_model="cpsam",
        use_bfloat16=False,
    )
    rng = np.random.default_rng(0)
    plane = rng.normal(1000.0, 120.0, size=PLANE_SHAPE).astype(np.float32)
    started = time.perf_counter()
    evaluation = model.eval(
        plane,
        batch_size=1,
        channels=None,
        channel_axis=None,
        z_axis=None,
        normalize=True,
        diameter=None,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        do_3D=False,
        anisotropy=None,
        min_size=15,
        tile_overlap=0.1,
    )
    elapsed = time.perf_counter() - started
    labels = np.asarray(evaluation[0])
    KEEPALIVE.extend((model, plane, labels))
    return {
        "detail": f"one {PLANE_SHAPE[0]}x{PLANE_SHAPE[1]} cpsam eval on {device}",
        "device": device,
        "eval_seconds": elapsed,
        "object_count": int(np.count_nonzero(np.unique(labels))),
        "torch_cuda_peak_bytes": (
            int(torch.cuda.max_memory_allocated()) if use_gpu else 0
        ),
    }


PROBES: dict[str, Callable[[], dict[str, object]]] = {
    "process_baseline": _probe_baseline,
    "neuroflow_import": _probe_neuroflow_import,
    "dask_runtime": _probe_dask_runtime,
    "remfile_cache": _probe_remfile_cache,
    "source_partition_array": _probe_source_partition,
    "temporary_numpy_arrays": _probe_temporary_arrays,
    "roi_index_state": _probe_roi_index,
    "trace_accumulators": _probe_accumulators,
    "output_buffers": _probe_output_buffer,
    "torch_import": _probe_torch_import,
    "cellpose_model_cpu": lambda: _probe_cellpose_model("cpu"),
    "cellpose_eval_cpu": lambda: _probe_cellpose_eval("cpu"),
    "cellpose_model_cuda": lambda: _probe_cellpose_model("cuda"),
    "cellpose_eval_cuda": lambda: _probe_cellpose_eval("cuda"),
}

LOCAL_PROBES = tuple(
    name for name in PROBES if name not in {"remfile_cache"} and "cuda" not in name
)


def run_component(name: str) -> dict[str, object]:
    """Measure one probe, reading resident set size before it releases anything.

    Probes publish the objects whose cost is being attributed through
    ``KEEPALIVE``. Reading ``VmRSS`` only after the probe function returned
    would measure the residue left once its locals were collected, which
    understates large allocations such as a loaded Cellpose network.
    """
    probe = PROBES[name]
    KEEPALIVE.clear()
    before = current_rss_bytes()
    detail = probe()
    after = current_rss_bytes()
    peak = peak_rss_bytes()
    KEEPALIVE.clear()
    return {
        "component": name,
        "rss_before_bytes": before,
        "rss_after_bytes": after,
        "rss_delta_bytes": after - before,
        "process_peak_rss_bytes": peak,
        **detail,
    }


def _spawn(name: str) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "benchmarks.memory_attribution", "--component", name],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    if process.returncode != 0:
        return {
            "component": name,
            "status": "failed",
            "returncode": process.returncode,
            "stderr": process.stderr[-2000:],
        }
    payload: dict[str, Any] | None = None
    for line in process.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
    if payload is None:
        return {
            "component": name,
            "status": "failed",
            "returncode": 0,
            "stderr": "probe produced no JSON line",
        }
    payload["status"] = "measured"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=sorted(PROBES))
    parser.add_argument("--record", type=Path)
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="probe names to run; defaults to every local (offline, CPU) probe",
    )
    arguments = parser.parse_args()
    if arguments.component is not None:
        print(json.dumps(run_component(arguments.component), sort_keys=True))
        return
    names = list(arguments.include) if arguments.include else list(LOCAL_PROBES)
    unknown = sorted(set(names) - set(PROBES))
    if unknown:
        parser.error(f"unknown probes: {', '.join(unknown)}")
    results = [_spawn(name) for name in names]
    record = {
        "schema_version": "1",
        "measurement": "component-resident-set-attribution",
        "method": (
            "each component runs in a fresh interpreter; rss_delta_bytes is the "
            "VmRSS increase caused by that component alone"
        ),
        "geometry": {
            "plane_shape": list(PLANE_SHAPE),
            "planes": PLANES,
            "cell_count": CELL_COUNT,
            "time_window": TIME_WINDOW,
        },
        "components": results,
    }
    if arguments.record is not None:
        arguments.record.parent.mkdir(parents=True, exist_ok=True)
        arguments.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
