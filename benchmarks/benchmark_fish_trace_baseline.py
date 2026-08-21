"""Run the manual PyNWB + transport + Dask fish trace baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import remfile
import zarr
from pynwb import NWBHDF5IO

from benchmarks.baselines import direct_dask_mean_traces
from benchmarks.benchmark_projection import _tree_size
from examples.dandi_fish_projection import (
    ASSET_ID,
    ASSET_PATH,
    DANDISET,
    OBJECT_NAME,
)
from neuroflow.benchmarking import benchmark_record, write_benchmark_record
from neuroflow.provenance import capture_environment

DEFAULT_SOURCE_URL = f"https://api.dandiarchive.org/api/assets/{ASSET_ID}/download/"


class _ResponseCounter:
    def __init__(self) -> None:
        self.responses = 0
        self.bytes = 0

    def __call__(self, response: object, *args: object, **kwargs: object) -> None:
        del args, kwargs
        headers = getattr(response, "headers", {})
        try:
            size = int(headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            size = 0
        self.responses += 1
        self.bytes += max(0, size)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _code_lines(path: Path) -> int:
    return _text_code_lines(path.read_text())


def _text_code_lines(value: str) -> int:
    return sum(
        1
        for line in value.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _open_direct(
    source_url: str,
    *,
    backend: Literal["lindi", "remfile"],
    block_size: int,
    cache_size: int,
    counter: _ResponseCounter,
) -> tuple[Any, NWBHDF5IO]:
    if backend == "lindi":
        lindi = importlib.import_module("lindi")
        remote = lindi.LindiH5pyFile.from_hdf5_file(source_url)
        return remote, NWBHDF5IO(file=remote, mode="r", load_namespaces=True)
    remote = remfile.File(
        source_url,
        _min_chunk_size=block_size,
        _max_cache_size=cache_size,
    )
    session = getattr(remote, "session", None)
    hooks = getattr(session, "hooks", None)
    if isinstance(hooks, dict):
        hooks.setdefault("response", []).append(counter)
    h5_file = h5py.File(remote, mode="r")
    return remote, NWBHDF5IO(file=h5_file, mode="r", load_namespaces=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--reference-traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--backend", choices=("lindi", "remfile"), required=True)
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=16,
        help=(
            "frames per accumulation pass; for a like-for-like comparison set "
            "this to trace_plan.time_window from the NeuroFlow record so both "
            "tools traverse the movie in the same temporal windows"
        ),
    )
    # Defaults match NeuroFlow's transport defaults (1 MiB blocks, 64 MiB
    # cache) so transfer figures are comparable unless deliberately varied.
    parser.add_argument("--block-size", type=int, default=1_048_576)
    parser.add_argument("--cache-size-mib", type=int, default=64)
    parser.add_argument(
        "--classification", choices=("current", "publication"), default="current"
    )
    args = parser.parse_args()
    environment = capture_environment()
    git = cast(dict[str, object], environment["git"])
    if args.classification == "publication" and git.get("dirty") is not False:
        parser.error("publication classification requires a clean Git tree")
    if not args.labels.exists():
        parser.error("--labels must point to a completed Cellpose Zarr output")
    if args.output.exists():
        parser.error("--output already exists; use a fresh baseline path")
    if args.classification == "publication" and (
        args.reference_traces is None or not args.reference_traces.exists()
    ):
        parser.error("publication baseline requires --reference-traces")

    label_group = zarr.open_group(str(args.labels), mode="r")
    labels = np.asarray(label_group["labels"], dtype=np.uint64)
    labels_checksum = hashlib.sha256(labels.tobytes()).hexdigest()
    counter = _ResponseCounter()
    remote: Any | None = None
    io: NWBHDF5IO | None = None
    started = time.perf_counter()
    try:
        remote, io = _open_direct(
            args.source_url,
            backend=cast(Literal["lindi", "remfile"], args.backend),
            block_size=args.block_size,
            cache_size=args.cache_size_mib * 1024 * 1024,
            counter=counter,
        )
        nwbfile: Any = io.read()
        dataset: Any = nwbfile.acquisition[OBJECT_NAME].data
        traces, cell_ids, plan = direct_dask_mean_traces(
            dataset,
            labels,
            time_chunk=args.time_chunk,
        )
        source_shape = tuple(int(value) for value in dataset.shape)
        source_chunks = tuple(int(value) for value in dataset.chunks)
        source_dtype = np.dtype(dataset.dtype)
    finally:
        if io is not None:
            io.close()
            if args.backend == "lindi":
                setattr(io, "_HDF5IO__file", None)
        if remote is not None and getattr(remote, "_is_open", True):
            remote.close()

    output_group = zarr.open_group(str(args.output), mode="w")
    output_group.create_dataset(
        "traces",
        data=traces,
        chunks=(min(args.time_chunk, traces.shape[0]), min(1024, traces.shape[1])),
    )
    output_group.create_dataset("cell_ids", data=cell_ids)
    wall_time = time.perf_counter() - started
    trace_checksum = hashlib.sha256(traces.tobytes()).hexdigest()

    validation: dict[str, object]
    if args.reference_traces is None:
        validation = {
            "valid": None,
            "status": "no NeuroFlow reference supplied",
            "atol": 1e-5,
            "rtol": 1e-5,
        }
    else:
        reference_group = zarr.open_group(str(args.reference_traces), mode="r")
        reference = np.asarray(reference_group["traces"], dtype=np.float32)
        reference_ids = np.asarray(reference_group["cell_ids"], dtype=np.uint64)
        ids_equal = bool(np.array_equal(cell_ids, reference_ids))
        shape_equal = traces.shape == reference.shape
        difference = (
            np.abs(traces.astype(np.float64) - reference.astype(np.float64))
            if shape_equal
            else np.array([np.inf])
        )
        maximum_error = float(np.max(difference, initial=0.0))
        validation = {
            "valid": ids_equal
            and shape_equal
            and bool(np.allclose(traces, reference, atol=1e-5, rtol=1e-5)),
            "cell_ids_equal": ids_equal,
            "shape_agreement": shape_equal,
            "maximum_absolute_error": maximum_error,
            "atol": 1e-5,
            "rtol": 1e-5,
        }

    script = Path(__file__)
    record = benchmark_record(
        name=f"dandi-fish-manual-{args.backend}-dask-traces",
        classification=args.classification,
        backend=f"pynwb-{args.backend}-dask",
        source={
            "dataset_identifier": DANDISET.split("@", 1)[0],
            "dataset_version": DANDISET.split("@", 1)[1],
            "asset": ASSET_PATH,
            "asset_id": ASSET_ID,
            "path": f"/acquisition/{OBJECT_NAME}",
            "shape": list(source_shape),
            "dtype": str(source_dtype),
            "physical_chunk_shape": list(source_chunks),
            "total_logical_bytes": int(np.prod(source_shape)) * source_dtype.itemsize,
            "selected_bytes": int(np.prod(source_shape)) * source_dtype.itemsize,
        },
        execution={
            "partition_configuration": plan.to_dict(),
            "memory_budget": None,
            "task_count": plan.dask_compute_calls,
            "bytes_read": counter.bytes if args.backend == "remfile" else None,
            "peak_rss_bytes": _peak_rss_bytes(),
            "wall_time_seconds": wall_time,
            "cache_state": (
                f"manual remfile: block {args.block_size} B, cache "
                f"{args.cache_size_mib} MiB"
                if args.backend == "remfile"
                else "LINDI-managed; counters unavailable through this bridge"
            ),
            # A comparison is only as fair as its shared configuration; these
            # are the knobs that must match the NeuroFlow record for transfer
            # and wall-time figures to be comparable, recorded so a reader can
            # check parity instead of trusting it.
            "configuration_parity": {
                "time_chunk_frames": args.time_chunk,
                "block_size": (
                    args.block_size if args.backend == "remfile" else None
                ),
                "cache_size_bytes": (
                    args.cache_size_mib * 1024 * 1024
                    if args.backend == "remfile"
                    else None
                ),
                "note": (
                    "set --time-chunk to the NeuroFlow record's "
                    "trace_plan.time_window and keep block/cache at the shared "
                    "defaults; the baseline reads the identical retained masks "
                    "via --labels"
                ),
            },
            "network_context": "public DANDI HTTPS; runner location is external",
        },
        result={
            "checksum": trace_checksum,
            "numerical_validation": validation,
            "integrity_verified": False,
            "resume": {"supported": False, "exercised": False},
            "output_bytes": _tree_size(args.output),
        },
        notes=[
            "This baseline uses the exact persisted Cellpose masks from NeuroFlow.",
            "It manually implements source-chunk traversal and Dask compute calls.",
            "Its raw Zarr output has no manifests, resume, integrity, or provenance.",
            "LINDI is treated as a transport backend, not a competing workflow engine.",
        ],
    )
    record["workflow_features"] = {
        "manual_trace_function_code_lines": _text_code_lines(
            inspect.getsource(direct_dask_mean_traces)
        ),
        "complete_benchmark_runner_code_lines": _code_lines(script),
        "manual_chunk_configuration": True,
        "manual_scheduler_configuration": True,
        "persistence": True,
        "resume": False,
        "integrity_verification": False,
        "output_provenance": False,
        "label_checksum": labels_checksum,
        "http_responses": counter.responses if args.backend == "remfile" else None,
    }
    write_benchmark_record(args.record, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
