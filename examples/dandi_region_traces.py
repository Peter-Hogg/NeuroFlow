"""Full-length fluorescence traces from one region, transferring ~13% of the movie.

The strongest case for bounded remote analysis is temporal, not spatial: a
trace needs *every* timepoint, so no preview or frame subset can produce it,
yet only the chunks that overlap the chosen region are ever transferred. On
DANDI:000223 the TwoPhotonSeries is stored as (1, 128, 128) chunks, so a
256x256 region touches four chunk columns across all 1,800 frames -- 236 MB
of logical data from a 3.6 GB movie. Measured with 32 KiB transport blocks,
the run transfers ~504 MB (13.4% of the movie; compressed chunks straddle
block boundaries) and the other ~87% is never downloaded.

Transport-block choice is the whole game on fine-chunked data, and the plan's
``estimated_transport_bytes_read`` predicts it: this same run measured
9,118 MB of transfer -- 2.4x the entire movie -- with the 1 MiB default
block, 924 MB with 64 KiB blocks, and 504 MB with 32 KiB blocks matching the
chunk size.

The workflow is the ordinary public API end to end: discover the series by
NWB type, slice the region by named axes, persist a bounded temporal mean of
the leading frames, segment that projection with Cellpose, then extract
verified, resumable traces for every detected cell across the entire
recording. This asset is dendritic-spine imaging, so cpsam finds few somata
(one, in the default region); the example's claim is the data-access ratio,
not segmentation richness. Internet access is required; Cellpose is an
optional extra.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

import neuroflow
from neuroflow.selection import NWBQuery
from neuroflow_cellpose.device import resolve_cellpose_device

DANDISET = "DANDI:000223@0.260528.0906"
ASSET_ID = "cc499fe1-fe23-42aa-8db0-0e689970fb89"
ASSET_PATH = "sub-3112/sub-3112_ecephys+ophys.nwb"
NEURODATA_TYPE = "TwoPhotonSeries"
MOVIE_SHAPE = (1800, 1024, 1024)  # axes: time, y, x
DEFAULT_OUTPUT_ROOT = Path(__file__).parent / "_output" / "region-traces"


@dataclass(frozen=True)
class RegionTracesConfig:
    # Chunk-aligned region where cpsam itself finds the most cells in the
    # 96-frame mean projection. This asset is dendritic-spine imaging, so
    # somata are sparse; the point of the example is the data-access ratio,
    # not segmentation richness.
    y_start: int = 512
    x_start: int = 384
    height: int = 256
    width: int = 256
    projection_frames: int = 96
    memory_limit: str = "2 GiB"
    backend: Literal["auto", "lindi", "remfile"] = "remfile"
    cellpose_device: str = "auto"
    # The transport block must match the fine 32 KiB chunks. Measured on this
    # exact run: 9,118 MB transferred with the 1 MiB library default (2.4x
    # the whole movie -- what the plan's estimated_transport_bytes_read
    # predicts), 924 MB with 64 KiB blocks, 504 MB with 32 KiB blocks.
    block_size: int = 32_768
    output_root: Path = DEFAULT_OUTPUT_ROOT


def run_example(config: RegionTracesConfig) -> dict[str, object]:
    device = resolve_cellpose_device(config.cellpose_device)
    root = config.output_root
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    source = neuroflow.open_dandi(
        DANDISET,
        backend=config.backend,
        storage_options=(
            None if config.backend == "lindi" else {"block_size": config.block_size}
        ),
    )
    try:
        selected = source.select(
            NWBQuery(asset=ASSET_ID, neurodata_type=NEURODATA_TYPE)
        )
        region = selected.isel(
            y=slice(config.y_start, config.y_start + config.height),
            x=slice(config.x_start, config.x_start + config.width),
        )
        movie = neuroflow.NeuroArray(source, region)

        # A bounded projection of the leading frames gives Cellpose an image;
        # the traces afterwards use every frame of the recording.
        projection = np.mean(
            movie[: config.projection_frames],
            axis="time",  # pyright: ignore[reportCallIssue, reportArgumentType]
        ).astype(np.float32)
        projection_path = root / "region-projection.zarr"
        persisted = projection.persist(
            projection_path, memory_limit=config.memory_limit
        )

        labels = persisted.cellpose(
            pretrained_model="cpsam",
            output=root / "region-cellpose.zarr",
            memory_limit=(
                config.memory_limit if device.gpu else "4 GiB"
            ),
            gpu=device.gpu,
            use_bfloat16=False,
            batch_size=1,
        )
        segmentation = labels.workflow
        assert segmentation is not None
        if not segmentation.verify().valid:
            raise RuntimeError("segmentation failed integrity verification")

        traces_path = root / "region-traces.zarr"
        traces = movie.extract_traces(
            labels, output=traces_path, memory_limit=config.memory_limit
        )
        verification = neuroflow.open_result(traces_path).verify()
        if not verification.valid:
            raise RuntimeError("traces failed integrity verification")

        stats = source.io_stats()
        trace_shape = tuple(int(value) for value in traces.shape)
        movie_dtype = np.dtype(selected.metadata.dtype)
        total_logical = int(np.prod(MOVIE_SHAPE)) * movie_dtype.itemsize
        region_logical = (
            MOVIE_SHAPE[0] * config.height * config.width * movie_dtype.itemsize
        )
        measured = stats.get("response_content_bytes")
        summary: dict[str, object] = {
            "dandiset": DANDISET,
            "asset": ASSET_PATH,
            "region": {
                "y": [config.y_start, config.y_start + config.height],
                "x": [config.x_start, config.x_start + config.width],
            },
            "movie_shape": list(MOVIE_SHAPE),
            "traces_shape": list(trace_shape),
            "cells": trace_shape[1] if len(trace_shape) == 2 else None,
            "timepoints": trace_shape[0] if trace_shape else None,
            "cellpose_device": device.to_dict(),
            "total_logical_bytes": total_logical,
            "region_logical_bytes": region_logical,
            "transport_block_size": (
                None if config.backend == "lindi" else config.block_size
            ),
            "measured_response_bytes": measured,
            "fraction_of_movie_transferred": (
                round(measured / total_logical, 4)
                if isinstance(measured, int)
                else None
            ),
            "integrity_verified": True,
            "wall_time_seconds": round(time.perf_counter() - started, 1),
        }
        traces.close()
        labels.close()
        persisted.close()
        return summary
    finally:
        source.close()


def parse_args(
    argv: list[str] | None = None,
) -> tuple[RegionTracesConfig, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y-start", type=int, default=512)
    parser.add_argument("--x-start", type=int, default=384)
    parser.add_argument("--block-size", type=int, default=32_768)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--projection-frames", type=int, default=96)
    parser.add_argument("--memory-limit", default="2 GiB")
    parser.add_argument(
        "--backend", choices=("auto", "lindi", "remfile"), default="remfile"
    )
    parser.add_argument(
        "--cellpose-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args(argv)
    for name in ("y_start", "x_start"):
        if getattr(args, name) % 128 != 0:
            parser.error(f"--{name.replace('_', '-')} should be chunk-aligned (128)")
    config = RegionTracesConfig(
        y_start=args.y_start,
        x_start=args.x_start,
        height=args.height,
        width=args.width,
        projection_frames=args.projection_frames,
        memory_limit=args.memory_limit,
        backend=cast(Any, args.backend),
        cellpose_device=args.cellpose_device,
        block_size=args.block_size,
        output_root=args.output_root,
    )
    return config, args.record


def main() -> None:
    config, record = parse_args()
    summary = run_example(config)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
