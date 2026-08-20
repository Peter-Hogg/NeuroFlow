"""Build a full, tiled temporal projection of a whole-brain fish movie.

np.median reduces 50 time frames for every x/y pixel in each of 29 z-planes.
The explicit float32 cast keeps this visualization-sized result compact. NeuroFlow
intercepts that ordinary NumPy expression and handles bounded remote reads, task
planning, tiled Zarr storage, provenance, resume, and verification. Internet
access is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

import neuroflow
from examples.dandi_hdf5 import save_reference_png
from neuroflow.exceptions import ProvenanceMismatchError
from neuroflow.selection import NWBQuery

DANDISET = "DANDI:000350@0.240822.1759"
DANDI_DOI = "https://doi.org/10.48324/dandi.000350/0.240822.1759"
ASSET_ID = "4f898ff7-6084-4e84-a449-f05811c1d951"
ASSET_PATH = "sub-20170113-4/sub-20170113-4_ses-20170113T171241_ophys.nwb"
OBJECT_NAME = "NeuronOnePhotonSeries"
MOVIE_SHAPE = (3065, 888, 2048, 29)  # axes: time, y, x, z
DEFAULT_OUTPUT = (
    Path(__file__).parent / "_output" / "fish-projection-numpy-t50-full-volume.zarr"
)
DEFAULT_PREVIEW = (
    Path(__file__).parent / "_output" / "fish-projection-numpy-t50-full-volume-z14.png"
)


@dataclass(frozen=True)
class FishProjectionConfig:
    frames: int = 50
    tile_y: int = 256
    tile_x: int = 256
    max_workers: int = 1
    backend: Literal["auto", "lindi", "remfile"] = "auto"
    block_size: int = 262_144
    cache_size: int = 67_108_864
    output: Path = DEFAULT_OUTPUT
    preview: Path = DEFAULT_PREVIEW


def build_projection(movie: neuroflow.NeuroArray) -> neuroflow.NeuroArray:
    """Describe a temporal projection with normal NumPy syntax; no reads occur."""
    projection = np.median(movie, axis="time")  # type: ignore[call-overload]
    return projection.astype(np.float32)


def run_example(config: FishProjectionConfig) -> dict[str, object]:
    storage_options: dict[str, object] | None = (
        {
            "block_size": config.block_size,
            "cache_size": config.cache_size,
        }
        if config.backend != "lindi"
        else None
    )
    source = neuroflow.open_dandi(
        DANDISET,
        backend=config.backend,
        storage_options=storage_options,
    )
    try:
        selected = source.select(NWBQuery(asset=ASSET_ID, name=OBJECT_NAME))
        bounded = selected.isel(time=slice(0, config.frames))
        movie = neuroflow.NeuroArray(source, bounded)
        try:
            projection_array = build_projection(movie).persist(
                config.output,
                chunks=(config.tile_y, config.tile_x, 1),
                max_workers=config.max_workers,
                memory_limit="2 GiB",
            )
        except ProvenanceMismatchError as exc:
            raise RuntimeError(
                f"{config.output} contains a different NeuroFlow workflow. "
                "Keep it for reproducibility and rerun with a fresh path, for "
                "example: --output examples/_output/fish-projection-new.zarr "
                "--preview examples/_output/fish-projection-new-z14.png"
            ) from exc
        result = projection_array.workflow
        assert result is not None
        verification = result.verify()
        projection = projection_array.selection.as_dask_array()
        attributes = bounded.metadata.attributes or {}
        middle_z = projection[:, :, MOVIE_SHAPE[3] // 2].compute()
        save_reference_png(middle_z, config.preview)
        return {
            "source": DANDISET,
            "source_doi": DANDI_DOI,
            "asset": ASSET_PATH,
            "input_axes": bounded.metadata.axes,
            "input_shape": bounded.metadata.shape,
            "input_dtype": bounded.metadata.dtype,
            "native_chunks": bounded.metadata.native_chunks,
            "transport": attributes.get("transport"),
            "task_count": result.plan.task_count,
            "native_chunks_touched": config.frames * MOVIE_SHAPE[3],
            "output_axes": result.plan.output_axes,
            "output_shape": projection.shape,
            "output_dtype": str(projection.dtype),
            "output_chunks": tuple(int(axis[0]) for axis in projection.chunks),
            "verified": verification.valid,
            "remote_io": source.io_stats(),
            "output_uri": str(config.output),
            "preview_uri": str(config.preview),
        }
    finally:
        source.close()


def parse_args(argv: list[str] | None = None) -> FishProjectionConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--frames", type=int, default=50, help="leading time frames")
    parser.add_argument("--tile-y", type=int, default=256, help="output y chunk")
    parser.add_argument("--tile-x", type=int, default=256, help="output x chunk")
    parser.add_argument(
        "--backend",
        choices=("auto", "lindi", "remfile"),
        default="auto",
        help="remote HDF5 backend (auto currently chooses remfile)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="concurrent projection tasks (capped for archive-friendly access)",
    )
    parser.add_argument(
        "--block-size", type=int, default=262_144, help="remote read block bytes"
    )
    parser.add_argument(
        "--cache-size-mib", type=int, default=64, help="bounded remote cache"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="durable projection Zarr path",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=DEFAULT_PREVIEW,
        help="middle-plane PNG path",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 50:
        parser.error("--frames must be between 1 and 50")
    if not 64 <= args.tile_y <= MOVIE_SHAPE[1]:
        parser.error("--tile-y must be between 64 and 888")
    if not 64 <= args.tile_x <= MOVIE_SHAPE[2]:
        parser.error("--tile-x must be between 64 and 2048")
    if not 1 <= args.max_workers <= 4:
        parser.error("--max-workers must be between 1 and 4")
    if args.block_size < 65_536:
        parser.error("--block-size must be at least 65536 bytes")
    if not 8 <= args.cache_size_mib <= 512:
        parser.error("--cache-size-mib must be between 8 and 512")
    return FishProjectionConfig(
        frames=args.frames,
        tile_y=args.tile_y,
        tile_x=args.tile_x,
        max_workers=args.max_workers,
        backend=args.backend,
        block_size=args.block_size,
        cache_size=args.cache_size_mib * 1024 * 1024,
        output=args.output,
        preview=args.preview,
    )


def main() -> None:
    for key, value in run_example(parse_args()).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
