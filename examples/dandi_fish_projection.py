"""Median-project a centered crop from Misha Ahrens' whole-brain fish movie.

The default selects 50 time frames and the middle z-plane, touching 50 native
HDF5 chunks from a 150 GB remote NWB file. It never downloads the file and
writes one 128 x 128 PNG under examples/_output. Internet access is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import dask.array as da

import neuroflow
from examples.dandi_hdf5 import save_reference_png
from neuroflow.selection import NWBQuery

DANDISET = "DANDI:000350@0.240822.1759"
ASSET_ID = "4f898ff7-6084-4e84-a449-f05811c1d951"
ASSET_PATH = "sub-20170113-4/sub-20170113-4_ses-20170113T171241_ophys.nwb"
OBJECT_NAME = "NeuronOnePhotonSeries"
MOVIE_SHAPE = (3065, 888, 2048, 29)  # axes: time, y, x, z


@dataclass(frozen=True)
class FishProjectionConfig:
    frames: int = 50
    crop_size: int = 128
    crop_y: int = 380
    crop_x: int = 960
    z_plane: int = 14
    block_size: int = 262_144
    cache_size: int = 67_108_864
    output: Path = Path(__file__).parent / "_output" / "fish-median-projection.png"


def projection_slices(config: FishProjectionConfig) -> tuple[object, ...]:
    """Return the bounded time, y, x, z selection without reading data."""
    return (
        slice(0, config.frames),
        slice(config.crop_y, config.crop_y + config.crop_size),
        slice(config.crop_x, config.crop_x + config.crop_size),
        config.z_plane,
    )


def run_example(config: FishProjectionConfig) -> dict[str, object]:
    source = neuroflow.open_source(
        DANDISET,
        storage_options={
            "transport": "remfile",
            "block_size": config.block_size,
            "cache_size": config.cache_size,
        },
    )
    try:
        selection = source.select(NWBQuery(asset=ASSET_ID, name=OBJECT_NAME))
        movie = selection.as_dask_array(chunks="native")
        crop = movie[projection_slices(config)]
        projection = da.median(crop, axis=0)
        image = projection.compute(scheduler="threads")
        save_reference_png(image, config.output)
        return {
            "source": DANDISET,
            "asset": ASSET_PATH,
            "movie_shape": selection.metadata.shape,
            "native_chunks": selection.metadata.native_chunks,
            "transport": selection.metadata.attributes["transport"],
            "crop_shape": crop.shape,
            "native_chunks_touched": config.frames,
            "projection_shape": image.shape,
            "output_uri": str(config.output),
        }
    finally:
        source.close()


def parse_args(argv: list[str] | None = None) -> FishProjectionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=50, help="time-axis frames")
    parser.add_argument("--crop-size", type=int, default=128, help="square y/x size")
    parser.add_argument("--crop-y", type=int, default=380, help="y-axis crop start")
    parser.add_argument("--crop-x", type=int, default=960, help="x-axis crop start")
    parser.add_argument("--z-plane", type=int, default=14, help="z-axis plane index")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "_output" / "fish-median-projection.png",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 50:
        parser.error("--frames must be between 1 and 50")
    if not 16 <= args.crop_size <= 128:
        parser.error("--crop-size must be between 16 and 128")
    if not 0 <= args.crop_y <= MOVIE_SHAPE[1] - args.crop_size:
        parser.error("--crop-y and --crop-size must fit within y-axis size 888")
    if not 0 <= args.crop_x <= MOVIE_SHAPE[2] - args.crop_size:
        parser.error("--crop-x and --crop-size must fit within x-axis size 2048")
    if not 0 <= args.z_plane < MOVIE_SHAPE[3]:
        parser.error("--z-plane must be between 0 and 28")
    return FishProjectionConfig(
        frames=args.frames,
        crop_size=args.crop_size,
        crop_y=args.crop_y,
        crop_x=args.crop_x,
        z_plane=args.z_plane,
        output=args.output,
    )


def main() -> None:
    for key, value in run_example(parse_args()).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
