"""Median-project a tiny crop from Misha Ahrens' whole-brain fish movie.

The default touches three native HDF5 chunks from a 150 GB remote NWB file,
never downloads the file, and writes one 64 x 64 PNG under examples/_output.
Internet access is required.
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


@dataclass(frozen=True)
class FishProjectionConfig:
    frames: int = 3
    crop_size: int = 64
    z_plane: int = 0
    block_size: int = 262_144
    cache_size: int = 67_108_864
    output: Path = Path(__file__).parent / "_output" / "fish-median-projection.png"


def projection_slices(config: FishProjectionConfig) -> tuple[object, ...]:
    """Return the bounded time, y, x, z selection without reading data."""
    return (
        slice(0, config.frames),
        slice(0, config.crop_size),
        slice(0, config.crop_size),
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
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--z-plane", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "_output" / "fish-median-projection.png",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= 9:
        parser.error("--frames must be between 1 and 9")
    if not 16 <= args.crop_size <= 128:
        parser.error("--crop-size must be between 16 and 128")
    if not 0 <= args.z_plane < 29:
        parser.error("--z-plane must be between 0 and 28")
    return FishProjectionConfig(
        frames=args.frames,
        crop_size=args.crop_size,
        z_plane=args.z_plane,
        output=args.output,
    )


def main() -> None:
    for key, value in run_example(parse_args()).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
