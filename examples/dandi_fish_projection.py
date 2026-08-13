"""Build a full, tiled temporal projection of a whole-brain fish movie.

The NumPy function in this example reduces 50 time frames for each of 29
z-planes. NeuroFlow handles bounded remote reads, task planning, tiled Zarr
storage, provenance, resume, and verification. Internet access is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import neuroflow
from examples.dandi_hdf5 import save_reference_png
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.exceptions import ProvenanceMismatchError
from neuroflow.selection import NWBQuery

DANDISET = "DANDI:000350@0.240822.1759"
DANDI_DOI = "https://doi.org/10.48324/dandi.000350/0.240822.1759"
ASSET_ID = "4f898ff7-6084-4e84-a449-f05811c1d951"
ASSET_PATH = "sub-20170113-4/sub-20170113-4_ses-20170113T171241_ophys.nwb"
OBJECT_NAME = "NeuronOnePhotonSeries"
MOVIE_SHAPE = (3065, 888, 2048, 29)  # axes: time, y, x, z
DEFAULT_OUTPUT = (
    Path(__file__).parent / "_output" / "fish-projection-t50-full-volume.zarr"
)
DEFAULT_PREVIEW = (
    Path(__file__).parent / "_output" / "fish-projection-t50-full-volume-z14.png"
)


@dataclass(frozen=True)
class FishProjectionConfig:
    frames: int = 50
    tile_y: int = 256
    tile_x: int = 256
    block_size: int = 262_144
    cache_size: int = 67_108_864
    output: Path = DEFAULT_OUTPUT
    preview: Path = DEFAULT_PREVIEW


def temporal_median(tile: np.ndarray) -> np.ndarray:
    """Reduce a bounded ``(time, y, x, z)`` tile to ``(y, x, z)``."""
    return np.asarray(np.median(tile, axis=0), dtype=np.float32)


def build_adapter(config: FishProjectionConfig) -> FunctionAdapter:
    """Declare the NumPy operation and its axis/storage contract."""
    return FunctionAdapter(
        function=temporal_median,
        input_kind="array",
        output=ArrayOutput(
            "float32",
            name="median_projection",
            reduced_axes=("time",),
            chunks=(config.tile_y, config.tile_x, 1),
        ),
        name="temporal-median-projection",
        version="1",
        splittable_axes=("z",),
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
        selected = source.select(NWBQuery(asset=ASSET_ID, name=OBJECT_NAME))
        bounded = selected.isel(time=slice(0, config.frames))
        movie = neuroflow.NeuroArray(source, bounded)
        try:
            projection_array = movie.median(
                "time",
                output=config.output,
                chunks=(config.tile_y, config.tile_x, 1),
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
        middle_z = projection[:, :, MOVIE_SHAPE[3] // 2].compute()
        save_reference_png(middle_z, config.preview)
        return {
            "source": DANDISET,
            "source_doi": DANDI_DOI,
            "asset": ASSET_PATH,
            "input_axes": bounded.metadata.axes,
            "input_shape": bounded.metadata.shape,
            "native_chunks": bounded.metadata.native_chunks,
            "transport": bounded.metadata.attributes["transport"],
            "task_count": result.plan.task_count,
            "native_chunks_touched": config.frames * MOVIE_SHAPE[3],
            "output_axes": result.plan.output_axes,
            "output_shape": projection.shape,
            "output_chunks": projection.chunksize,
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
    if args.block_size < 65_536:
        parser.error("--block-size must be at least 65536 bytes")
    if not 8 <= args.cache_size_mib <= 512:
        parser.error("--cache-size-mib must be between 8 and 512")
    return FishProjectionConfig(
        frames=args.frames,
        tile_y=args.tile_y,
        tile_x=args.tile_x,
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
