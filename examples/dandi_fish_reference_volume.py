"""Create a bounded 29-plane reference volume from DANDI:000350.

The remote movie has axes ``(time, y, x, z)`` and physical HDF5 chunks
``(1, 888, 2048, 1)``.  This example keeps those source chunks intact, computes
one z-plane at a time, and writes a NumPy volume before optionally opening it
in napari.  A y/x crop limits result and working memory, but does *not* reduce
the full image-plane chunks transferred and decompressed by HDF5.
"""

from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path

import dask.array as da
import numpy as np
from numpy.lib.format import open_memmap

import neuroflow
from neuroflow.selection import NWBQuery

DANDISET = "DANDI:000350@0.240822.1759"
ASSET_ID = "4f898ff7-6084-4e84-a449-f05811c1d951"
ASSET_PATH = "sub-20170113-4/sub-20170113-4_ses-20170113T171241_ophys.nwb"
OBJECT_NAME = "NeuronOnePhotonSeries"
MOVIE_SHAPE = (3065, 888, 2048, 29)  # time, y, x, z
NATIVE_CHUNKS = (1, 888, 2048, 1)
MAX_FRAMES = 50
MAX_CROP_PIXELS = 888 * 2048


@dataclass(frozen=True)
class ReferenceVolumeConfig:
    frames: int = 9
    start_frame: int = 0
    y_start: int = 188
    y_size: int = 512
    x_start: int = 768
    x_size: int = 512
    block_size: int = 262_144
    cache_size: int = 67_108_864
    output: Path = Path(__file__).parent / "_output" / "fish-reference.npy"
    tiff: Path | None = None
    view: bool = True


def temporal_median_plane(plane: da.Array) -> da.Array:
    """Return a lazy float32 temporal median for one ``(t, y, x, 1)`` plane."""
    if plane.ndim != 4 or plane.shape[-1] != 1:
        raise ValueError("plane must have shape (time, y, x, 1)")
    # Dask needs the reduced axis in one logical chunk. The input chunks still
    # correspond one-for-one to the native (time, z) HDF5 image-plane chunks.
    rechunked = plane.rechunk((int(plane.shape[0]), -1, -1, 1))  # pyright: ignore[reportArgumentType]
    return da.median(rechunked[..., 0], axis=0).astype(np.float32)


def compute_reference_volume(
    movie: da.Array, output: Path, *, scheduler: str = "synchronous"
) -> np.memmap:
    """Compute z planes serially into an on-disk ``(z, y, x)`` NumPy array."""
    if movie.ndim != 4:
        raise ValueError("movie must have axes (time, y, x, z)")
    output.parent.mkdir(parents=True, exist_ok=True)
    volume = open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=tuple(int(size) for size in (movie.shape[3], *movie.shape[1:3])),
    )
    for z_index in range(int(movie.shape[3])):
        # Each graph contains exactly this z-plane, so no native source chunk is
        # requested by two plane computations. Serial execution bounds memory.
        volume[z_index] = temporal_median_plane(
            movie[:, :, :, z_index : z_index + 1]
        ).compute(scheduler=scheduler)
        volume.flush()
    return volume


def save_tiff(volume: np.ndarray, path: Path) -> None:
    """Save a z/y/x TIFF stack when the optional tifffile package is present."""
    try:
        tifffile = importlib.import_module("tifffile")
    except ImportError as exc:
        raise RuntimeError(
            "TIFF output requires the optional 'tifffile' package; "
            "install it or omit --tiff"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, volume, photometric="minisblack", metadata={"axes": "ZYX"})


def show_in_napari(volume: np.ndarray) -> bool:
    """Open the saved volume in napari, returning false when GUI use is unavailable."""
    if os.name != "nt" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        print("Napari not opened: no display detected. Use --no-view on servers.")
        return False
    try:
        napari = importlib.import_module("napari")
    except ImportError:
        print("Napari not opened: install 'napari', or use --no-view.")
        return False
    viewer = napari.Viewer()
    viewer.add_image(volume, name="DANDI:000350 temporal median", colormap="gray")
    napari.run()
    return True


def run_example(config: ReferenceVolumeConfig) -> dict[str, object]:
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
        bounded = selected.isel(
            time=slice(config.start_frame, config.start_frame + config.frames),
            y=slice(config.y_start, config.y_start + config.y_size),
            x=slice(config.x_start, config.x_start + config.x_size),
        )
        # Keep one source chunk per time point and z-plane. Although y/x are
        # logically cropped, each read touches its complete native HDF5 chunk.
        movie = bounded.as_dask_array(chunks=(1, config.y_size, config.x_size, 1))
        volume = compute_reference_volume(movie, config.output)
        if config.tiff is not None:
            save_tiff(volume, config.tiff)
        viewed = show_in_napari(volume) if config.view else False
        return {
            "source": DANDISET,
            "asset": ASSET_PATH,
            "input_shape": bounded.metadata.shape,
            "native_chunks": selected.metadata.native_chunks,
            "native_chunks_touched": config.frames * MOVIE_SHAPE[3],
            "output_shape": volume.shape,
            "output_npy": str(config.output),
            "output_tiff": str(config.tiff) if config.tiff else None,
            "napari_opened": viewed,
        }
    finally:
        source.close()


def parse_args(argv: list[str] | None = None) -> ReferenceVolumeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--y-start", type=int, default=188)
    parser.add_argument("--y-size", type=int, default=512)
    parser.add_argument("--x-start", type=int, default=768)
    parser.add_argument("--x-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=262_144)
    parser.add_argument("--cache-size-mib", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ReferenceVolumeConfig.output)
    parser.add_argument("--tiff", type=Path, help="optional TIFF stack output")
    parser.add_argument(
        "--no-view", action="store_true", help="save without opening napari"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.frames <= MAX_FRAMES:
        parser.error(f"--frames must be between 1 and {MAX_FRAMES}")
    if not 0 <= args.start_frame < MOVIE_SHAPE[0]:
        parser.error("--start-frame is outside the movie")
    if args.start_frame + args.frames > MOVIE_SHAPE[0]:
        parser.error("the requested time window extends past the movie")
    for axis, start, size, limit in (
        ("y", args.y_start, args.y_size, MOVIE_SHAPE[1]),
        ("x", args.x_start, args.x_size, MOVIE_SHAPE[2]),
    ):
        if start < 0 or size < 1 or start + size > limit:
            parser.error(f"--{axis}-start/--{axis}-size must select within 0:{limit}")
    if args.y_size * args.x_size > MAX_CROP_PIXELS:
        parser.error("the requested spatial crop exceeds one full image plane")
    if args.block_size < 65_536:
        parser.error("--block-size must be at least 65536 bytes")
    if not 8 <= args.cache_size_mib <= 512:
        parser.error("--cache-size-mib must be between 8 and 512")
    return ReferenceVolumeConfig(
        frames=args.frames,
        start_frame=args.start_frame,
        y_start=args.y_start,
        y_size=args.y_size,
        x_start=args.x_start,
        x_size=args.x_size,
        block_size=args.block_size,
        cache_size=args.cache_size_mib * 1024 * 1024,
        output=args.output,
        tiff=args.tiff,
        view=not args.no_view,
    )


def main() -> None:
    for key, value in run_example(parse_args()).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
