"""Bounded remote NWB-HDF5 example using DANDI:000049.

Requires internet access. The default reads metadata plus one 1 x 512 x 512
float32 dataset (~1 MiB uncompressed) through HTTP byte-range requests. It does
not download the 27.8 MB NWB file. Generated results go under examples/_output.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ZarrOutput

DANDISET = "DANDI:000049@0.230223.1424"
ASSET_ID = "82fd3c31-37b7-4261-a6ab-0979bc78877c"
ASSET_PATH = "sub-760940732/sub-760940732_ses-798500537_behavior+ophys.nwb"
OBJECT_NAME = "max_project"


@dataclass(frozen=True)
class ExampleConfig:
    source: str = DANDISET
    asset: str = ASSET_ID
    object_name: str = OBJECT_NAME
    factor: float = 0.5
    block_size: int = 262_144
    preview_size: int = 128
    output: Path = Path(__file__).parent / "_output" / "dandi-hdf5-scaled.zarr"
    preview: Path = Path(__file__).parent / "_output" / "dandi-hdf5-preview.png"


def scale_block(block: np.ndarray, factor: float) -> np.ndarray:
    """A user-supplied function receiving only the bounded selected block."""
    return np.asarray(block, dtype=np.float32) * np.float32(factor)


def build_query(config: ExampleConfig) -> NWBQuery:
    """Build the pinned semantic query without touching the network."""
    return NWBQuery(asset=config.asset, name=config.object_name)


def save_reference_png(image: np.ndarray, path: Path) -> None:
    """Save a 2-D numerical block as a dependency-free grayscale PNG."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("reference image must be two-dimensional")
    finite = np.isfinite(values)
    if finite.any():
        low, high = np.percentile(values[finite], (1.0, 99.0))
        scaled = np.clip((values - low) / max(float(high - low), 1e-12), 0, 1)
        pixels = np.where(finite, scaled * 255, 0).astype(np.uint8)
    else:
        pixels = np.zeros(values.shape, dtype=np.uint8)

    height, width = pixels.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body))
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def run_example(config: ExampleConfig) -> dict[str, object]:
    config.output.parent.mkdir(parents=True, exist_ok=True)
    source = neuroflow.open_source(
        config.source,
        storage_options={"block_size": config.block_size, "cache_type": "readahead"},
    )
    try:
        selected = source.select(build_query(config))
        lazy = selected.as_dask_array(
            chunks=(1, config.preview_size, config.preview_size)
        )
        # Only this single logical Dask block is computed for the preview.
        preview_block = lazy.blocks[0, 0, 0].compute()[0]
        save_reference_png(preview_block, config.preview)
        adapter = FunctionAdapter(
            function=scale_block,
            input_kind="array",
            output=ArrayOutput("float32", name="scaled_max_projection"),
            name="scale-max-projection",
            version="1",
            splittable_axes=("time",),
            parameters={"factor": config.factor},
        )
        result = neuroflow.run(
            source=source,
            selection=selected,
            adapter=adapter,
            partition=TimeWindowPlan(size=1),
            output=ZarrOutput(str(config.output)),
        )
        result.execute()
        verification = result.verify()
        shape = selected.metadata.shape
    finally:
        source.close()

    reopened = neuroflow.open_result(config.output)
    output = reopened.arrays["scaled_max_projection"].as_dask_array()
    return {
        "source": config.source,
        "asset": ASSET_PATH,
        "selection_shape": shape,
        "dask_chunks": lazy.chunks,
        "preview_chunk_shape": preview_block.shape,
        "preview_uri": str(config.preview),
        "task_count": result.plan.task_count,
        "verified": verification.valid,
        "output_shape": output.shape,
        "output_uri": str(config.output),
    }


def parse_args(argv: list[str] | None = None) -> ExampleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=262_144)
    parser.add_argument("--preview-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "_output" / "dandi-hdf5-scaled.zarr",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path(__file__).parent / "_output" / "dandi-hdf5-preview.png",
    )
    args = parser.parse_args(argv)
    if args.block_size < 65_536:
        parser.error("--block-size must be at least 65536 bytes")
    if not 16 <= args.preview_size <= 512:
        parser.error("--preview-size must be between 16 and 512 pixels")
    return ExampleConfig(
        factor=args.factor,
        block_size=args.block_size,
        preview_size=args.preview_size,
        output=args.output,
        preview=args.preview,
    )


def main() -> None:
    for key, value in run_example(parse_args()).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
