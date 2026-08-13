"""Bounded remote NWB-HDF5 example using DANDI:000049.

Requires internet access. The default reads metadata plus one 1 x 512 x 512
float32 dataset (~1 MiB uncompressed) through HTTP byte-range requests. It does
not download the 27.8 MB NWB file. Generated results go under examples/_output.
"""

from __future__ import annotations

import argparse
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
    output: Path = Path(__file__).parent / "_output" / "dandi-hdf5-scaled.zarr"


def scale_block(block: np.ndarray, factor: float) -> np.ndarray:
    """A user-supplied function receiving only the bounded selected block."""
    return np.asarray(block, dtype=np.float32) * np.float32(factor)


def build_query(config: ExampleConfig) -> NWBQuery:
    """Build the pinned semantic query without touching the network."""
    return NWBQuery(asset=config.asset, name=config.object_name)


def run_example(config: ExampleConfig) -> dict[str, object]:
    config.output.parent.mkdir(parents=True, exist_ok=True)
    source = neuroflow.open_source(
        config.source,
        storage_options={"block_size": config.block_size, "cache_type": "readahead"},
    )
    try:
        selected = source.select(build_query(config))
        lazy = selected.as_dask_array()
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
        "task_count": result.plan.task_count,
        "verified": verification.valid,
        "output_shape": output.shape,
        "output_uri": str(config.output),
    }


def parse_args(argv: list[str] | None = None) -> ExampleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=262_144)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "_output" / "dandi-hdf5-scaled.zarr",
    )
    args = parser.parse_args(argv)
    if args.block_size < 65_536:
        parser.error("--block-size must be at least 65536 bytes")
    return ExampleConfig(
        factor=args.factor, block_size=args.block_size, output=args.output
    )


def main() -> None:
    for key, value in run_example(parse_args()).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
