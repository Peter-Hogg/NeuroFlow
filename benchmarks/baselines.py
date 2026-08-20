"""Fair baseline implementations over the same NWB array and selection."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import dask.array as da
import numpy as np
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO


def direct_pynwb_zarr_projection(
    source: Path, *, object_name: str, frames: int
) -> np.ndarray:
    """Materialize the selected local array through PyNWB/HDMF-Zarr."""
    io = cast(Any, NWBZarrIO)(source, mode="r", load_namespaces=True)
    with io:
        nwbfile: Any = io.read()
        dataset = nwbfile.acquisition[object_name].data
        return np.median(np.asarray(dataset[:frames]), axis=0)


def direct_dask_zarr_projection(
    source: Path, *, object_name: str, frames: int
) -> np.ndarray:
    """Run the same median through direct Dask over the HDMF-Zarr array."""
    io = cast(Any, NWBZarrIO)(source, mode="r", load_namespaces=True)
    with io:
        nwbfile: Any = io.read()
        dataset = nwbfile.acquisition[object_name].data
        chunks = getattr(dataset, "chunks", None) or "auto"
        lazy = da.from_array(dataset, chunks=chunks, asarray=False, fancy=False)
        return np.asarray(da.median(lazy[:frames], axis=0).compute())


def lindi_hdf5_projection(
    source: str | Path,
    *,
    object_name: str,
    frames: int,
    z_axis: int | None = None,
) -> np.ndarray:
    """Read a local/remote HDF5 NWB file via LINDI's documented PyNWB bridge.

    When ``z_axis`` is supplied, planes are reduced independently so the
    baseline remains bounded rather than materializing a complete 4-D movie.
    """
    try:
        lindi = importlib.import_module("lindi")
    except ImportError as exc:
        raise RuntimeError(
            "the LINDI baseline requires the 'baselines' extra: "
            "uv sync --extra baselines"
        ) from exc
    lindi_file = lindi.LindiH5pyFile.from_hdf5_file(str(source))
    try:
        with NWBHDF5IO(file=lindi_file, mode="r", load_namespaces=True) as io:
            nwbfile: Any = io.read()
            dataset: Any = nwbfile.acquisition[object_name].data
            if z_axis is None:
                return np.median(np.asarray(dataset[:frames]), axis=0)
            if z_axis != dataset.ndim - 1:
                raise ValueError("the current bounded LINDI baseline expects z last")
            planes = [
                np.median(np.asarray(dataset[:frames, ..., index]), axis=0)
                for index in range(int(dataset.shape[z_axis]))
            ]
            return np.stack(planes, axis=-1)
    finally:
        close = getattr(lindi_file, "close", None)
        if callable(close):
            close()
