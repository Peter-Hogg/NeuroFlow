from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks.baselines import (
    direct_dask_zarr_projection,
    direct_pynwb_zarr_projection,
    lindi_hdf5_projection,
)


def test_direct_baselines_use_the_same_operation(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    source, values = nwb_zarr
    expected = np.median(values[:5], axis=0)

    pynwb_result = direct_pynwb_zarr_projection(
        source, object_name="movie", frames=5
    )
    dask_result = direct_dask_zarr_projection(
        source, object_name="movie", frames=5
    )

    np.testing.assert_allclose(pynwb_result, expected)
    np.testing.assert_allclose(dask_result, expected)


def test_lindi_baseline_explains_its_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def reject_lindi(name: str):  # type: ignore[no-untyped-def]
        if name == "lindi":
            raise ImportError(name)
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", reject_lindi)

    with pytest.raises(RuntimeError, match="uv sync --extra baselines"):
        lindi_hdf5_projection("unused.nwb", object_name="movie", frames=1)
