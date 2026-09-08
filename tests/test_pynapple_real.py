"""Opt-in equivalence test using the actual pynapple package.

The default suite validates the adapter against an in-process fake module;
this test proves the real pynapple constructors accept what the adapter
builds and hand the user function numerically identical data.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import neuroflow
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ParquetOutput
from neuroflow_pynapple import PynappleAdapter

pytestmark = pytest.mark.pynapple_real


@pytest.mark.skipif(
    os.environ.get("NEUROFLOW_RUN_REAL_PYNAPPLE") != "1",
    reason="set NEUROFLOW_RUN_REAL_PYNAPPLE=1 for real-pynapple validation",
)
def test_real_pynapple_receives_equivalent_time_series(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    nap = pytest.importorskip("pynapple")
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    expected = nwb_zarr[1]  # (10, 3, 4) float32 at 2 Hz

    received: list[tuple[np.ndarray, np.ndarray]] = []

    def summarize(series: object) -> pd.DataFrame:
        assert isinstance(series, nap.TsdTensor), type(series).__name__
        timestamps = np.asarray(series.t)  # type: ignore[attr-defined]
        data = np.asarray(series.d)  # type: ignore[attr-defined]
        received.append((timestamps.copy(), data.copy()))
        return pd.DataFrame(
            {
                "start": [float(timestamps[0])],
                "samples": [len(timestamps)],
                "mean": [float(data.mean())],
            }
        )

    adapter = PynappleAdapter(function=summarize)
    result = neuroflow.run(
        source=source,
        selection=movie,
        adapter=adapter,
        partition=TimeWindowPlan(size=5, overlap=0),
        output=ParquetOutput(str(tmp_path / "pynapple-real")),
        max_workers=1,
        execute=True,
    )

    assert result.verify().valid
    received.sort(key=lambda item: float(item[0][0]))
    stitched = np.concatenate([data for _, data in received], axis=0)
    np.testing.assert_array_equal(stitched, expected)

    # The 2 Hz sampling rate must survive the trip through real pynapple.
    all_timestamps = np.concatenate([timestamps for timestamps, _ in received])
    np.testing.assert_allclose(all_timestamps, np.arange(10) / 2.0)

    # Direct construction with the adapter's kwargs is accepted and identical.
    first_timestamps, first_data = received[0]
    direct = nap.TsdTensor(t=first_timestamps, d=expected[: len(first_timestamps)])
    np.testing.assert_array_equal(
        np.asarray(direct.d),  # type: ignore[attr-defined]
        first_data,
    )

    table = result.tables["result"].as_dask_dataframe().compute()
    assert sorted(table["start"].tolist()) == [0.0, 2.5]
    assert result.provenance is not None
    assert result.provenance["external_libraries"]["pynapple"]  # type: ignore[index]
    source.close()
