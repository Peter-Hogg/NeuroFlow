import importlib
import importlib.metadata
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from distributed import Client, LocalCluster

import neuroflow
from neuroflow.adapters import LoadedPartition, TaskContext
from neuroflow.exceptions import AdapterCompatibilityError
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ParquetOutput
from neuroflow_pynapple import PynappleAdapter
from neuroflow_pynapple import adapter as pynapple_adapter_module


def _install_fake_pynapple(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("pynapple")

    class TimeSeries:
        def __init__(
            self,
            *,
            t: np.ndarray,
            d: np.ndarray,
            columns: list[str | int] | None = None,
        ) -> None:
            self.t = np.asarray(t)
            self.d = np.asarray(d)
            self.columns = columns

    module.Tsd = TimeSeries  # type: ignore[attr-defined]
    module.TsdFrame = TimeSeries  # type: ignore[attr-defined]
    module.TsdTensor = TimeSeries  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynapple", module)
    return module


def test_pynapple_dependency_is_deferred(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def missing(name: str, package: str | None = None) -> ModuleType:
        if name == "pynapple":
            raise ImportError("not installed")
        return original(name, package)

    monkeypatch.setattr(pynapple_adapter_module.importlib, "import_module", missing)
    adapter = PynappleAdapter(function=lambda value: value)
    loaded = LoadedPartition(
        np.zeros((4,), dtype="float32"),
        (slice(0, 4),),
        (slice(0, 4),),
        (slice(0, 4),),
        timestamps=np.arange(4, dtype="float64"),
    )
    with pytest.raises(AdapterCompatibilityError, match="optional"):
        adapter.prepare(loaded, TaskContext("partition"))


def test_regular_nwb_series_runs_as_partitioned_pynapple_workflow(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pynapple(monkeypatch)
    original_version = importlib.metadata.version

    def package_version(name: str) -> str:
        return "0.11.3" if name == "pynapple" else original_version(name)

    monkeypatch.setattr(importlib.metadata, "version", package_version)
    source = neuroflow.open_source(nwb_zarr[0])
    movie = source.select(NWBQuery(name="movie"))
    calls: list[float] = []

    def summarize(series: object) -> pd.DataFrame:
        timestamps = np.asarray(series.t)  # type: ignore[attr-defined]
        calls.append(float(timestamps[0]))
        return pd.DataFrame(
            {
                "start": [float(timestamps[0])],
                "end": [float(timestamps[-1])],
                "samples": [len(timestamps)],
            }
        )

    adapter = PynappleAdapter(function=summarize)
    output = ParquetOutput(str(tmp_path / "pynapple-regular"))
    with (
        LocalCluster(n_workers=2, threads_per_worker=1, processes=False) as cluster,
        Client(cluster),
    ):
        result = neuroflow.run(
            source=source,
            selection=movie,
            adapter=adapter,
            partition=TimeWindowPlan(size=4, overlap=1),
            output=output,
            scheduler="distributed",
            execute=True,
        )
        calls.clear()
        result.resume()
        assert calls == []
    table = result.tables["result"].as_dask_dataframe().compute()
    assert sorted(table["start"].tolist()) == [0.0, 1.5, 3.5]
    assert result.provenance is not None
    assert result.provenance["external_libraries"]["pynapple"] == "0.11.3"  # type: ignore[index]
    assert result.verify().valid
    source.close()


def test_irregular_nwb_timestamps_are_bounded_per_pynapple_partition(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pynapple(monkeypatch)
    source = neuroflow.open_source(nwb_zarr[0])
    irregular = source.select(NWBQuery(name="irregular"))

    def summarize(series: object) -> pd.DataFrame:
        timestamps = np.asarray(series.t)  # type: ignore[attr-defined]
        return pd.DataFrame({"start": [timestamps[0]], "end": [timestamps[-1]]})

    result = neuroflow.run(
        source=source,
        selection=irregular,
        adapter=PynappleAdapter(function=summarize),
        partition=TimeWindowPlan(size=2),
        output=ParquetOutput(str(tmp_path / "pynapple-irregular")),
        execute=True,
    )
    table = result.tables["result"].as_dask_dataframe().compute()
    assert sorted(table["start"].tolist()) == [0.0, 1.1]
    assert sorted(table["end"].tolist()) == [0.4, 2.0]
    source.close()
