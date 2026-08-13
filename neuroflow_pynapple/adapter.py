"""Pynapple time-series adapter with deferred imports."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neuroflow.adapters.base import AdapterRequirements, LoadedPartition, TaskContext
from neuroflow.adapters.numpy import TableOutput
from neuroflow.exceptions import AdapterCompatibilityError
from neuroflow.execution.resources import ResourceSpec


@dataclass(frozen=True)
class PynappleAdapter:
    """Apply a user function to a bounded Pynapple time-series object."""

    function: Callable[..., object]
    columns: Sequence[str | int] | None = None
    parameters: Mapping[str, object] | None = None
    output: TableOutput = TableOutput("result")
    name: str = "pynapple-function"
    version: str = "1"
    deterministic: bool = True
    random_seed: int | None = None
    resources: ResourceSpec = ResourceSpec()

    external_packages: tuple[str, ...] = ("pynapple",)

    def requirements(self) -> AdapterRequirements:
        return AdapterRequirements(
            input_kinds=("time_series",),
            splittable_axes=("time",),
            requires_overlap={},
            output_kinds=("table",),
            resources=self.resources,
            deterministic=self.deterministic,
        )

    def identity_parameters(self) -> Mapping[str, object]:
        return {
            "columns": tuple(self.columns) if self.columns is not None else None,
            **dict(self.parameters or {}),
        }

    def prepare(self, partition: LoadedPartition, context: TaskContext) -> object:
        if partition.timestamps is None:
            raise AdapterCompatibilityError(
                "Pynapple requires NWB timestamps or a regular sampling rate"
            )
        data = np.asarray(partition.data)
        timestamps = np.asarray(partition.timestamps, dtype="float64")
        if data.shape[0] != len(timestamps):
            raise AdapterCompatibilityError(
                "timestamp count does not match the partition time dimension"
            )
        try:
            nap = importlib.import_module("pynapple")
        except ImportError as exc:
            raise AdapterCompatibilityError(
                "Pynapple is optional; install NeuroFlow with the 'pynapple' extra"
            ) from exc
        if data.ndim == 1:
            constructor = getattr(nap, "Tsd", None)
            kwargs: dict[str, Any] = {"t": timestamps, "d": data}
        elif data.ndim == 2:
            constructor = getattr(nap, "TsdFrame", None)
            kwargs = {"t": timestamps, "d": data}
            if self.columns is not None:
                if len(self.columns) != data.shape[1]:
                    raise AdapterCompatibilityError(
                        "Pynapple column count does not match the data width"
                    )
                kwargs["columns"] = list(self.columns)
        else:
            constructor = getattr(nap, "TsdTensor", None)
            kwargs = {"t": timestamps, "d": data}
        if constructor is None:
            raise AdapterCompatibilityError(
                "installed Pynapple lacks the required time-series constructor"
            )
        return constructor(**kwargs)

    def run(self, prepared: object, context: TaskContext) -> pd.DataFrame:
        result = self.function(prepared, **dict(self.parameters or {}))
        return _as_dataframe(result)

    def persist(self, output: object, writer: object, context: TaskContext) -> object:
        if not isinstance(output, pd.DataFrame):
            raise TypeError("PynappleAdapter must normalize output to a DataFrame")
        write = getattr(writer, "write_table", None)
        if write is None:
            raise TypeError("PynappleAdapter requires a table writer")
        return write(output)


def _as_dataframe(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return (
            value.rename_axis("time").reset_index()
            if not isinstance(value.index, pd.RangeIndex)
            else value
        )
    if isinstance(value, pd.Series):
        name = value.name if value.name is not None else "value"
        return value.rename(name).rename_axis("time").reset_index()
    as_dataframe = getattr(value, "as_dataframe", None)
    if callable(as_dataframe):
        return _as_dataframe(as_dataframe())
    as_series = getattr(value, "as_series", None)
    if callable(as_series):
        return _as_dataframe(as_series())
    if isinstance(value, Mapping):
        return pd.DataFrame([value])
    if np.isscalar(value):
        return pd.DataFrame({"value": [value]})
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Pynapple function output cannot be converted to a table"
        ) from exc
