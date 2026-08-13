"""Lightweight adapter for declared NumPy-like functions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neuroflow.adapters.base import (
    AdapterRequirements,
    LoadedPartition,
    TaskContext,
)
from neuroflow.execution.resources import ResourceSpec


@dataclass(frozen=True)
class ArrayOutput:
    dtype: str
    name: str = "result"


@dataclass(frozen=True)
class TableOutput:
    name: str = "result"


@dataclass(frozen=True)
class FunctionAdapter:
    function: Callable[..., object]
    input_kind: str
    output: ArrayOutput | TableOutput
    name: str = "function"
    version: str = "1"
    splittable_axes: tuple[str, ...] = ()
    requires_overlap: Mapping[str, int | str] | None = None
    parameters: Mapping[str, object] | None = None
    resources: ResourceSpec = ResourceSpec()
    deterministic: bool = True
    random_seed: int | None = None

    def requirements(self) -> AdapterRequirements:
        return AdapterRequirements(
            input_kinds=(self.input_kind,),
            splittable_axes=self.splittable_axes,
            requires_overlap=self.requires_overlap or {},
            output_kinds=(
                "array" if isinstance(self.output, ArrayOutput) else "table",
            ),
            resources=self.resources,
            deterministic=self.deterministic,
        )

    def prepare(self, partition: LoadedPartition, context: TaskContext) -> object:
        return partition

    def run(self, prepared: object, context: TaskContext) -> object:
        if not isinstance(prepared, LoadedPartition):
            raise TypeError("FunctionAdapter expects a LoadedPartition")
        kwargs: dict[str, Any] = dict(self.parameters or {})
        return self.function(np.asarray(prepared.data), **kwargs)

    def persist(self, output: object, writer: object, context: TaskContext) -> object:
        if isinstance(self.output, ArrayOutput):
            write_array = getattr(writer, "write_array", None)
            if write_array is None:
                raise TypeError("array adapter requires an array partition writer")
            return write_array(np.asarray(output, dtype=self.output.dtype))
        write_table = getattr(writer, "write_table", None)
        if write_table is None:
            raise TypeError("table adapter requires a table partition writer")
        if not isinstance(output, pd.DataFrame):
            output = pd.DataFrame(output)
        return write_table(output)
