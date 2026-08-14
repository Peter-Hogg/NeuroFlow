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
from neuroflow.expression import (
    EXPRESSION_SCHEMA_VERSION,
    Expression,
    estimate_working_memory,
    evaluate_numpy,
    expression_to_dict,
    output_shape_for_input,
)
from neuroflow.storage.base import validate_component_name


@dataclass(frozen=True)
class ArrayOutput:
    dtype: str
    name: str = "result"
    reduced_axes: tuple[str, ...] = ()
    chunks: tuple[int, ...] | None = None
    kept_reduced_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_component_name(self.name)
        if set(self.reduced_axes) & set(self.kept_reduced_axes):
            raise ValueError("dropped and kept reduced axes cannot overlap")


@dataclass(frozen=True)
class TableOutput:
    name: str = "result"

    def __post_init__(self) -> None:
        validate_component_name(self.name)


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


@dataclass(frozen=True)
class ExpressionAdapter:
    """Evaluate one canonical lazy expression on bounded source partitions."""

    expression: Expression
    output: ArrayOutput
    splittable_axes: tuple[str, ...]
    name: str = "numpy-expression"
    version: str = "1"
    deterministic: bool = True
    random_seed: int | None = None
    external_packages: tuple[str, ...] = ("numpy",)

    @property
    def output_axes(self) -> tuple[str, ...]:
        return self.expression.axes

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.expression.shape

    @property
    def parameters(self) -> Mapping[str, object]:
        return self.identity_parameters()

    def identity_parameters(self) -> Mapping[str, object]:
        return {
            "expression_schema_version": EXPRESSION_SCHEMA_VERSION,
            "numpy_version": np.__version__,
            "expression": expression_to_dict(self.expression),
        }

    def requirements(self) -> AdapterRequirements:
        return AdapterRequirements(
            input_kinds=("array",),
            splittable_axes=self.splittable_axes,
            requires_overlap={},
            output_kinds=("array",),
            deterministic=True,
        )

    def prepare(self, partition: LoadedPartition, context: TaskContext) -> object:
        return partition

    def run(self, prepared: object, context: TaskContext) -> object:
        if not isinstance(prepared, LoadedPartition):
            raise TypeError("ExpressionAdapter expects a LoadedPartition")
        return evaluate_numpy(self.expression, np.asarray(prepared.data))

    def persist(self, output: object, writer: object, context: TaskContext) -> object:
        write_array = getattr(writer, "write_array", None)
        if write_array is None:
            raise TypeError("expression adapter requires an array partition writer")
        return write_array(np.asarray(output, dtype=np.dtype(self.expression.dtype)))

    def estimate_task_memory(self, read_shape: tuple[int, ...]) -> int:
        working = estimate_working_memory(self.expression, input_shape=read_shape)
        output_shape = output_shape_for_input(self.expression, read_shape)
        checksum_copy = (
            int(np.prod(output_shape)) * np.dtype(self.expression.dtype).itemsize
        )
        return working + checksum_copy
