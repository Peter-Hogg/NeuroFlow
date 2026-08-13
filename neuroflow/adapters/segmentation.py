"""Algorithm-neutral adapter for tiled segmentation functions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from neuroflow.adapters.base import AdapterRequirements, LoadedPartition, TaskContext
from neuroflow.execution.resources import ResourceSpec


@dataclass(frozen=True)
class SegmentationOutputSchema:
    label_dtype: str = "uint64"
    labels_name: str = "labels"
    objects_name: str = "objects"

    def __post_init__(self) -> None:
        dtype = np.dtype(self.label_dtype)
        if dtype.kind != "u" or dtype.itemsize < 8:
            raise ValueError("segmentation labels require an unsigned 64-bit dtype")


@dataclass(frozen=True)
class SegmentationTaskOutput:
    labels: np.ndarray
    objects: pd.DataFrame


@dataclass(frozen=True)
class SegmentationFunctionAdapter:
    """Run a declared external segmentation function on bounded tiles."""

    function: Callable[..., object]
    output: SegmentationOutputSchema = SegmentationOutputSchema()
    name: str = "segmentation-function"
    version: str = "1"
    splittable_axes: tuple[str, ...] = ("z", "y", "x")
    requires_overlap: Mapping[str, int | str] | None = None
    parameters: Mapping[str, object] | None = None
    resources: ResourceSpec = ResourceSpec()
    deterministic: bool = True
    random_seed: int | None = None

    def requirements(self) -> AdapterRequirements:
        return AdapterRequirements(
            input_kinds=("array",),
            splittable_axes=self.splittable_axes,
            requires_overlap=self.requires_overlap or {},
            output_kinds=("labels", "objects"),
            resources=self.resources,
            deterministic=self.deterministic,
        )

    def prepare(self, partition: LoadedPartition, context: TaskContext) -> object:
        return partition

    def run(self, prepared: object, context: TaskContext) -> SegmentationTaskOutput:
        if not isinstance(prepared, LoadedPartition):
            raise TypeError("SegmentationFunctionAdapter expects a LoadedPartition")
        kwargs: dict[str, Any] = dict(self.parameters or {})
        value = self.function(np.asarray(prepared.data), **kwargs)
        if isinstance(value, SegmentationTaskOutput):
            return value
        if isinstance(value, tuple) and len(value) == 2:
            labels, objects = value
            table = (
                objects if isinstance(objects, pd.DataFrame) else pd.DataFrame(objects)
            )
            return SegmentationTaskOutput(np.asarray(labels), table)
        raise TypeError(
            "segmentation function must return SegmentationTaskOutput or "
            "(labels, objects)"
        )

    def persist(self, output: object, writer: object, context: TaskContext) -> object:
        if not isinstance(output, SegmentationTaskOutput):
            raise TypeError("segmentation adapter produced an invalid task output")
        write = getattr(writer, "write_segmentation", None)
        if write is None:
            raise TypeError(
                "segmentation adapter requires a composite partition writer"
            )
        return write(output.labels, output.objects)
