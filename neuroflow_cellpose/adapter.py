"""Cellpose 4.x adapter with deferred imports and thread-local model state."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
import pandas as pd

from neuroflow.adapters.base import AdapterRequirements, LoadedPartition, TaskContext
from neuroflow.adapters.segmentation import (
    SegmentationOutputSchema,
    SegmentationTaskOutput,
)
from neuroflow.exceptions import AdapterCompatibilityError
from neuroflow.execution.resources import ResourceSpec

_THREAD_STATE = threading.local()


class _CellposeModel(Protocol):
    def eval(self, value: np.ndarray, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class CellposeAdapter:
    """Apply a Cellpose 4.x model to one bounded spatial partition."""

    pretrained_model: str
    gpu: bool = False
    device: str | None = None
    use_bfloat16: bool = True
    batch_size: int = 8
    channels: tuple[int, int] | None = None
    channel_axis: int | None = None
    z_axis: int | None = None
    normalize: bool | Mapping[str, object] = True
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    do_3d: bool = False
    anisotropy: float | None = None
    min_size: int = 15
    tile_overlap: float = 0.1
    halo: Mapping[str, int] | None = None
    parameters: Mapping[str, object] | None = None
    output: SegmentationOutputSchema = SegmentationOutputSchema()
    name: str = "cellpose"
    version: str = "1"
    deterministic: bool = True
    random_seed: int | None = None
    cpu: int = 4
    memory: str = "16 GiB"

    external_packages: tuple[str, ...] = ("cellpose",)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.min_size < 0:
            raise ValueError("min_size cannot be negative")
        if not 0 <= self.tile_overlap <= 1:
            raise ValueError("tile_overlap must be between zero and one")

    def requirements(self) -> AdapterRequirements:
        return AdapterRequirements(
            input_kinds=("array",),
            splittable_axes=("z", "y", "x"),
            requires_overlap=self.halo or {},
            output_kinds=("labels", "objects"),
            resources=ResourceSpec(
                cpu=self.cpu,
                memory=self.memory,
                gpu=1 if self.gpu else 0,
            ),
            deterministic=self.deterministic,
        )

    def identity_parameters(self) -> Mapping[str, object]:
        return {
            "pretrained_model": self.pretrained_model,
            "gpu": self.gpu,
            "device": self.device,
            "use_bfloat16": self.use_bfloat16,
            "batch_size": self.batch_size,
            "channels": self.channels,
            "channel_axis": self.channel_axis,
            "z_axis": self.z_axis,
            "normalize": self.normalize,
            "diameter": self.diameter,
            "flow_threshold": self.flow_threshold,
            "cellprob_threshold": self.cellprob_threshold,
            "do_3d": self.do_3d,
            "anisotropy": self.anisotropy,
            "min_size": self.min_size,
            "tile_overlap": self.tile_overlap,
            "halo": dict(self.halo or {}),
            **dict(self.parameters or {}),
        }

    def prepare(self, partition: LoadedPartition, context: TaskContext) -> object:
        value = np.asarray(partition.data)
        if value.ndim < 2:
            raise AdapterCompatibilityError("Cellpose requires at least a 2D input")
        return LoadedPartition(
            data=value,
            read_slices=partition.read_slices,
            output_slices=partition.output_slices,
            trim_slices=partition.trim_slices,
        )

    def run(self, prepared: object, context: TaskContext) -> SegmentationTaskOutput:
        if not isinstance(prepared, LoadedPartition):
            raise TypeError("CellposeAdapter expects a LoadedPartition")
        model = self._model()
        evaluation = model.eval(
            np.asarray(prepared.data),
            batch_size=self.batch_size,
            channels=list(self.channels) if self.channels is not None else None,
            channel_axis=self.channel_axis,
            z_axis=self.z_axis,
            normalize=self.normalize,
            diameter=self.diameter,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            do_3D=self.do_3d,
            anisotropy=self.anisotropy,
            min_size=self.min_size,
            tile_overlap=self.tile_overlap,
        )
        if not isinstance(evaluation, tuple) or not evaluation:
            raise TypeError("CellposeModel.eval() returned an unsupported value")
        labels = np.asarray(evaluation[0])
        if labels.shape != np.asarray(prepared.data).shape:
            raise AdapterCompatibilityError(
                "Cellpose masks must match the prepared partition shape"
            )
        return SegmentationTaskOutput(labels, _object_table(labels))

    def persist(self, output: object, writer: object, context: TaskContext) -> object:
        if not isinstance(output, SegmentationTaskOutput):
            raise TypeError("CellposeAdapter produced an invalid task output")
        write = getattr(writer, "write_segmentation", None)
        if write is None:
            raise TypeError("CellposeAdapter requires a segmentation writer")
        return write(output.labels, output.objects)

    def _model(self) -> _CellposeModel:
        cache = getattr(_THREAD_STATE, "models", None)
        if cache is None:
            cache = {}
            _THREAD_STATE.models = cache
        key = (
            self.pretrained_model,
            self.gpu,
            self.device,
            self.use_bfloat16,
        )
        if key not in cache:
            try:
                models = importlib.import_module("cellpose.models")
            except ImportError as exc:
                raise AdapterCompatibilityError(
                    "Cellpose is optional; install NeuroFlow with the 'cellpose' extra"
                ) from exc
            model_type = getattr(models, "CellposeModel", None)
            if model_type is None:
                raise AdapterCompatibilityError(
                    "installed Cellpose has no CellposeModel API"
                )
            cache[key] = model_type(
                gpu=self.gpu,
                pretrained_model=self.pretrained_model,
                device=self.device,
                use_bfloat16=self.use_bfloat16,
            )
        return cast(_CellposeModel, cache[key])


def _object_table(labels: np.ndarray) -> pd.DataFrame:
    records: list[dict[str, int | float]] = []
    for value in np.unique(labels):
        label_id = int(value)
        if label_id == 0:
            continue
        coordinates = np.nonzero(labels == value)
        record: dict[str, int | float] = {
            "label_id": label_id,
            "voxel_count": int(len(coordinates[0])),
        }
        for axis, coordinate in enumerate(coordinates):
            record[f"centroid_{axis}"] = float(coordinate.mean())
            record[f"bbox_min_{axis}"] = int(coordinate.min())
            record[f"bbox_max_{axis}"] = int(coordinate.max()) + 1
        records.append(record)
    columns = ["label_id", "voxel_count"]
    columns.extend(
        name
        for axis in range(labels.ndim)
        for name in (
            f"centroid_{axis}",
            f"bbox_min_{axis}",
            f"bbox_max_{axis}",
        )
    )
    return pd.DataFrame.from_records(records, columns=columns)
