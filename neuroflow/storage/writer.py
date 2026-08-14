"""Partition-local durable writes."""

from __future__ import annotations

import hashlib
import uuid
from typing import cast

import fsspec
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from neuroflow.partition.base import Partition
from neuroflow.provenance.hashing import stable_hash
from neuroflow.storage.base import join_uri, write_json_atomic
from neuroflow.storage.manifest import PartitionManifest


def _write_parquet_atomic(uri: str, table: pa.Table) -> tuple[str, int]:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    payload = sink.getvalue().to_pybytes()
    fs, path = fsspec.core.url_to_fs(uri)
    parent = path.rsplit("/", 1)[0]
    fs.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with fs.open(temporary, "wb") as stream:
            stream.write(payload)
        if fs.exists(path):
            fs.rm(path)
        fs.mv(temporary, path)
    finally:
        if fs.exists(temporary):
            fs.rm(temporary)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _partition_shapes(partition: Partition) -> tuple[tuple[int, ...], tuple[int, ...]]:
    expected = tuple(
        (item.stop or 0) - (item.start or 0) for item in partition.output_slices
    )
    read_shape = tuple(
        (item.stop or 0) - (item.start or 0) for item in partition.read_slices
    )
    return expected, read_shape


class ArrayPartitionWriter:
    def __init__(
        self,
        *,
        uri: str,
        array_name: str,
        partition: Partition,
        workflow_id: str,
        partition_id: str,
        reduced_axis_indices: tuple[int, ...] = (),
        singleton_axis_indices: tuple[int, ...] = (),
    ) -> None:
        self.uri = uri
        self.array_name = array_name
        self.partition = partition
        self.workflow_id = workflow_id
        self.partition_id = partition_id
        self.reduced_axis_indices = reduced_axis_indices
        self.singleton_axis_indices = singleton_axis_indices

    def write_array(self, value: np.ndarray) -> PartitionManifest:
        expected, full_read_shape = _partition_shapes(self.partition)
        read_shape = tuple(
            1 if index in self.singleton_axis_indices else size
            for index, size in enumerate(full_read_shape)
            if index not in self.reduced_axis_indices
        )
        trim_slices = tuple(
            slice(0, 1) if index in self.singleton_axis_indices else item
            for index, item in enumerate(self.partition.trim_slices)
            if index not in self.reduced_axis_indices
        )
        if value.shape == read_shape:
            value = value[trim_slices]
        if value.shape != expected:
            raise ValueError(
                f"adapter returned shape {value.shape}; expected {expected} "
                f"or halo-inclusive {read_shape}"
            )
        mapper = fsspec.get_mapper(self.uri)
        group = zarr.open_group(mapper, mode="a")
        group[self.array_name][self.partition.output_slices] = value
        checksum = hashlib.sha256(value.tobytes(order="C")).hexdigest()
        manifest = PartitionManifest(
            partition_id=self.partition_id,
            workflow_id=self.workflow_id,
            status="complete",
            outputs={self.array_name: self.uri},
            checksums={self.array_name: checksum},
            sizes={self.array_name: int(value.nbytes)},
        )
        write_json_atomic(
            join_uri(
                self.uri,
                ".neuroflow",
                "manifests",
                f"{self.partition_id}.json",
            ),
            manifest.to_dict(),
        )
        return manifest


class TablePartitionWriter:
    def __init__(
        self,
        *,
        uri: str,
        table_name: str,
        workflow_id: str,
        partition_id: str,
        partition_on: tuple[str, ...] = (),
    ) -> None:
        self.uri = uri
        self.table_name = table_name
        self.workflow_id = workflow_id
        self.partition_id = partition_id
        self.partition_on = partition_on

    def write_table(self, value: pd.DataFrame | pa.Table) -> PartitionManifest:
        table = (
            pa.Table.from_pandas(value, preserve_index=False)
            if isinstance(value, pd.DataFrame)
            else value
        )
        missing = set(self.partition_on) - set(table.column_names)
        if missing:
            raise ValueError(
                "table is missing partition columns: " + ", ".join(sorted(missing))
            )
        destinations: list[tuple[str, pa.Table]] = []
        if self.partition_on and table.num_rows:
            frame = table.to_pandas()
            group_keys: str | list[str] = (
                self.partition_on[0]
                if len(self.partition_on) == 1
                else list(self.partition_on)
            )
            for key, group in frame.groupby(
                group_keys, dropna=False, sort=True, observed=True
            ):
                values = key if isinstance(key, tuple) else (key,)
                bucket = stable_hash(dict(zip(self.partition_on, values, strict=True)))[
                    :16
                ]
                destination = join_uri(
                    self.uri,
                    "tables",
                    self.table_name,
                    "partitions",
                    bucket,
                    f"part-{self.partition_id}.parquet",
                )
                destinations.append(
                    (destination, pa.Table.from_pandas(group, preserve_index=False))
                )
        else:
            destinations.append(
                (
                    join_uri(
                        self.uri,
                        "tables",
                        self.table_name,
                        f"part-{self.partition_id}.parquet",
                    ),
                    table,
                )
            )
        outputs: dict[str, str] = {}
        checksums: dict[str, str] = {}
        sizes: dict[str, int] = {}
        for index, (destination, partition_table) in enumerate(destinations):
            component = (
                self.table_name
                if len(destinations) == 1
                else f"{self.table_name}:{index}"
            )
            outputs[component] = destination
            checksum, size = _write_parquet_atomic(destination, partition_table)
            checksums[component] = checksum
            sizes[component] = size
        manifest = PartitionManifest(
            partition_id=self.partition_id,
            workflow_id=self.workflow_id,
            status="complete",
            outputs=outputs,
            checksums=checksums,
            sizes=sizes,
        )
        write_json_atomic(
            join_uri(
                self.uri,
                ".neuroflow",
                "manifests",
                f"{self.partition_id}.json",
            ),
            manifest.to_dict(),
        )
        return manifest


class SegmentationPartitionWriter:
    """Commit label and object outputs before one completion manifest."""

    def __init__(
        self,
        *,
        uri: str,
        labels_name: str,
        objects_name: str,
        partition: Partition,
        workflow_id: str,
        partition_id: str,
    ) -> None:
        self.uri = uri
        self.labels_name = labels_name
        self.objects_name = objects_name
        self.partition = partition
        self.workflow_id = workflow_id
        self.partition_id = partition_id

    def write_segmentation(
        self, labels: np.ndarray, objects: pd.DataFrame
    ) -> PartitionManifest:
        expected, read_shape = _partition_shapes(self.partition)
        labels = np.asarray(labels)
        if labels.shape == read_shape:
            labels = labels[self.partition.trim_slices]
        if labels.shape != expected:
            raise ValueError(
                f"segmentation labels have shape {labels.shape}; expected {expected} "
                f"or halo-inclusive {read_shape}"
            )
        if labels.dtype.kind not in "iu" or np.any(labels < 0):
            raise ValueError("segmentation labels must be non-negative integers")
        if "label_id" not in objects.columns:
            raise ValueError("segmentation object table requires a label_id column")
        local_ids = {int(value) for value in np.unique(labels) if int(value) != 0}
        table_ids = {int(value) for value in objects["label_id"].tolist()}
        missing_ids = local_ids - table_ids
        if missing_ids:
            raise ValueError(
                "object table is missing nonzero label IDs: "
                + ", ".join(str(value) for value in sorted(missing_ids))
            )
        objects = cast(
            pd.DataFrame,
            objects.loc[objects["label_id"].isin(list(local_ids))].copy(),
        )
        if any(value <= 0 or value >= 2**32 for value in local_ids):
            raise ValueError("local segmentation label IDs must be in [1, 2**32)")
        try:
            tile_index = int(self.partition.key.rsplit("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "segmentation partitions require a numeric tile key"
            ) from exc
        if tile_index >= 2**32 - 1:
            raise ValueError("segmentation tile index exceeds the global ID namespace")
        namespace = np.uint64(tile_index + 1) << np.uint64(32)
        global_labels = labels.astype(np.uint64, copy=True)
        foreground = global_labels != 0
        global_labels[foreground] += namespace
        global_objects = objects.copy()
        global_objects["local_label_id"] = global_objects["label_id"].astype("uint64")
        global_objects["label_id"] = (
            global_objects["label_id"].astype("uint64") + namespace
        )
        global_objects["tile_id"] = self.partition.key
        global_objects["partition_id"] = self.partition_id
        for axis, item in enumerate(self.partition.read_slices):
            offset = item.start or 0
            for prefix in ("centroid", "bbox_min", "bbox_max"):
                column = f"{prefix}_{axis}"
                if column in global_objects.columns:
                    global_objects[column] = global_objects[column] + offset

        mapper = fsspec.get_mapper(self.uri)
        group = zarr.open_group(mapper, mode="a")
        label_array = group[self.labels_name]
        if not isinstance(label_array, zarr.Array):
            raise TypeError("segmentation label component is not an array")
        label_array[self.partition.output_slices] = global_labels
        table = pa.Table.from_pandas(global_objects, preserve_index=False)
        table_uri = join_uri(
            self.uri,
            "tables",
            self.objects_name,
            f"part-{self.partition_id}.parquet",
        )
        table_checksum, table_size = _write_parquet_atomic(table_uri, table)
        manifest = PartitionManifest(
            partition_id=self.partition_id,
            workflow_id=self.workflow_id,
            status="complete",
            outputs={self.labels_name: self.uri, self.objects_name: table_uri},
            checksums={
                self.labels_name: hashlib.sha256(
                    global_labels.tobytes(order="C")
                ).hexdigest(),
                self.objects_name: table_checksum,
            },
            sizes={
                self.labels_name: int(global_labels.nbytes),
                self.objects_name: table_size,
            },
        )
        write_json_atomic(
            join_uri(
                self.uri,
                ".neuroflow",
                "manifests",
                f"{self.partition_id}.json",
            ),
            manifest.to_dict(),
        )
        return manifest
