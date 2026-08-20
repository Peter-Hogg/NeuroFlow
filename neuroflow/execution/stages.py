"""Bounded, resumable partial reductions for expression dependencies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import cast

import numpy as np

from neuroflow.diagnostics.estimates import slice_shape
from neuroflow.execution.resources import parse_bytes
from neuroflow.expression import (
    CastExpr,
    Expression,
    InputExpr,
    ReductionExpr,
    ScalarExpr,
    UFuncExpr,
    estimate_working_memory,
    evaluate_numpy,
    expression_identity,
    expression_to_dict,
    staged_reductions,
)
from neuroflow.partition.base import Partition
from neuroflow.provenance.hashing import stable_hash
from neuroflow.selection.query import Selection
from neuroflow.storage.base import join_uri, read_json, write_json_atomic

DEFAULT_STAGE_MEMORY_LIMIT = 1024**3
STAGE_SCHEMA_VERSION = "1"
MAX_STAGE_PARTITIONS = 1_000_000


@dataclass(frozen=True)
class ReductionStagePlan:
    stage_id: str
    reduction: ReductionExpr
    partitions: tuple[Partition, ...]
    maximum_partition_shape: tuple[int, ...]
    memory_per_partition: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": STAGE_SCHEMA_VERSION,
            "stage_id": self.stage_id,
            "kind": "global-reduction",
            "operation": self.reduction.operation,
            "dtype": self.reduction.dtype,
            "expression": expression_to_dict(self.reduction),
            "task_count": len(self.partitions),
            "maximum_partition_shape": list(self.maximum_partition_shape),
            "memory_per_partition": self.memory_per_partition,
            "bounded": True,
            "partitions": [item.to_dict() for item in self.partitions],
        }


def build_reduction_stage_plans(
    selection: Selection,
    expression: Expression,
    *,
    memory_limit: int | str | None,
) -> tuple[ReductionStagePlan, ...]:
    """Plan bounded source tiles for each supported scalar dependency."""
    budget = (
        DEFAULT_STAGE_MEMORY_LIMIT
        if memory_limit is None
        else parse_bytes(memory_limit)
    )
    plans: list[ReductionStagePlan] = []
    for reduction in staged_reductions(expression):
        _validate_stage_operand(reduction.operand)
        tile = list(selection.metadata.native_chunks or selection.metadata.shape)
        tile = [
            min(size, chunk)
            for size, chunk in zip(selection.metadata.shape, tile, strict=True)
        ]
        while (
            estimate_working_memory(reduction.operand, input_shape=tuple(tile)) > budget
        ):
            candidates = [index for index, size in enumerate(tile) if size > 1]
            if not candidates:
                required = estimate_working_memory(
                    reduction.operand, input_shape=tuple(tile)
                )
                raise ValueError(
                    f"global {reduction.operation} needs at least {required} bytes "
                    f"for one logical element, exceeding the {budget}-byte "
                    "workflow memory limit"
                )
            largest = max(candidates, key=lambda index: tile[index])
            tile[largest] = max(1, math.ceil(tile[largest] / 2))
        partitions = _tile_partitions(selection.metadata.shape, tuple(tile))
        memory = max(
            estimate_working_memory(
                reduction.operand, input_shape=slice_shape(item.read_slices)
            )
            for item in partitions
        )
        plans.append(
            ReductionStagePlan(
                expression_identity(reduction),
                reduction,
                partitions,
                tuple(tile),
                memory,
            )
        )
    return tuple(plans)


def execute_reduction_stages(
    *,
    selection: Selection,
    staged_values: dict[str, object],
    output_uri: str,
    workflow_id: str,
    plans: tuple[ReductionStagePlan, ...],
) -> tuple[dict[str, object], ...]:
    """Execute or resume each stage and populate its in-memory scalar cache."""
    records: list[dict[str, object]] = []
    for plan in plans:
        partials: list[dict[str, object]] = []
        skipped = 0
        recomputed = 0
        for partition in plan.partitions:
            partition_id = _stage_partition_identity(
                workflow_id, plan.stage_id, partition
            )
            uri = _stage_partial_uri(output_uri, plan.stage_id, partition_id)
            existing = read_json(uri)
            if _valid_partial(
                existing,
                workflow_id=workflow_id,
                stage_id=plan.stage_id,
                partition_id=partition_id,
                partition=partition,
            ):
                skipped += 1
                partials.append(existing)  # type: ignore[arg-type]
                continue
            data = np.asarray(selection._array[partition.read_slices])
            operand = evaluate_numpy(
                plan.reduction.operand,
                data,
                staged_values=staged_values,
            )
            payload = _partial_payload(
                workflow_id=workflow_id,
                plan=plan,
                partition=partition,
                partition_id=partition_id,
                value=operand,
            )
            write_json_atomic(uri, payload)
            partials.append(payload)
            recomputed += 1
        value = _combine_partials(plan.reduction, partials)
        staged_values[plan.stage_id] = value
        result_payload: dict[str, object] = {
            "schema_version": STAGE_SCHEMA_VERSION,
            "workflow_id": workflow_id,
            "stage_id": plan.stage_id,
            "operation": plan.reduction.operation,
            "value": _encode_scalar(value),
            "task_count": len(plan.partitions),
            "skipped_partitions": skipped,
            "computed_partitions": recomputed,
            "partial_ids": [str(partial["partition_id"]) for partial in partials],
        }
        result_payload["checksum"] = stable_hash(result_payload)
        write_json_atomic(
            join_uri(output_uri, ".neuroflow", "stages", plan.stage_id, "result.json"),
            result_payload,
        )
        records.append(
            {
                "stage_id": plan.stage_id,
                "operation": plan.reduction.operation,
                "task_count": len(plan.partitions),
                "skipped_partitions": skipped,
                "computed_partitions": recomputed,
                "status": "complete",
            }
        )
    return tuple(records)


def verify_reduction_stages(
    output_uri: str, provenance: dict[str, object]
) -> tuple[str, ...]:
    """Validate retained stage partials and scalar result checksums."""
    raw_stages = provenance.get("stages", [])
    if not isinstance(raw_stages, list):
        return ("invalid staged-reduction provenance",)
    workflow_id = provenance.get("workflow_id")
    if not isinstance(workflow_id, str):
        return ("staged reductions have no workflow identity",) if raw_stages else ()
    errors: list[str] = []
    for raw_stage in raw_stages:
        if not isinstance(raw_stage, dict):
            errors.append("invalid staged-reduction descriptor")
            continue
        stage_id = raw_stage.get("stage_id")
        raw_partitions = raw_stage.get("partitions")
        if not isinstance(stage_id, str) or not isinstance(raw_partitions, list):
            errors.append("invalid staged-reduction identity or partitions")
            continue
        partial_ids: list[str] = []
        for raw_partition in raw_partitions:
            try:
                if not isinstance(raw_partition, dict):
                    raise ValueError("descriptor is not an object")
                partition = Partition.from_dict(raw_partition)
                partition_id = _stage_partition_identity(
                    workflow_id, stage_id, partition
                )
            except (KeyError, ValueError) as exc:
                errors.append(f"stage {stage_id}: invalid partition: {exc}")
                continue
            partial_ids.append(partition_id)
            partial = read_json(_stage_partial_uri(output_uri, stage_id, partition_id))
            if not _valid_partial(
                partial,
                workflow_id=workflow_id,
                stage_id=stage_id,
                partition_id=partition_id,
                partition=partition,
            ):
                errors.append(
                    f"stage {stage_id}: missing or corrupt partial {partition_id}"
                )
        result = read_json(
            join_uri(output_uri, ".neuroflow", "stages", stage_id, "result.json")
        )
        if not isinstance(result, dict):
            errors.append(f"stage {stage_id}: missing scalar result")
            continue
        checksum = result.get("checksum")
        unsigned = {key: value for key, value in result.items() if key != "checksum"}
        if (
            not isinstance(checksum, str)
            or checksum != stable_hash(unsigned)
            or result.get("workflow_id") != workflow_id
            or result.get("stage_id") != stage_id
            or result.get("partial_ids") != partial_ids
        ):
            errors.append(f"stage {stage_id}: corrupt scalar result")
            continue
        try:
            _decode_scalar(result.get("value"))
        except ValueError as exc:
            errors.append(f"stage {stage_id}: invalid scalar result: {exc}")
    return tuple(errors)


def _tile_partitions(
    shape: tuple[int, ...], tile_shape: tuple[int, ...]
) -> tuple[Partition, ...]:
    counts = [
        math.ceil(size / tile) for size, tile in zip(shape, tile_shape, strict=True)
    ]
    task_count = math.prod(counts)
    if task_count > MAX_STAGE_PARTITIONS:
        raise ValueError(
            f"global reduction would create {task_count} partials; increase the "
            "stage memory limit or rechunk the source"
        )
    starts = [
        range(0, size, tile) for size, tile in zip(shape, tile_shape, strict=True)
    ]
    partitions: list[Partition] = []
    for number, coordinates in enumerate(product(*starts)):
        slices = tuple(
            slice(start, min(size, start + tile))
            for start, size, tile in zip(coordinates, shape, tile_shape, strict=True)
        )
        partitions.append(
            Partition(
                key=f"stage-{number:08d}",
                read_slices=slices,
                output_slices=slices,
                trim_slices=tuple(slice(0, size) for size in slice_shape(slices)),
                coordinates=tuple(int(value) for value in coordinates),
            )
        )
    return tuple(partitions)


def _validate_stage_operand(expression: Expression) -> None:
    if isinstance(expression, (InputExpr, ScalarExpr)):
        return
    if isinstance(expression, CastExpr):
        _validate_stage_operand(expression.operand)
        return
    if isinstance(expression, UFuncExpr):
        for operand in expression.operands:
            _validate_stage_operand(operand)
        return
    if isinstance(expression, ReductionExpr) and not expression.shape:
        return
    raise ValueError(
        "a staged global reduction may contain elementwise operations, casts, "
        "and other global scalar dependencies, but not a nested non-global "
        "reduction; persist the intermediate reduction explicitly"
    )


def _partial_payload(
    *,
    workflow_id: str,
    plan: ReductionStagePlan,
    partition: Partition,
    partition_id: str,
    value: np.ndarray,
) -> dict[str, object]:
    operation = plan.reduction.operation
    target_dtype = np.dtype(plan.reduction.dtype)
    if operation == "mean":
        partial_value = np.sum(value, dtype=target_dtype)
        partial: dict[str, object] = {
            "sum": _encode_scalar(partial_value),
            "count": int(value.size),
        }
    else:
        function = getattr(np, operation)
        kwargs = {"dtype": target_dtype} if operation == "sum" else {}
        partial = {"value": _encode_scalar(function(value, **kwargs))}
    payload: dict[str, object] = {
        "schema_version": STAGE_SCHEMA_VERSION,
        "workflow_id": workflow_id,
        "stage_id": plan.stage_id,
        "partition_id": partition_id,
        "partition": partition.to_dict(),
        "operation": operation,
        "partial": partial,
    }
    payload["checksum"] = stable_hash(payload)
    return payload


def _combine_partials(
    reduction: ReductionExpr, partials: list[dict[str, object]]
) -> object:
    dtype = np.dtype(reduction.dtype)
    if reduction.operation == "mean":
        sums = [_decode_scalar(_mapping(item["partial"])["sum"]) for item in partials]
        counts = [_mapping(item["partial"])["count"] for item in partials]
        if not all(
            isinstance(item, int) and not isinstance(item, bool) for item in counts
        ):
            raise ValueError("mean partial counts must be integers")
        count = sum(cast(int, item) for item in counts)
        total = np.sum(np.asarray(sums, dtype=dtype), dtype=dtype)
        return np.asarray(total / count, dtype=dtype)[()]
    values = [_decode_scalar(_mapping(item["partial"])["value"]) for item in partials]
    array = np.asarray(values, dtype=dtype)
    if reduction.operation == "sum":
        return np.asarray(np.sum(array, dtype=dtype), dtype=dtype)[()]
    return np.asarray(getattr(np, reduction.operation)(array), dtype=dtype)[()]


def _valid_partial(
    value: dict[str, object] | None,
    *,
    workflow_id: str,
    stage_id: str,
    partition_id: str,
    partition: Partition,
) -> bool:
    if not isinstance(value, dict):
        return False
    checksum = value.get("checksum")
    unsigned = {key: item for key, item in value.items() if key != "checksum"}
    if (
        not isinstance(checksum, str)
        or checksum != stable_hash(unsigned)
        or value.get("schema_version") != STAGE_SCHEMA_VERSION
        or value.get("workflow_id") != workflow_id
        or value.get("stage_id") != stage_id
        or value.get("partition_id") != partition_id
        or value.get("partition") != partition.to_dict()
    ):
        return False
    try:
        partial = _mapping(value["partial"])
        if value.get("operation") == "mean":
            _decode_scalar(partial["sum"])
            count = partial["count"]
            return isinstance(count, int) and not isinstance(count, bool) and count > 0
        _decode_scalar(partial["value"])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _encode_scalar(value: object) -> dict[str, object]:
    scalar = np.asarray(value)
    if scalar.shape != () or scalar.dtype.hasobject:
        raise ValueError("staged reductions must produce one non-object scalar")
    return {"dtype": scalar.dtype.str, "bytes": scalar.tobytes().hex()}


def _decode_scalar(value: object) -> np.generic:
    mapping = _mapping(value)
    dtype_value = mapping.get("dtype")
    bytes_value = mapping.get("bytes")
    if not isinstance(dtype_value, str) or not isinstance(bytes_value, str):
        raise ValueError("encoded scalar needs dtype and bytes")
    dtype = np.dtype(dtype_value)
    if dtype.hasobject or len(bytes_value) != dtype.itemsize * 2:
        raise ValueError("encoded scalar has an unsafe dtype or incorrect size")
    try:
        return np.frombuffer(bytes.fromhex(bytes_value), dtype=dtype, count=1)[0]
    except ValueError as exc:
        raise ValueError("encoded scalar contains invalid bytes") from exc


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("value must be an object")
    return value


def _stage_partition_identity(
    workflow_id: str, stage_id: str, partition: Partition
) -> str:
    return stable_hash(
        {
            "workflow_id": workflow_id,
            "stage_id": stage_id,
            "partition": partition.to_dict(),
        }
    )


def _stage_partial_uri(output_uri: str, stage_id: str, partition_id: str) -> str:
    return join_uri(
        output_uri,
        ".neuroflow",
        "stages",
        stage_id,
        "partials",
        f"{partition_id}.json",
    )
