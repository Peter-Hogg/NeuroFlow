"""Safe, versioned, portable NumPy-expression workflow specifications."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from neuroflow._version import __version__
from neuroflow.adapters.numpy import ArrayOutput, ExpressionAdapter
from neuroflow.exceptions import WorkflowSpecError
from neuroflow.expression import (
    expression_from_dict,
    expression_to_dict,
    input_expression,
)
from neuroflow.partition import (
    AssetPlan,
    PartitionPlan,
    SessionPlan,
    SpatialTilePlan,
    TimeWindowPlan,
)
from neuroflow.selection import NWBQuery, Selection, absolute_selection_bounds
from neuroflow.storage import ZarrOutput

if TYPE_CHECKING:
    from neuroflow.diagnostics.plan import ExecutionPlan
    from neuroflow.results.workflow import WorkflowResult

WORKFLOW_SCHEMA_VERSION = "1"
MAX_WORKFLOW_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True)
class WorkflowSpec:
    """Portable declaration of one allowlisted NumPy-expression workflow."""

    source: dict[str, object]
    selection: dict[str, object]
    expression: dict[str, object]
    adapter: dict[str, object]
    partition: dict[str, object]
    output: dict[str, object]
    execution: dict[str, object]
    provenance: dict[str, object]
    schema_version: str = WORKFLOW_SCHEMA_VERSION
    kind: Literal["numpy-expression"] = "numpy-expression"

    def __post_init__(self) -> None:
        _validate_spec_dict(self.to_dict(validate=False))

    @classmethod
    def from_workflow(cls, workflow: WorkflowResult) -> WorkflowSpec:
        """Create a portable spec without serializing a Python callable."""
        if not isinstance(workflow.adapter, ExpressionAdapter):
            raise WorkflowSpecError(
                "portable workflows support only the allowlisted NumPy-expression "
                "adapter; arbitrary Python functions cannot be serialized safely"
            )
        if not isinstance(workflow.output, ZarrOutput):
            raise WorkflowSpecError("NumPy-expression workflows require Zarr output")
        if workflow.partition is None:
            raise WorkflowSpecError("workflow has no serializable partition strategy")
        metadata = workflow.selection.metadata
        attributes = metadata.attributes or {}
        if metadata.source.uri.startswith("DANDI:"):
            source_type = "dandi"
        elif attributes.get("backend") == "zarr-array":
            source_type = "neuroflow-array"
        elif attributes.get("backend") == "nwb-hdf5":
            source_type = "nwb-hdf5"
        else:
            source_type = "nwb-zarr"
        provenance = workflow.provenance or {}
        spec = cls(
            source={
                "type": source_type,
                "uri": metadata.source.uri,
                "version": metadata.source.version,
                "asset_id": metadata.source.asset_id,
                "checksum": metadata.source.checksum,
                "backend": attributes.get("transport"),
            },
            selection={
                "path": metadata.path,
                "name": metadata.name,
                "neurodata_type": metadata.neurodata_type,
                "bounds": [list(item) for item in absolute_selection_bounds(metadata)],
                "shape": list(metadata.shape),
                "dtype": np.dtype(metadata.dtype).str,
                "native_chunks": (
                    list(metadata.native_chunks)
                    if metadata.native_chunks is not None
                    else None
                ),
                "axes": list(metadata.axes),
            },
            expression=expression_to_dict(workflow.adapter.expression),
            adapter={
                "identifier": "neuroflow.numpy-expression",
                "version": workflow.adapter.version,
                "splittable_axes": list(workflow.adapter.splittable_axes),
                "output": {
                    "name": workflow.adapter.output.name,
                    "dtype": np.dtype(workflow.adapter.output.dtype).str,
                    "reduced_axes": list(workflow.adapter.output.reduced_axes),
                    "kept_reduced_axes": list(
                        workflow.adapter.output.kept_reduced_axes
                    ),
                    "chunks": (
                        list(workflow.adapter.output.chunks)
                        if workflow.adapter.output.chunks is not None
                        else None
                    ),
                },
            },
            partition=_partition_to_dict(workflow.partition),
            output={
                "type": "zarr",
                "uri": workflow.output.uri,
                # Portable files never grant overwrite authority. A caller may
                # choose a fresh output override when reproducing.
                "mode": "create",
                "compressor": workflow.output.compressor,
            },
            execution={
                "scheduler": workflow.scheduler,
                "resume": workflow.resume_enabled,
                "max_workers": workflow.max_workers,
                "memory_limit": workflow.memory_limit,
            },
            provenance={
                "generated_by": {"name": "neuroflow", "version": __version__},
                "original_workflow_id": workflow.plan.workflow_id,
                "original_status": provenance.get("status", "planned"),
            },
        )
        return spec

    @classmethod
    def from_dict(cls, value: object) -> WorkflowSpec:
        validated = _validate_spec_dict(value)
        return cls(
            source=_mapping(validated["source"], "source"),
            selection=_mapping(validated["selection"], "selection"),
            expression=_mapping(validated["expression"], "expression"),
            adapter=_mapping(validated["adapter"], "adapter"),
            partition=_mapping(validated["partition"], "partition"),
            output=_mapping(validated["output"], "output"),
            execution=_mapping(validated["execution"], "execution"),
            provenance=_mapping(validated["provenance"], "provenance"),
            schema_version=_string(validated["schema_version"], "schema version"),
            kind=cast(Literal["numpy-expression"], validated["kind"]),
        )

    @classmethod
    def from_json(cls, source: str | Path) -> WorkflowSpec:
        """Load a JSON string or a bounded local JSON file."""
        if isinstance(source, Path) or not str(source).lstrip().startswith("{"):
            path = Path(source)
            if not path.is_file():
                raise WorkflowSpecError(f"workflow file does not exist: {path}")
            if path.stat().st_size > MAX_WORKFLOW_BYTES:
                raise WorkflowSpecError("workflow file exceeds the 5 MiB safety limit")
            text = path.read_text(encoding="utf-8")
        else:
            text = str(source)
            if len(text.encode()) > MAX_WORKFLOW_BYTES:
                raise WorkflowSpecError("workflow JSON exceeds the 5 MiB safety limit")
        try:
            value = json.loads(text, object_pairs_hook=_without_duplicate_keys)
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            raise WorkflowSpecError(f"invalid workflow JSON: {exc}") from exc
        return cls.from_dict(value)

    def to_dict(self, *, validate: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source": self.source,
            "selection": self.selection,
            "expression": self.expression,
            "adapter": self.adapter,
            "partition": self.partition,
            "output": self.output,
            "execution": self.execution,
            "provenance": self.provenance,
        }
        if validate:
            _validate_spec_dict(value)
        return cast(dict[str, object], _json_copy(value))

    def to_json(
        self, destination: str | Path | None = None, *, overwrite: bool = False
    ) -> str:
        """Return deterministic JSON and optionally create a workflow file."""
        payload = (
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        if destination is not None:
            path = Path(destination)
            if path.is_symlink():
                raise WorkflowSpecError(
                    "refusing to write a workflow through a symlink"
                )
            mode = "w" if overwrite else "x"
            try:
                with path.open(mode, encoding="utf-8") as stream:
                    stream.write(payload)
            except FileExistsError as exc:
                raise WorkflowSpecError(
                    f"workflow file already exists: {path}; pass overwrite=True "
                    "to replace it explicitly"
                ) from exc
        return payload

    def reproduce(
        self,
        *,
        output: str | Path | None = None,
        storage_options: dict[str, object] | None = None,
        execute: bool = True,
    ) -> WorkflowResult:
        return reproduce(
            self,
            output=output,
            storage_options=storage_options,
            execute=execute,
        )

    def plan(
        self, *, storage_options: dict[str, object] | None = None
    ) -> ExecutionPlan:
        workflow = reproduce(self, storage_options=storage_options, execute=False)
        try:
            return workflow.plan
        finally:
            workflow.source.close()


def reproduce(
    spec: WorkflowSpec | str | Path,
    *,
    output: str | Path | None = None,
    storage_options: dict[str, object] | None = None,
    execute: bool = True,
) -> WorkflowResult:
    """Safely rebuild and optionally execute a portable workflow."""
    from neuroflow.api import open_array, open_source, run

    value = spec if isinstance(spec, WorkflowSpec) else WorkflowSpec.from_json(spec)
    source_record = value.source
    source_type = source_record["type"]
    source_uri = str(source_record["uri"])
    version = source_record.get("version")
    if version is not None and not isinstance(version, str):
        raise WorkflowSpecError("source version must be a string or null")
    if source_type == "neuroflow-array":
        component = source_record.get("asset_id")
        if not isinstance(component, str) or not component:
            raise WorkflowSpecError("NeuroFlow array source requires a component")
        source, base_selection = open_array(source_uri, component=component)
    else:
        resolved_storage_options = dict(storage_options or {})
        backend = source_record.get("backend")
        if backend in {"lindi", "remfile", "fsspec"}:
            requested_backend = resolved_storage_options.get("transport")
            if requested_backend not in (None, "auto", backend):
                raise WorkflowSpecError(
                    "storage transport conflicts with the workflow backend"
                )
            resolved_storage_options["transport"] = backend
        source = open_source(
            source_uri,
            version=version,
            storage_options=resolved_storage_options or None,
        )
        base_selection = source.select(
            NWBQuery(
                path=str(value.selection["path"]),
                asset=(
                    str(source_record["asset_id"])
                    if source_record.get("asset_id") is not None
                    else None
                ),
            )
        )
    try:
        selection = _apply_and_validate_selection(base_selection, value)
        expression = expression_from_dict(value.expression)
        expression_input = input_expression(expression)
        if (
            expression_input.shape != selection.metadata.shape
            or expression_input.axes != selection.metadata.axes
            or np.dtype(expression_input.dtype) != np.dtype(selection.metadata.dtype)
        ):
            raise WorkflowSpecError(
                "expression input does not match the resolved source selection"
            )
        adapter_output = _mapping(value.adapter["output"], "adapter output")
        chunks_value = adapter_output.get("chunks")
        chunks = (
            None
            if chunks_value is None
            else _positive_int_tuple(chunks_value, "chunks")
        )
        adapter = ExpressionAdapter(
            expression=expression,
            output=ArrayOutput(
                dtype=_dtype(adapter_output.get("dtype")),
                name=_string(adapter_output.get("name"), "output name"),
                reduced_axes=_string_tuple(
                    adapter_output.get("reduced_axes"), "reduced_axes"
                ),
                kept_reduced_axes=_string_tuple(
                    adapter_output.get("kept_reduced_axes"), "kept_reduced_axes"
                ),
                chunks=chunks,
            ),
            splittable_axes=_string_tuple(
                value.adapter.get("splittable_axes"), "splittable_axes"
            ),
            version=_string(value.adapter.get("version"), "adapter version"),
        )
        partition = _partition_from_dict(value.partition)
        output_record = value.output
        execution = value.execution
        result = run(
            source=source,
            selection=selection,
            adapter=adapter,
            partition=partition,
            output=ZarrOutput(
                str(output) if output is not None else str(output_record["uri"]),
                mode="create",
                compressor=cast(Any, output_record.get("compressor", "default")),
            ),
            scheduler=cast(Any, execution.get("scheduler", "threads")),
            resume=bool(execution.get("resume", True)),
            execute=execute,
            max_workers=cast(int | None, execution.get("max_workers")),
            memory_limit=cast(int | str | None, execution.get("memory_limit")),
        )
    except Exception:
        source.close()
        raise
    return result


def _apply_and_validate_selection(base: Selection, spec: WorkflowSpec) -> Selection:
    if not isinstance(base, Selection):
        raise WorkflowSpecError("source did not resolve to a NeuroFlow selection")
    axes = _string_tuple(spec.selection.get("axes"), "selection axes")
    bounds = _bounds(spec.selection.get("bounds"), len(axes))
    if axes != base.metadata.axes:
        raise WorkflowSpecError(
            f"resolved source axes {base.metadata.axes} do not match workflow "
            f"axes {axes}"
        )
    for axis, (_, stop), extent in zip(axes, bounds, base.metadata.shape, strict=True):
        if stop > extent:
            raise WorkflowSpecError(
                f"selection bound for {axis!r} extends past source extent {extent}"
            )
    selection = base.isel(
        **{
            axis: slice(start, stop)
            for axis, (start, stop) in zip(axes, bounds, strict=True)
        }
    )
    expected_shape = _positive_int_tuple(spec.selection.get("shape"), "shape")
    expected_chunks_value = spec.selection.get("native_chunks")
    expected_chunks = (
        None
        if expected_chunks_value is None
        else _positive_int_tuple(expected_chunks_value, "native_chunks")
    )
    actual_source = selection.metadata.source
    for key, actual in (
        ("uri", actual_source.uri),
        ("version", actual_source.version),
        ("asset_id", actual_source.asset_id),
        ("checksum", actual_source.checksum),
    ):
        expected = spec.source.get(key)
        if expected is not None and expected != actual:
            raise WorkflowSpecError(
                f"resolved source {key} does not match the workflow specification"
            )
    expected_backend = spec.source.get("backend")
    actual_backend = (selection.metadata.attributes or {}).get("transport")
    if expected_backend is not None and expected_backend != actual_backend:
        raise WorkflowSpecError(
            "resolved source backend does not match the workflow specification"
        )
    if (
        selection.metadata.shape != expected_shape
        or np.dtype(selection.metadata.dtype)
        != np.dtype(_dtype(spec.selection.get("dtype")))
        or selection.metadata.native_chunks != expected_chunks
        or selection.metadata.path != spec.selection.get("path")
    ):
        raise WorkflowSpecError(
            "resolved selection metadata does not match the workflow specification"
        )
    return selection


def _validate_spec_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise WorkflowSpecError("workflow specification must be a JSON object")
    required = {
        "schema_version",
        "kind",
        "source",
        "selection",
        "expression",
        "adapter",
        "partition",
        "output",
        "execution",
        "provenance",
    }
    if set(value) != required:
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - required)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        raise WorkflowSpecError("invalid workflow fields (" + "; ".join(details) + ")")
    version = value.get("schema_version")
    if version != WORKFLOW_SCHEMA_VERSION:
        raise WorkflowSpecError(
            f"unsupported workflow schema version {version!r}; this NeuroFlow "
            f"release supports version {WORKFLOW_SCHEMA_VERSION!r} and has no "
            "migration registered for the requested version"
        )
    if value.get("kind") != "numpy-expression":
        raise WorkflowSpecError("only kind='numpy-expression' is supported")
    source = _mapping(value["source"], "source")
    if source.get("type") not in {
        "dandi",
        "nwb-hdf5",
        "nwb-zarr",
        "neuroflow-array",
    }:
        raise WorkflowSpecError("source type is not allowlisted")
    _string(source.get("uri"), "source URI")
    if source.get("backend") not in {
        None,
        "local",
        "local-or-fsspec",
        "lindi",
        "remfile",
        "fsspec",
    }:
        raise WorkflowSpecError("source backend is not allowlisted")
    selection = _mapping(value["selection"], "selection")
    axes = _string_tuple(selection.get("axes"), "selection axes")
    shape = _positive_int_tuple(selection.get("shape"), "selection shape")
    if len(axes) != len(shape):
        raise WorkflowSpecError("selection axes and shape ranks differ")
    _bounds(selection.get("bounds"), len(shape))
    _dtype(selection.get("dtype"))
    _string(selection.get("path"), "selection path")
    if selection.get("native_chunks") is not None:
        chunks = _positive_int_tuple(selection["native_chunks"], "native_chunks")
        if len(chunks) != len(shape):
            raise WorkflowSpecError("native chunk rank does not match selection")
    try:
        expression_from_dict(value["expression"])
    except ValueError as exc:
        raise WorkflowSpecError(f"invalid expression: {exc}") from exc
    adapter = _mapping(value["adapter"], "adapter")
    if adapter.get("identifier") != "neuroflow.numpy-expression":
        raise WorkflowSpecError("adapter identifier is not allowlisted")
    _string(adapter.get("version"), "adapter version")
    _string_tuple(adapter.get("splittable_axes"), "splittable_axes")
    adapter_output = _mapping(adapter.get("output"), "adapter output")
    _string(adapter_output.get("name"), "output name")
    _dtype(adapter_output.get("dtype"))
    _string_tuple(adapter_output.get("reduced_axes"), "reduced_axes")
    _string_tuple(adapter_output.get("kept_reduced_axes"), "kept_reduced_axes")
    if adapter_output.get("chunks") is not None:
        _positive_int_tuple(adapter_output["chunks"], "output chunks")
    _partition_from_dict(_mapping(value["partition"], "partition"))
    output = _mapping(value["output"], "output")
    if output.get("type") != "zarr" or output.get("mode") != "create":
        raise WorkflowSpecError("portable workflows require create-only Zarr output")
    _string(output.get("uri"), "output URI")
    if output.get("compressor") not in {"default", "none"}:
        raise WorkflowSpecError("output compressor must be 'default' or 'none'")
    execution = _mapping(value["execution"], "execution")
    if execution.get("scheduler") not in {"threads", "processes", "distributed"}:
        raise WorkflowSpecError("execution scheduler is invalid")
    if not isinstance(execution.get("resume"), bool):
        raise WorkflowSpecError("execution resume flag must be a boolean")
    workers = execution.get("max_workers")
    if workers is not None and (
        not isinstance(workers, int) or isinstance(workers, bool) or workers < 1
    ):
        raise WorkflowSpecError("max_workers must be a positive integer or null")
    memory = execution.get("memory_limit")
    if memory is not None and (
        not isinstance(memory, (int, str)) or isinstance(memory, bool)
    ):
        raise WorkflowSpecError("memory_limit must be bytes, a size string, or null")
    _mapping(value["provenance"], "provenance")
    return cast(dict[str, object], _json_copy(value))


def _partition_to_dict(value: object) -> dict[str, object]:
    if isinstance(value, SpatialTilePlan):
        return {"type": "spatial-tile", **asdict(value)}
    if isinstance(value, TimeWindowPlan):
        return {"type": "time-window", **asdict(value)}
    if isinstance(value, AssetPlan):
        return {"type": "asset", **asdict(value)}
    if isinstance(value, SessionPlan):
        return {"type": "session", **asdict(value)}
    raise WorkflowSpecError(
        f"partition strategy {type(value).__name__!r} is not portable"
    )


def _partition_from_dict(value: dict[str, object]) -> PartitionPlan:
    kind = value.get("type")
    try:
        if kind == "spatial-tile":
            return SpatialTilePlan(
                _positive_int_tuple(value.get("tile_shape"), "tile_shape"),
                _nonnegative_int_tuple(value.get("halo"), "halo"),
                _string_tuple(value.get("axes"), "partition axes"),
            )
        if kind == "time-window":
            size = value.get("size")
            overlap = value.get("overlap")
            if not isinstance(size, (int, str)) or isinstance(size, bool):
                raise WorkflowSpecError(
                    "time window size must be an integer or duration"
                )
            if not isinstance(overlap, (int, str)) or isinstance(overlap, bool):
                raise WorkflowSpecError("time overlap must be an integer or duration")
            if value.get("units") not in {None, "samples"}:
                raise WorkflowSpecError("time window units must be 'samples' or null")
            if value.get("align_to") not in {None, "timestamps"}:
                raise WorkflowSpecError(
                    "time window alignment must be 'timestamps' or null"
                )
            return TimeWindowPlan(
                size=size,
                overlap=overlap,
                units=cast(Any, value.get("units")),
                align_to=cast(Any, value.get("align_to")),
            )
        if kind == "asset":
            return AssetPlan(filter=_mapping(value.get("filter", {}), "asset filter"))
        if kind == "session":
            return SessionPlan(
                group_by=_string_tuple(value.get("group_by"), "session grouping")
            )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, WorkflowSpecError):
            raise
        raise WorkflowSpecError(f"invalid {kind!r} partition: {exc}") from exc
    raise WorkflowSpecError(f"partition type {kind!r} is not allowlisted")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WorkflowSpecError(f"{name} must be a JSON object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 8192:
        raise WorkflowSpecError(f"{name} must be a non-empty bounded string")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise WorkflowSpecError(f"{name} must contain unique non-empty strings")
    return tuple(value)


def _positive_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in value
    ):
        raise WorkflowSpecError(f"{name} must contain positive integers")
    return tuple(value)


def _nonnegative_int_tuple(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value
    ):
        raise WorkflowSpecError(f"{name} must contain non-negative integers")
    return tuple(value)


def _bounds(value: object, rank: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != rank:
        raise WorkflowSpecError("selection bounds must contain one pair per axis")
    bounds: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(
                isinstance(part, int) and not isinstance(part, bool) for part in item
            )
            or item[0] < 0
            or item[1] <= item[0]
        ):
            raise WorkflowSpecError("selection bounds must be increasing integer pairs")
        bounds.append((item[0], item[1]))
    return tuple(bounds)


def _dtype(value: object) -> str:
    if not isinstance(value, str) or len(value) > 128:
        raise WorkflowSpecError("dtype must be a short string")
    try:
        dtype = np.dtype(value)
    except TypeError as exc:
        raise WorkflowSpecError(f"invalid dtype {value!r}") from exc
    if dtype.hasobject:
        raise WorkflowSpecError("object dtypes are not portable")
    return dtype.str


def _json_copy(value: object) -> object:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise WorkflowSpecError(f"workflow contains a non-JSON value: {exc}") from exc


def _without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value
