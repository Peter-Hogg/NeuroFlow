from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.exceptions import AdapterCompatibilityError, ProvenanceMismatchError
from neuroflow.expression import expression_to_dict
from neuroflow.partition import TimeWindowPlan
from neuroflow.provenance import stable_hash
from neuroflow.selection import NWBQuery, Selection
from neuroflow.source import SourceIdentity
from neuroflow.source.array import ArraySource
from neuroflow.storage import ZarrOutput


def _typed_array(tmp_path: Path, values: np.ndarray) -> neuroflow.NeuroArray:
    path = tmp_path / f"{values.dtype}.zarr"
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset("values", data=values, chunks=(2, 3))
    source = ArraySource(path, component="values", axes=("row", "column"))
    return neuroflow.NeuroArray(source, source.select())


def test_expression_construction_repr_and_metadata_are_lazy(
    nwb_zarr: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    reads: list[object] = []
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        if array.path == "acquisition/movie/data":
            reads.append(key)
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    expression = np.sqrt((movie[:5] + 1) * 2)

    assert isinstance(expression, neuroflow.NeuroArray)
    assert expression.shape == (5, 3, 4)
    assert expression.axes == ("time", "y", "x")
    assert expression.dtype == np.dtype("float32")
    assert "lazy=True" in repr(expression)
    assert reads == []

    np.testing.assert_allclose(expression.compute(), np.sqrt((nwb_zarr[1][:5] + 1) * 2))
    assert reads
    movie.close()


def test_size_accepts_one_axis_and_rejects_reduction_axis_tuples(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")

    assert np.size(movie, axis=0) == movie.shape[0]
    assert np.size(movie, axis="x") == movie.shape[2]  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="one name or integer"):
        np.size(movie, axis=(0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="optional axis"):
        movie.__array_function__(np.size, (type(movie),), (movie,), {"bad": 1})
    movie.close()


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda value: value + 2, lambda value: value + 2),
        (lambda value: 2 + value, lambda value: 2 + value),
        (lambda value: value - 2, lambda value: value - 2),
        (lambda value: 2 - value, lambda value: 2 - value),
        (lambda value: value * 2, lambda value: value * 2),
        (lambda value: value / 2, lambda value: value / 2),
        (lambda value: value // 2, lambda value: value // 2),
        (lambda value: value**2, lambda value: value**2),
        (lambda value: value % 7, lambda value: value % 7),
        (lambda value: -value, lambda value: -value),
        (lambda value: abs(value), lambda value: abs(value)),
        (lambda value: value >= 30, lambda value: value >= 30),
        (np.log1p, np.log1p),
        (np.isfinite, np.isfinite),
    ],
)
def test_operators_and_ufuncs_match_numpy(
    nwb_zarr: tuple[Path, np.ndarray],
    operation: object,
    expected: object,
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    actual_array = operation(movie)  # type: ignore[operator]
    expected_array = expected(nwb_zarr[1])  # type: ignore[operator]

    assert isinstance(actual_array, neuroflow.NeuroArray)
    assert actual_array.dtype == expected_array.dtype
    if expected_array.dtype.kind in "bui":
        np.testing.assert_array_equal(actual_array.compute(), expected_array)
    else:
        np.testing.assert_allclose(actual_array.compute(), expected_array)
    movie.close()


@pytest.mark.parametrize(
    ("dtype", "scalar"),
    [
        ("float32", 1.0),
        ("float32", np.float64(1)),
        ("uint16", 1),
        ("uint16", np.int64(1)),
    ],
)
def test_numpy_two_scalar_promotion_is_preserved(
    tmp_path: Path, dtype: str, scalar: object
) -> None:
    values = np.arange(6, dtype=dtype).reshape(2, 3)
    array = _typed_array(tmp_path, values)
    expression = array + scalar

    assert expression.dtype == (values + scalar).dtype
    np.testing.assert_array_equal(expression.compute(), values + scalar)
    array.close()


@pytest.mark.parametrize(
    ("method", "axis", "keepdims", "dtype"),
    [
        ("sum", "time", False, None),
        ("sum", ("time", "x"), True, "float64"),
        ("mean", 0, False, None),
        ("min", -1, False, None),
        ("max", (0, 2), False, None),
        ("median", "time", True, None),
    ],
)
def test_reductions_match_numpy_with_named_and_positional_axes(
    nwb_zarr: tuple[Path, np.ndarray],
    method: str,
    axis: object,
    keepdims: bool,
    dtype: str | None,
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    named_axes = movie.axes
    normalized_values: list[int] = []
    for item in axis if isinstance(axis, tuple) else (axis,):
        if isinstance(item, str):
            normalized_values.append(named_axes.index(item))
        elif isinstance(item, int):
            normalized_values.append(item)
        else:  # pragma: no cover - parametrization contains only valid axes
            raise AssertionError("invalid test axis")
    normalized = tuple(normalized_values)
    numpy_axis: int | tuple[int, ...]
    numpy_axis = normalized[0] if len(normalized) == 1 else normalized
    kwargs: dict[str, object] = {"axis": axis, "keepdims": keepdims}
    numpy_kwargs: dict[str, object] = {
        "axis": numpy_axis,
        "keepdims": keepdims,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
        numpy_kwargs["dtype"] = dtype

    actual = getattr(movie, method)(**kwargs)
    expected = getattr(np, method)(nwb_zarr[1], **numpy_kwargs)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.compute(), expected)
    movie.close()


def test_numpy_function_dispatch_and_reduction_chaining(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")[:5]
    expression = (
        np.mean(
            movie + 1,
            axis="time",  # pyright: ignore[reportCallIssue, reportArgumentType]
        ).astype("float64")
        + np.max(movie, axis=0)
    ) / 2
    expected = (
        np.mean(nwb_zarr[1][:5] + 1, axis=0).astype("float64")
        + np.max(nwb_zarr[1][:5], axis=0)
    ) / 2

    assert np.shape(expression) == (3, 4)
    assert np.ndim(expression) == 2
    assert np.size(expression) == 12
    np.testing.assert_allclose(expression.compute(), expected)
    movie.close()


@pytest.mark.parametrize(("function", "q"), [(np.percentile, 75), (np.quantile, 0.75)])
def test_scalar_percentiles_match_numpy(
    nwb_zarr: tuple[Path, np.ndarray], function: object, q: float
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    actual = function(movie, q, axis="time")  # type: ignore[operator]
    expected = function(nwb_zarr[1], q, axis=0)  # type: ignore[operator]
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual.compute(), expected)
    movie.close()


def test_rank_preserving_slicing_and_absolute_bounds(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    selected = movie[1:8, :2, ...][:3]

    assert selected.shape == (3, 2, 4)
    assert selected.selection.metadata.selection_bounds == ((1, 4), (0, 2), (0, 4))
    np.testing.assert_array_equal(selected.compute(), nwb_zarr[1][1:4, :2])
    movie.close()


@pytest.mark.parametrize(
    "key",
    [0, (slice(None), 0), None, (Ellipsis, Ellipsis), slice(None, None, 2), [0, 1]],
)
def test_unsupported_indexing_fails_without_reads(
    nwb_zarr: tuple[Path, np.ndarray], key: object
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    with pytest.raises((IndexError, ValueError)):
        movie[key]
    movie.close()


def test_implicit_materialization_and_unsupported_numpy_are_rejected(
    nwb_zarr: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    reads = 0
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        nonlocal reads
        if array.path == "acquisition/movie/data":
            reads += 1
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    with pytest.raises(TypeError, match="lazy"):
        np.asarray(movie)
    with pytest.raises(TypeError, match="lazy"):
        np.array(movie)
    with pytest.raises(TypeError, match="truth value"):
        bool(movie)
    with pytest.raises(TypeError, match="iteration"):
        iter(movie)
    with pytest.raises(TypeError):
        np.concatenate([movie, movie])
    with pytest.raises(TypeError):
        np.add.reduce(movie)
    with pytest.raises(TypeError, match="NumPy array operands"):
        _ = movie + np.ones(movie.shape, dtype=np.float32)
    assert reads == 0
    movie.close()


def test_axis_and_array_operand_validation_precedes_reads(
    nwb_zarr: tuple[Path, np.ndarray],
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    with pytest.raises(ValueError, match="no axis"):
        movie.mean("channel")
    with pytest.raises(np.exceptions.AxisError):
        movie.mean(3)
    with pytest.raises(ValueError, match="duplicate"):
        movie.mean(("time", 0))
    with pytest.raises(TypeError, match="axis"):
        movie.mean(1.5)
    with pytest.raises(ValueError, match="same source selection"):
        _ = movie[:5] + movie[5:10]
    with pytest.raises(ValueError, match="identical axes and shapes"):
        _ = movie + movie.mean("time")
    movie.close()


def test_compute_memory_guard_fails_before_source_read(
    nwb_zarr: tuple[Path, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    reads = 0
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        nonlocal reads
        if array.path == "acquisition/movie/data":
            reads += 1
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    with pytest.raises(ValueError, match="total process-memory target"):
        (movie + 1).compute(memory_limit=100)
    assert reads == 0
    movie.close()


def test_fused_expression_persists_resumes_and_verifies(
    nwb_zarr: tuple[Path, np.ndarray],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")[:5]
    expression = np.sqrt(movie + 1).mean("time", keepdims=True)
    output = tmp_path / "expression.zarr"
    persisted = expression.persist(
        output,
        chunks=(1, 2, 2),
        max_workers=1,
        memory_limit="64 MiB",
    )

    expected = np.sqrt(nwb_zarr[1][:5] + 1).mean(axis=0, keepdims=True)
    assert persisted.shape == (1, 3, 4)
    assert persisted.axes == ("time", "y", "x")
    assert persisted.selection.metadata.native_chunks == (1, 2, 2)
    np.testing.assert_allclose(persisted.compute(), expected)
    assert persisted.workflow.verify().valid
    assert neuroflow.open_result(output).verify().valid

    source_reads = 0
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        nonlocal source_reads
        if array.path == "acquisition/movie/data":
            source_reads += 1
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    resumed = expression.persist(
        output,
        chunks=(1, 2, 2),
        max_workers=1,
        memory_limit="64 MiB",
    )
    assert source_reads == 0
    assert resumed.workflow.verify().valid
    resumed.close()
    persisted.close()
    movie.close()


def test_temporal_projection_tiles_retained_z_planes_without_xy_rereads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = np.arange(2 * 4 * 6 * 3, dtype=np.uint16).reshape(2, 4, 6, 3)
    source_path = tmp_path / "four-dimensional.zarr"
    group = zarr.open_group(str(source_path), mode="w")
    group.create_dataset("movie", data=values, chunks=(1, 4, 6, 1))
    source = ArraySource(
        source_path,
        component="movie",
        axes=("time", "y", "x", "z"),
    )
    movie = neuroflow.NeuroArray(source, source.select())
    reads: list[tuple[slice, ...]] = []
    original = zarr.Array.__getitem__

    def recorded(array: zarr.Array, key: object) -> object:
        if array.path == "movie" and isinstance(key, tuple):
            reads.append(key)  # type: ignore[arg-type]
        return original(array, key)

    monkeypatch.setattr(zarr.Array, "__getitem__", recorded)
    projection = np.median(
        movie,
        axis="time",  # pyright: ignore[reportCallIssue, reportArgumentType]
    ).persist(
        tmp_path / "projection.zarr",
        chunks=(2, 3, 1),
        max_workers=1,
        memory_limit="64 MiB",
    )

    assert projection.workflow.plan.task_count == 3
    assert projection.workflow.plan.processing_partition_shape == (2, 4, 6, 1)
    assert len(reads) == 3
    assert all(item[:3] == (slice(0, 2), slice(0, 4), slice(0, 6)) for item in reads)
    np.testing.assert_array_equal(projection.compute(), np.median(values, axis=0))
    projection.close()
    movie.close()


def test_scalar_reduction_persists_as_zero_dimensional_array(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")[:5]
    persisted = np.sum(movie).persist(
        tmp_path / "scalar.zarr", memory_limit="64 MiB", max_workers=1
    )

    assert persisted.shape == ()
    assert persisted.axes == ()
    assert persisted.compute().shape == ()
    assert persisted.compute().item() == np.sum(nwb_zarr[1][:5]).item()
    assert persisted.workflow.verify().valid
    persisted.close()
    movie.close()


def test_expression_identity_is_canonical_and_conflicts_are_detected(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")[:5]
    named = np.mean(
        movie + 1,
        axis="time",  # pyright: ignore[reportCallIssue, reportArgumentType]
    )
    positional = (movie + 1).mean(0)

    assert stable_hash(expression_to_dict(named.expression)) == stable_hash(
        expression_to_dict(positional.expression)
    )
    first = named.persist(tmp_path / "named.zarr", memory_limit="64 MiB")
    second = positional.persist(tmp_path / "positional.zarr", memory_limit="64 MiB")
    assert first.workflow.plan.workflow_id == second.workflow.plan.workflow_id

    with pytest.raises(ProvenanceMismatchError):
        (movie + 2).mean("time").persist(tmp_path / "named.zarr", memory_limit="64 MiB")
    first.close()
    second.close()
    movie.close()


def test_full_slice_has_canonical_bounds_and_workflow_identity(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")
    unsliced = movie.mean("time")
    explicitly_full = movie[:].mean("time")

    first = unsliced.persist(tmp_path / "unsliced.zarr", memory_limit="64 MiB")
    second = explicitly_full.persist(
        tmp_path / "explicitly-full.zarr", memory_limit="64 MiB"
    )

    first_workflow = first.workflow
    second_workflow = second.workflow
    assert first_workflow is not None
    assert second_workflow is not None
    assert first_workflow.plan.workflow_id == second_workflow.plan.workflow_id
    expected_bounds = [[0, size] for size in movie.shape]
    assert first_workflow.provenance is not None
    assert second_workflow.provenance is not None
    assert first_workflow.provenance["selection"]["bounds"] == expected_bounds  # type: ignore[index]
    assert second_workflow.provenance["selection"]["bounds"] == expected_bounds  # type: ignore[index]
    first.close()
    second.close()
    movie.close()


def test_selected_asset_identity_participates_in_workflow_hash(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    selection = source.select(NWBQuery(name="movie"))
    first = Selection(
        replace(
            selection.metadata,
            source=SourceIdentity("DANDI:000001", "1", asset_id="asset-a"),
        ),
        selection._array,
    )
    second = Selection(
        replace(
            selection.metadata,
            source=SourceIdentity("DANDI:000001", "1", asset_id="asset-b"),
        ),
        selection._array,
    )
    adapter = FunctionAdapter(
        function=lambda value: value,
        input_kind="array",
        output=ArrayOutput("float32"),
        name="identity",
        version="1",
        splittable_axes=("time",),
    )
    first_plan = neuroflow.plan(
        source=source,
        selection=first,
        adapter=adapter,
        partition=TimeWindowPlan(5),
        output=ZarrOutput(str(tmp_path / "first.zarr")),
    )
    second_plan = neuroflow.plan(
        source=source,
        selection=second,
        adapter=adapter,
        partition=TimeWindowPlan(5),
        output=ZarrOutput(str(tmp_path / "second.zarr")),
    )

    assert first_plan.workflow_id != second_plan.workflow_id
    source.close()


def test_parallel_array_writes_require_chunk_aligned_boundaries(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    source = neuroflow.open_source(nwb_zarr[0])
    selection = source.select(NWBQuery(name="movie"))
    adapter = FunctionAdapter(
        function=lambda value: value,
        input_kind="array",
        output=ArrayOutput("float32", chunks=(5, 3, 4)),
        name="unsafe-chunks",
        version="1",
        splittable_axes=("time",),
    )

    with pytest.raises(AdapterCompatibilityError, match="crosses a processing"):
        neuroflow.plan(
            source=source,
            selection=selection,
            adapter=adapter,
            partition=TimeWindowPlan(4),
            output=ZarrOutput(str(tmp_path / "unsafe.zarr")),
        )
    source.close()


def test_chained_result_identity_includes_upstream_workflow(
    nwb_zarr: tuple[Path, np.ndarray], tmp_path: Path
) -> None:
    movie = neuroflow.load(nwb_zarr[0], name="movie")[:5]
    upstream_path = tmp_path / "upstream.zarr"
    first_upstream = (movie + 1).persist(upstream_path, memory_limit="64 MiB")
    first_downstream = (first_upstream * 2).persist(
        tmp_path / "first-downstream.zarr", memory_limit="64 MiB"
    )
    first_id = first_downstream.workflow.plan.workflow_id
    first_downstream.close()
    first_upstream.close()

    second_upstream = (movie + 2).persist(
        upstream_path, mode="overwrite", memory_limit="64 MiB"
    )
    second_downstream = (second_upstream * 2).persist(
        tmp_path / "second-downstream.zarr", memory_limit="64 MiB"
    )

    assert second_downstream.workflow.plan.workflow_id != first_id
    second_downstream.close()
    second_upstream.close()
    movie.close()
