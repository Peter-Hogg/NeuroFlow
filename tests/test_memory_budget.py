"""Regression tests for total-process memory-budget semantics.

``memory_limit`` is an approximate *total process* target, not a per-task
allowance. These tests pin the properties that make that claim defensible:
the budget decomposes exactly, it is monotone, it declares when a target is
physically unattainable, it accounts for third-party model residency, and the
resulting plan reports planned and measured numbers side by side.
"""

from pathlib import Path

import numpy as np
import pytest
import zarr

import neuroflow
from neuroflow.execution.resources import (
    DEFAULT_PROCESS_OVERHEAD_BYTES,
    PROCESS_OVERHEAD_COMPONENTS,
    resolve_memory_budget,
)
from neuroflow.source.array import ArraySource


def _array(
    path: Path,
    name: str,
    data: np.ndarray,
    axes: tuple[str, ...],
    *,
    chunks: tuple[int, ...] | None = None,
) -> neuroflow.NeuroArray:
    group = zarr.open_group(str(path), mode="w")
    group.create_dataset(name, data=data, chunks=chunks)
    source = ArraySource(path, component=name, axes=axes)
    return neuroflow.NeuroArray(source, source.select())


def _fish_like_movie(
    tmp_path: Path, *, frames: int, height: int, width: int, planes: int
) -> tuple[neuroflow.NeuroArray, neuroflow.NeuroArray]:
    """A small movie chunked one (y, x) plane per source chunk, like the fish."""
    rng = np.random.default_rng(0)
    movie_values = rng.integers(
        0, 4096, size=(frames, height, width, planes), dtype=np.int16
    )
    movie = _array(
        tmp_path / "movie.zarr",
        "movie",
        movie_values,
        ("time", "y", "x", "z"),
        chunks=(1, height, width, 1),
    )
    label_values = np.zeros((height, width, planes), dtype=np.uint64)
    label_values[0:2, 0:2, :] = 1
    label_values[2:4, 2:4, :] = 2
    labels = _array(
        tmp_path / "labels.zarr",
        "labels",
        label_values,
        ("y", "x", "z"),
        chunks=(height, width, 1),
    )
    return movie, labels


def test_budget_decomposes_exactly_into_overhead_and_task_bytes() -> None:
    budget = resolve_memory_budget("2 GiB")

    assert budget.total_bytes == 2 * 1024**3
    assert budget.process_overhead_bytes == DEFAULT_PROCESS_OVERHEAD_BYTES
    assert budget.reserved_bytes + budget.task_bytes == budget.total_bytes
    assert budget.task_bytes < budget.total_bytes


def test_process_overhead_floor_is_the_sum_of_its_named_components() -> None:
    assert DEFAULT_PROCESS_OVERHEAD_BYTES == sum(PROCESS_OVERHEAD_COMPONENTS.values())
    # Every component must be attributed to a measured cause, not a round number
    # invented at the top level.
    assert set(PROCESS_OVERHEAD_COMPONENTS) == {
        "interpreter_and_libraries",
        "dask_runtime",
        "source_read_cache",
        "output_write_buffers",
    }


def test_task_bytes_never_decrease_when_the_target_rises() -> None:
    """Raising a limit must never shrink the window a planner may choose."""
    previous = 0
    for exponent in range(1, 36):
        budget = resolve_memory_budget(2**exponent)
        assert budget.task_bytes >= previous, f"regression at 2**{exponent}"
        previous = budget.task_bytes


def test_targets_below_the_measured_floor_are_reported_unattainable() -> None:
    small = resolve_memory_budget(64 * 1024 * 1024)
    large = resolve_memory_budget(2 * 1024**3)

    assert small.attainable is False
    assert "cannot be met" in str(small.to_dict()["note"])
    assert large.attainable is True
    # A small target is still useful: it bounds the task working set.
    assert small.task_bytes > 0


def test_budget_report_names_its_interpretation_and_lack_of_enforcement() -> None:
    report = resolve_memory_budget("4 GiB").to_dict()

    assert report["interpretation"] == "approximate-total-process-target"
    # The absence of an OS-level cap must be stated, because a reader could
    # otherwise assume the process is hard-limited.
    assert "no OS-level memory cap" in str(report["enforcement"])


def test_declared_model_residency_is_subtracted_from_the_target() -> None:
    plain = resolve_memory_budget("4 GiB")
    with_model = resolve_memory_budget("4 GiB", external_reserve_bytes=1024**3)

    assert with_model.task_bytes == plain.task_bytes - 1024**3
    assert with_model.external_reserve_bytes == 1024**3


def test_a_target_fully_consumed_by_reserves_is_refused_with_guidance() -> None:
    with pytest.raises(ValueError) as error:
        resolve_memory_budget("1 GiB", external_reserve_bytes=4 * 1024**3)

    message = str(error.value)
    assert "total process-memory target" in message
    assert "Raise memory_limit above" in message


def test_raising_the_target_widens_the_trace_window_and_cuts_task_count(
    tmp_path: Path,
) -> None:
    # Planes large enough that the budget, not the movie length, sets the window.
    movie, labels = _fish_like_movie(
        tmp_path, frames=64, height=256, width=256, planes=1
    )

    narrow = movie.plan_traces(labels, memory_limit=8 * 1024 * 1024)
    wide = movie.plan_traces(labels, memory_limit=64 * 1024 * 1024)

    assert narrow.time_chunk < wide.time_chunk
    assert narrow.task_count > wide.task_count
    labels.close()
    movie.close()


def test_trace_plan_reports_planned_task_memory_and_planned_process_peak(
    tmp_path: Path,
) -> None:
    movie, labels = _fish_like_movie(
        tmp_path, frames=32, height=32, width=32, planes=2
    )
    limit = 64 * 1024 * 1024

    plan = movie.plan_traces(labels, memory_limit=limit)
    resources = plan.to_dict()["resources"]
    assert isinstance(resources, dict)

    task_estimate = resources["estimated_memory_per_task"]
    assert isinstance(task_estimate, dict)
    peak_estimate = resources["estimated_process_peak_bytes"]
    assert isinstance(peak_estimate, dict)

    # Both numbers are present, and the process peak is strictly the larger:
    # a reader must never mistake a task working set for a process total.
    assert task_estimate["value"] == plan.estimated_memory_per_task
    assert peak_estimate["value"] > task_estimate["value"]
    assert (
        peak_estimate["value"]
        == plan.memory_budget.reserved_bytes + plan.estimated_memory_per_task
    )
    # The planned process peak must respect the target the user asked for.
    assert peak_estimate["value"] <= limit
    # Measured RSS is unknowable at plan time and must say so rather than
    # reporting a fabricated number.
    measured = resources["measured_process_peak_rss_bytes"]
    assert isinstance(measured, dict)
    assert measured["status"] == "unknown"
    assert measured["value"] is None
    labels.close()
    movie.close()


def test_task_estimate_excludes_the_process_overhead_it_used_to_double_count(
    tmp_path: Path,
) -> None:
    """The per-task estimate must not re-add a scheduler/cache reserve.

    An earlier revision folded a flat 128 MiB into every per-task estimate.
    That overhead is now charged once against the process target, so a tiny
    single-frame window must estimate far below it.
    """
    movie, labels = _fish_like_movie(tmp_path, frames=4, height=16, width=16, planes=1)

    plan = movie.plan_traces(labels, time_chunk=1, memory_limit=64 * 1024 * 1024)

    assert plan.estimated_memory_per_task < 8 * 1024 * 1024
    labels.close()
    movie.close()


def test_extraction_records_planned_against_measured_process_peak(
    tmp_path: Path,
) -> None:
    movie, labels = _fish_like_movie(tmp_path, frames=8, height=16, width=16, planes=1)

    traces = movie.extract_traces(
        labels,
        output=tmp_path / "traces.zarr",
        memory_limit=64 * 1024 * 1024,
    )
    provenance = neuroflow.open_result(tmp_path / "traces.zarr").provenance
    metrics = provenance["execution_metrics"]
    assert isinstance(metrics, dict)
    memory = metrics["memory"]
    assert isinstance(memory, dict)

    assert memory["planned_task_working_bytes"] > 0
    assert (
        memory["planned_process_peak_bytes"] > memory["planned_task_working_bytes"]
    )
    # The measured figure is a real observation, so it must be a positive count
    # and be labelled as a whole-process high-water mark.
    assert isinstance(memory["measured_process_peak_rss_bytes"], int)
    assert memory["measured_process_peak_rss_bytes"] > 0
    assert "whole process" in str(memory["measurement_scope"])
    assert memory["budget"]["interpretation"] == "approximate-total-process-target"
    traces.close()
    labels.close()
    movie.close()


def test_cellpose_declares_measured_model_residency_only_for_known_models() -> None:
    from neuroflow_cellpose import CellposeAdapter

    cpu = CellposeAdapter(pretrained_model="cpsam", gpu=False)
    gpu = CellposeAdapter(pretrained_model="cpsam", gpu=True)
    unknown = CellposeAdapter(pretrained_model="some-custom-net", gpu=False)

    # cpsam weights are ~1.2 GB; the host reserve must be at least that.
    assert cpu.external_memory_reserve_bytes() >= 1_218_515_720
    # Running on GPU moves the weights to VRAM, so the host reserve is smaller.
    assert gpu.external_memory_reserve_bytes() < cpu.external_memory_reserve_bytes()
    # An unmeasured model reports zero rather than guessing.
    assert unknown.external_memory_reserve_bytes() == 0


def test_cellpose_per_task_estimate_scales_with_tile_area() -> None:
    from neuroflow_cellpose import CellposeAdapter

    adapter = CellposeAdapter(pretrained_model="cpsam")

    small = adapter.estimate_task_memory((256, 256))
    large = adapter.estimate_task_memory((512, 512))

    # Four times the pixels must cost meaningfully more, and the estimate must
    # not be a constant that ignores geometry.
    assert large > small
    assert large - small >= 3 * 256 * 256 * 4


def test_cellpose_convenience_does_not_echo_the_users_limit_as_its_own_cost() -> None:
    """The old behaviour made the budget check vacuous.

    ``.cellpose()`` used to declare the adapter's per-task memory as whatever
    ``memory_limit`` the caller passed, so every limit trivially satisfied
    itself. The adapter must now state a cost derived from geometry instead.
    """
    from neuroflow_cellpose import CellposeAdapter

    adapter = CellposeAdapter(pretrained_model="cpsam")

    assert adapter.memory is None
    assert adapter.requirements().resources.memory is None


def test_cpsam_cpu_cannot_fit_a_two_gibibyte_total_target() -> None:
    """A laptop-scale target must be refused rather than silently accepted.

    One loaded ``cpsam`` network needs roughly 1.9 GiB resident on CPU, so a
    2 GiB *total* process target is not achievable. The previous semantics
    accepted it because the reserve was never counted.
    """
    from neuroflow_cellpose import CellposeAdapter

    adapter = CellposeAdapter(pretrained_model="cpsam", gpu=False)

    with pytest.raises(ValueError, match="Raise memory_limit above"):
        resolve_memory_budget(
            "2 GiB",
            external_reserve_bytes=adapter.external_memory_reserve_bytes(),
        )

    # The same target is feasible once the weights move to VRAM.
    on_gpu = CellposeAdapter(pretrained_model="cpsam", gpu=True)
    budget = resolve_memory_budget(
        "2 GiB", external_reserve_bytes=on_gpu.external_memory_reserve_bytes()
    )
    assert budget.task_bytes > 0


def test_stated_worker_availability_is_clamped_not_refused(tmp_path: Path) -> None:
    """Declaring more workers than the target affords must not fail the run.

    The user-facing contract is that a caller states the resources they *have*
    and the planner chooses concurrency to fit. An earlier version raised
    ``max_workers=N exceeds the memory-safe limit of M``, which forced the
    caller to hand-tune a low-level knob to rediscover M -- a number the
    planner had already computed. Concurrency is clamped instead, and the
    granted count is recorded in provenance so the reduction stays auditable.
    """
    granted_by_limit: dict[str, int] = {}
    for limit in ("8 MiB", "16 MiB", "64 MiB"):
        movie, _ = _fish_like_movie(
            tmp_path / limit.replace(" ", ""),
            frames=8,
            height=256,
            width=256,
            planes=4,
        )
        output = tmp_path / f"projection-{limit.replace(' ', '')}.zarr"
        # Sixty-four workers are declared as *available* every time. Only the
        # target changes, so any difference in the granted count is attributable
        # to the memory budget rather than to the request or the core count.
        projection = np.median(movie, axis="time").astype(  # type: ignore[call-overload]
            np.float32
        )
        result = projection.persist(
            output,
            chunks=(256, 256, 1),
            max_workers=64,
            memory_limit=limit,
        )
        policy = neuroflow.open_result(output).provenance["execution_policy"]
        assert isinstance(policy, dict)
        granted = policy["max_workers"]
        assert isinstance(granted, int)
        granted_by_limit[limit] = granted
        result.close()
        movie.close()

    # Nothing was refused, and concurrency rose with the target instead of the
    # caller having to discover the safe number themselves.
    assert granted_by_limit["8 MiB"] == 1
    assert (
        granted_by_limit["8 MiB"]
        < granted_by_limit["16 MiB"]
        < granted_by_limit["64 MiB"]
    )
    # Memory, not the core count, is what bound these runs.
    assert granted_by_limit["64 MiB"] < 64


def test_compute_bounds_array_data_by_task_bytes_not_the_headline_total(
    tmp_path: Path,
) -> None:
    """``compute()`` must read ``memory_limit`` the same way ``persist()`` does.

    ``compute()`` used to compare the expression estimate against the raw
    parsed limit, so the same keyword meant "total process target" on
    ``persist()`` and "allowance for array data alone" here. A 1 GiB request
    could then peak well above 1 GiB of resident set.
    """
    from neuroflow.expression import estimate_working_memory

    movie, _ = _fish_like_movie(tmp_path, frames=4, height=64, width=64, planes=1)
    expression = np.median(movie, axis="time").astype(  # type: ignore[call-overload]
        np.float32
    )
    estimate = estimate_working_memory(expression.expression)

    # Choose a target the estimate fits inside but its task share does not.
    # Below the taper, overhead is half the target, so any total in
    # [estimate, 2 * estimate) is accepted under the old per-task reading and
    # refused under the corrected total-process reading.
    limit = int(estimate * 1.4)
    assert resolve_memory_budget(limit).task_bytes < estimate <= limit

    with pytest.raises(ValueError, match="total process-memory target"):
        expression.compute(memory_limit=limit)

    # Raising the target past the overhead makes the same expression fit.
    assert isinstance(expression.compute(memory_limit=estimate * 4), np.ndarray)
    movie.close()


def test_declared_overhead_envelope_still_covers_measured_attribution() -> None:
    """The declared floor must not drift below what was actually measured.

    ``PROCESS_OVERHEAD_COMPONENTS`` is documented as a rounded-up envelope over
    ``benchmarks/memory_attribution.py``. That claim is only meaningful if the
    recorded measurements are checked against it, so lowering a component to
    make a budget look better fails here instead of quietly overrunning.
    """
    import json

    # Anchored to the repository rather than the working directory, so the
    # check does not silently skip when pytest runs from elsewhere.
    record = (
        Path(__file__).resolve().parent.parent
        / "benchmarks/results/current-memory-attribution.json"
    )
    if not record.is_file():  # pragma: no cover - record is committed
        pytest.skip("no attribution record retained")
    components = {
        entry["component"]: entry
        for entry in json.loads(record.read_text())["components"]
    }

    # Each declared component must cover the measurement it claims to envelope.
    measured_library_floor = (
        components["process_baseline"]["rss_delta_bytes"]
        + components["neuroflow_import"]["rss_delta_bytes"]
    )
    assert (
        PROCESS_OVERHEAD_COMPONENTS["interpreter_and_libraries"]
        >= measured_library_floor
    )
    assert (
        PROCESS_OVERHEAD_COMPONENTS["dask_runtime"]
        >= components["dask_runtime"]["rss_delta_bytes"]
    )
    assert (
        PROCESS_OVERHEAD_COMPONENTS["output_write_buffers"]
        >= components["output_buffers"]["rss_delta_bytes"]
    )
    if "remfile_cache" in components:
        assert (
            PROCESS_OVERHEAD_COMPONENTS["source_read_cache"]
            >= components["remfile_cache"]["rss_delta_bytes"]
        )
    assert DEFAULT_PROCESS_OVERHEAD_BYTES == sum(PROCESS_OVERHEAD_COMPONENTS.values())


def test_default_segmentation_limit_admits_the_default_model() -> None:
    """The out-of-the-box ``cellpose()`` call must not refuse itself.

    ``memory_limit`` is a total process target and one loaded ``cpsam`` network
    measures ~1.9 GiB resident on CPU, so the 2 GiB persist default was
    consumed entirely by model weights and the default segmentation call raised
    before reading a pixel. Segmentation therefore carries its own default.
    """
    from neuroflow.array import (
        DEFAULT_PERSIST_MEMORY_LIMIT,
        DEFAULT_SEGMENT_MEMORY_LIMIT,
    )
    from neuroflow_cellpose import CellposeAdapter

    adapter = CellposeAdapter(pretrained_model="cpsam")
    reserve = adapter.external_memory_reserve_bytes()

    # The persist default genuinely cannot host the default model on CPU; that
    # is the condition this default exists to avoid.
    with pytest.raises(ValueError):
        resolve_memory_budget(
            DEFAULT_PERSIST_MEMORY_LIMIT, external_reserve_bytes=reserve
        )

    budget = resolve_memory_budget(
        DEFAULT_SEGMENT_MEMORY_LIMIT, external_reserve_bytes=reserve
    )
    # One real fish plane must fit in what is left after the model reserve.
    assert adapter.estimate_task_memory((888, 2048)) <= budget.task_bytes
    assert budget.attainable
