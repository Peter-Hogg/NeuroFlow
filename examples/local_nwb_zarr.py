"""End-to-end local NWB-Zarr workflow with no network access.

Run from the repository root with::

    uv run python -m examples.local_nwb_zarr

Generated data and results are written under ``examples/_output`` (gitignored).
Running the command again exercises NeuroFlow's partition-level resume path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from hdmf_zarr import NWBZarrIO, ZarrDataIO
from pynwb import NWBFile, TimeSeries

import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ZarrOutput


def scale_block(block: np.ndarray, factor: float) -> np.ndarray:
    """Example user analysis: a pure function over one bounded NumPy block."""
    return np.asarray(block, dtype=np.float32) * np.float32(factor)


def create_demo_source(path: Path) -> np.ndarray:
    """Create a small, natively chunked NWB-Zarr source if it is absent."""
    expected = np.arange(12 * 3 * 4, dtype=np.float32).reshape(12, 3, 4)
    if path.exists():
        return expected

    nwbfile = NWBFile(
        session_description="NeuroFlow local example",
        identifier="neuroflow-local-example",
        session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        session_id="local-example",
    )
    nwbfile.add_acquisition(
        TimeSeries(
            name="movie",
            data=ZarrDataIO(expected, chunks=(3, 3, 4)),
            unit="a.u.",
            rate=2.0,
        )
    )
    with NWBZarrIO(path, mode="w") as io:
        io.write(nwbfile)
    return expected


def run_example(workdir: Path) -> dict[str, object]:
    """Run, verify, resume, and reopen the demo workflow."""
    workdir.mkdir(parents=True, exist_ok=True)
    source_path = workdir / "session.nwb.zarr"
    output_path = workdir / "scaled.zarr"
    expected = create_demo_source(source_path)

    with neuroflow.open_source(source_path) as source:
        movie = source.select(NWBQuery(name="movie"))
        lazy_input = movie.as_dask_array(chunks="native")
        adapter = FunctionAdapter(
            function=scale_block,
            input_kind="array",
            output=ArrayOutput("float32", name="scaled_movie"),
            name="scale-block",
            version="1",
            splittable_axes=("time",),
            parameters={"factor": 2.5},
        )
        result = neuroflow.run(
            source=source,
            selection=movie,
            adapter=adapter,
            partition=TimeWindowPlan(size=3),
            output=ZarrOutput(str(output_path)),
        )
        result.execute()
        first_verification = result.verify()

        # A second execution validates completed partitions and skips valid work.
        result.resume()
        resumed_verification = result.verify()
        task_count = result.plan.task_count

    reopened = neuroflow.open_result(output_path)
    lazy_output = reopened.arrays["scaled_movie"].as_dask_array()
    actual = lazy_output.compute()
    np.testing.assert_array_equal(actual, expected * np.float32(2.5))

    return {
        "input_shape": lazy_input.shape,
        "input_chunks": lazy_input.chunks,
        "task_count": task_count,
        "status": reopened.status.state,
        "verified_before_resume": first_verification.valid,
        "verified_after_resume": resumed_verification.valid,
        "output_shape": actual.shape,
        "output_uri": str(output_path),
    }


def main() -> None:
    summary = run_example(Path(__file__).parent / "_output")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
