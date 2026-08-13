from pathlib import Path

from examples.local_nwb_zarr import run_example


def test_local_nwb_zarr_example_is_rerunnable(tmp_path: Path) -> None:
    first = run_example(tmp_path)
    second = run_example(tmp_path)

    for summary in (first, second):
        assert summary["input_shape"] == (12, 3, 4)
        assert summary["input_chunks"] == ((3, 3, 3, 3), (3,), (4,))
        assert summary["task_count"] == 4
        assert summary["status"] == "complete"
        assert summary["verified_before_resume"] is True
        assert summary["verified_after_resume"] is True
        assert summary["output_shape"] == (12, 3, 4)
