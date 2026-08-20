from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.baselines import direct_dask_mean_traces


class _ChunkedArray:
    def __init__(self, values: np.ndarray, chunks: tuple[int, ...]) -> None:
        self.values = values
        self.shape = values.shape
        self.ndim = values.ndim
        self.dtype = values.dtype
        self.chunks = chunks
        self.reads: list[object] = []

    def __getitem__(self, key: Any) -> np.ndarray:
        self.reads.append(key)
        return self.values[key]


def test_direct_dask_trace_baseline_is_numerically_equivalent() -> None:
    values = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    source = _ChunkedArray(values, (2, 2, 2))
    labels = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.uint64,
    )

    actual, cell_ids, plan = direct_dask_mean_traces(
        source,
        labels,
        time_chunk=2,
    )

    expected = np.column_stack(
        [values[:, labels == label_id].mean(axis=1) for label_id in (1, 2)]
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cell_ids, np.array([1, 2], dtype=np.uint64))
    assert plan.active_spatial_chunks == 2
    assert plan.skipped_spatial_chunks == 2
    assert plan.dask_compute_calls == 4
    assert plan.source_chunks_touched == 4
