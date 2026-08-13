"""Bounded fluorescence trace extraction from dense label images."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import zarr

from neuroflow.array import NeuroArray


def extract_traces(
    movie: NeuroArray,
    labels: NeuroArray,
    *,
    output: str | Path,
    time_chunk: int = 10,
) -> NeuroArray:
    """Average movie voxels per nonzero label using bounded time/z blocks."""
    if "time" not in movie.axes:
        raise ValueError("movie requires a time axis")
    spatial_axes = tuple(axis for axis in movie.axes if axis != "time")
    if labels.axes != spatial_axes:
        raise ValueError("label axes must equal the movie axes excluding time")
    expected = tuple(movie.shape[movie.axes.index(axis)] for axis in spatial_axes)
    if labels.shape != expected:
        raise ValueError("label and movie spatial shapes differ")
    if time_chunk < 1:
        raise ValueError("time_chunk must be positive")

    label_data = labels.selection.as_dask_array()
    ids = np.asarray(da.unique(label_data[label_data != 0]).compute(), dtype=np.uint64)
    ids.sort()
    if not len(ids):
        raise ValueError("labels contain no cells")
    root = zarr.open_group(str(output), mode="a")
    trace_shape = (movie.shape[movie.axes.index("time")], len(ids))
    trace_chunks = (min(time_chunk, trace_shape[0]), min(1024, len(ids)))
    if "cell_ids" in root:
        if not np.array_equal(np.asarray(root["cell_ids"]), ids):
            raise ValueError("existing trace output has different cell IDs")
    else:
        root.create_dataset("cell_ids", data=ids)
    if "traces" in root:
        traces = root["traces"]
        if not isinstance(traces, zarr.Array) or tuple(traces.shape) != trace_shape:
            raise ValueError("existing trace output has an incompatible shape")
    else:
        traces = root.create_dataset(
            "traces",
            shape=trace_shape,
            chunks=trace_chunks,
            dtype="float32",
            fill_value=np.nan,
        )
    completed = {int(value) for value in root.attrs.get("completed_z", [])}
    root.attrs.update({"axes": ["time", "cell"], "status": "running"})
    movie_data = movie.selection.as_dask_array()
    z_axis = spatial_axes.index("z") if "z" in spatial_axes else None
    z_values = range(labels.shape[z_axis]) if z_axis is not None else (None,)
    id_to_column = {int(value): index for index, value in enumerate(ids)}
    time_size = traces.shape[0]
    for z_value in z_values:
        z_key = -1 if z_value is None else z_value
        if z_key in completed:
            continue
        spatial_key = [slice(None)] * labels.selection._array.ndim
        if z_axis is not None:
            spatial_key[z_axis] = slice(z_value, z_value + 1)  # type: ignore[operator]
        plane_labels = np.asarray(label_data[tuple(spatial_key)].compute())
        plane_ids = np.unique(plane_labels)
        plane_ids = plane_ids[plane_ids != 0]
        columns = np.asarray([id_to_column[int(value)] for value in plane_ids])
        flat_labels = plane_labels.reshape(-1)
        for start in range(0, time_size, time_chunk):
            stop = min(time_size, start + time_chunk)
            movie_key = [slice(None)] * movie.selection._array.ndim
            movie_key[movie.axes.index("time")] = slice(start, stop)
            if z_axis is not None:
                movie_key[movie.axes.index("z")] = slice(z_value, z_value + 1)  # type: ignore[operator]
            block = np.asarray(movie_data[tuple(movie_key)].compute(), dtype=np.float32)
            block = np.moveaxis(block, movie.axes.index("time"), 0).reshape(
                stop - start, -1
            )
            values = np.empty((stop - start, len(plane_ids)), dtype=np.float32)
            for index, label_id in enumerate(plane_ids):
                values[:, index] = block[:, flat_labels == label_id].mean(axis=1)
            traces[start:stop, columns] = values
        completed.add(z_key)
        root.attrs["completed_z"] = sorted(completed)
    root.attrs["status"] = "complete"
    from neuroflow.source.array import ArraySource

    source = ArraySource(output, component="traces", axes=("time", "cell"))
    return NeuroArray(source, source.select())
