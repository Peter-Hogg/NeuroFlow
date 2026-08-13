# High-level named-array API

`NeuroArray` is the stable user-facing API for version 0.1. Users select data by
named axes and write ordinary NumPy-like operations; NeuroFlow chooses bounded
Dask tasks and durable output layouts.

```python
movie = neuroflow.load(source, name="NeuronOnePhotonSeries")
projection = movie.isel(time=slice(0, 50)).median(
    "time",
    output="projection.zarr",
    chunks=(256, 256, 1),
    max_workers=2,
    memory_limit="2 GiB",
)
```

Persisted arrays can re-enter a workflow with `neuroflow.open_array()`. Use
`projection.segment(...)` for segmentation and
`movie.extract_traces(labels, ...)` for mean fluorescence traces.

## Guarantees

- Named-axis slices are lazy and contiguous.
- Reductions declare removed axes and output chunks.
- Durable outputs have deterministic workflow identity, provenance, partition
  manifests, checksums, resume, and verification.
- `max_workers` and `memory_limit` bound aggregate scheduled task estimates.
- Remote NWB-HDF5 data remain range-streamed with a bounded memory cache.

## Limitations in 0.1

- NumPy dispatch through `__array_function__` is not implemented; use named
  methods such as `.median("time")`.
- Memory limits cover declared adapter memory and source partition estimates,
  not memory hidden inside arbitrary third-party native code.
- Friendly segmentation rejects y/x partitioning that would require object
  reconciliation. Use complete planes with Cellpose internal tiling, or opt into
  explicitly `unmerged` low-level results.
- Remote HDF5 supports the threaded scheduler only.
- Trace extraction supports dense labels with axes matching the movie's
  non-time axes and computes mean fluorescence per nonzero label.
