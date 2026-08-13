# Local NWB-Zarr example

Run the complete example from the repository root:

```bash
uv run python -m examples.local_nwb_zarr
```

The script creates a tiny, chunked NWB-Zarr source locally, selects its movie
without reading the numerical chunks, and builds four Dask tasks. Each task reads
one bounded time block and passes a NumPy array to the user-defined
`scale_block` function. NeuroFlow persists the blocks to Zarr with provenance
and completion manifests, verifies their checksums, exercises resume, and
reopens the output as a lazy Dask array.

All generated files live in `examples/_output/`, which is ignored by Git. The
example uses only NeuroFlow's core dependencies and never accesses the network.
