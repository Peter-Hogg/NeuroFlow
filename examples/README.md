# Runnable examples

## Local NWB-Zarr

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

## Remote NWB-HDF5 from DANDI

```bash
uv run python -m examples.dandi_hdf5
```

This example requires public internet access. It pins DANDI:000049 version
`0.230223.1424` and asset
`sub-760940732/sub-760940732_ses-798500537_behavior+ophys.nwb` (27.8 MB). The
asset was chosen because it is the smallest recording in that version. The
example selects only `max_project`, a `1 x 512 x 512` float32 dataset. It first
asks Dask for one logical `128 x 128` block and saves that block as
`examples/_output/dandi-hdf5-preview.png`. It then runs one bounded task
(~1 MiB of numerical input) before persisting and verifying a local Zarr result.

The HDF5 file is opened with the PyNWB-recommended `remfile` HTTP range reader,
using 256 KiB minimum requests and a bounded 64 MiB in-memory cache. NeuroFlow
does not download it to local disk or convert the whole file to an in-memory
array. HDF5 metadata discovery can still make many small remote requests, and
this asset's datasets are contiguous rather than physically chunked.
`--block-size` tunes the minimum request and `--cache-size-mib` bounds the cache;
neither makes HDF5 datasets physically chunked or increases the selected work.
`--preview-size` changes the logical Dask block used for the PNG and is capped at
512 pixels. Run `uv run python -m examples.dandi_hdf5 --help` for all options.

## Whole-brain zebrafish median projection

```bash
uv run python -m examples.dandi_fish_projection
```

This example uses Misha Ahrens' DANDI:000350 version `0.240822.1759`. It pins
the smallest recording in that Dandiset, a 150 GB NWB-HDF5 asset, and selects
the real `NeuronOnePhotonSeries` calcium movie with shape
`3065 x 888 x 2048 x 29` (`time, y, x, z`).

The default Dask graph selects the first three frames, upper-left `64 x 64`
pixels, and z-plane 0, then computes a median over time and saves
`examples/_output/fish-median-projection.png`. The source's physical chunks are
`1 x 888 x 2048 x 1`, so this touches exactly three native chunks—about 10.4 MiB
uncompressed in total before gzip—not three tiny 64-pixel tiles. This is still
bounded and tiny relative to the recording, but the distinction matters.

The example uses PyNWB's recommended `remfile` transport with a bounded 64 MiB
memory cache. `--frames` is capped at 9, `--crop-size` at 128, and `--z-plane`
at 28 so a casual run cannot silently become a large archive workload.
