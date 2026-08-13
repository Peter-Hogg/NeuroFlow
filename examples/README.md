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

The movie axes are `(time, y, x, z)`. The example bounds the selection to the
first 50 time frames and passes each z-plane to a normal NumPy function that
returns `np.median(tile, axis=0)`. NeuroFlow declares `time` as the reduced axis,
plans 29 resumable tasks, and assembles the resulting `(y, x, z)` array in
`examples/_output/fish-projection.zarr`. The output is chunked into
`256 x 256 x 1` y/x/z tiles. Z-plane 14 is also saved as
`examples/_output/fish-projection-z14.png` for a quick visual check.

The source's physical HDF5 chunks are `1 x 888 x 2048 x 1`. Subdividing source
reads over y/x would repeatedly fetch and decode the same complete image-plane
chunks, so NeuroFlow partitions computation over z and tiles y/x in the output
store instead. The default touches exactly 1,450 native chunks—about 5 GiB
uncompressed before gzip—without downloading the 150 GB file.

The example uses PyNWB's recommended `remfile` transport with a bounded 64 MiB
memory cache. `--frames` is capped at 50. `--tile-y` and `--tile-x` configure
bounded output chunks, while `--block-size` and `--cache-size-mib` retain safe
transport limits. Run `uv run python -m examples.dandi_fish_projection --help`
for all options.

### NumPy reference volume and napari

To save a compact reference volume for all 29 z-planes and open it in napari:

```bash
uv run python -m examples.dandi_fish_reference_volume
```

The default takes the temporal median of 9 frames over a centered `512 x 512`
crop, writing a guaranteed `float32` `(z, y, x)` NumPy file at
`examples/_output/fish-reference.npy`. It computes one z-plane at a time, so
working data are bounded to one plane's 9 input frames plus Dask's median
workspace and the 64 MiB remfile cache (roughly 84 MiB for uint16 source data,
excluding Python/HDF5 overhead). The output is about 29 MiB.

The physical chunks are still `1 x 888 x 2048 x 1`: the spatial crop reduces
the retained array and median workspace, but HDF5 must transfer and decompress
the complete native image-plane chunk for every selected `(time, z)` pair. The
default therefore touches 261 native chunks (about 0.88 GiB uncompressed at
uint16), exactly once each. Use `--tiff path/to/reference.tif` for an optional
TIFF stack when `tifffile` is installed. Use `--no-view` for batch/headless
runs; napari is imported only after outputs are saved, and a missing package or
display produces a friendly message. `--frames` is capped at 50 and all crop,
transport, and cache arguments are validated. See `--help` for every option.
