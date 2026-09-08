# Examples

The repository examples are executable analyses, not pseudocode. At a glance:

| Example | Network | Data moved | Extras | Runtime |
|---|---|---|---|---|
| `examples.local_nwb_zarr` | none | ~1 MiB, generated locally | none | seconds |
| `examples.dandi_hdf5` | DANDI | bounded region (~MiB) | none | seconds–minutes |
| `examples.dandi_region_traces` | DANDI | ~504 MB (13% of movie) | `cellpose` | ~30–60 min |
| `examples.dandi_fish_projection` | DANDI | ~GBs (50 frames × 29 planes) | none | minutes–hour |
| `examples.dandi_dual_channel_cells` | DANDI | bounded, budgeted | `cellpose` (optional detect) | minutes |
| `benchmarks.benchmark_fish_pipeline` | DANDI | ~230 GB read | `cellpose` (+`lindi`) | hours |

No example requires a GPU; Cellpose falls back to CPU with a larger memory
target (the planners refuse rather than exceed it).

## Local NWB-Zarr

```bash
uv run python -m examples.local_nwb_zarr
```

Use this first because it requires no network and exercises durable resume.

## Remote NWB-HDF5 preview

```bash
uv run python -m examples.dandi_hdf5
```

This requests a bounded region through a seekable HTTP range reader. It does not
save the full remote NWB file locally.

## Region traces: full-length traces at 13% of the movie

```bash
uv sync --locked --dev --extra cellpose
uv run python -m examples.dandi_region_traces
```

The strongest case for bounded remote analysis is temporal: a fluorescence
trace needs every timepoint, so no preview can produce it, yet only the chunks
overlapping the chosen region are transferred. On DANDI:000223 the movie is
stored as `(1, 128, 128)` chunks; a 256×256 region across all 1,800 frames
moves ~504 MB of a 3.6 GB movie with 32 KiB transport blocks — and the plan's
`estimated_transport_bytes_read` predicts the block-size sensitivity (9,118 MB
at the 1 MiB default, 924 MB at 64 KiB, 504 MB at 32 KiB) before any download
starts. The example runs the ordinary public API end to end: query, named-axis
slicing, bounded projection, Cellpose segmentation, verified resumable traces.

## Whole-volume fish projection

```bash
time uv run python -m examples.dandi_fish_projection
```

The default processes 50 time frames for all 29 z-planes and saves a full
`(y, x, z)` temporal median. The analysis is expressed as
`np.median(movie[:50], axis="time").astype(np.float32)`; NeuroFlow lowers it to
29 bounded tasks. It touches 1,450 native image-plane chunks. Preserve completed
results and use new `--output` and `--preview` paths for changed runs.

The retained case-study measurement in
`benchmarks/results/fish-case-study-2026-08-13.json` used the legacy median
adapter. It has the same source, selection, float32 output, and task geometry,
but it is historical context—not a timing claim for the NumPy-expression engine.

## Projection to soma traces

Install Cellpose and the optional LINDI transport, then run the end-to-end
publication harness from a clean checkout:

```bash
uv sync --locked --dev --extra cellpose --extra lindi
uv run python -m benchmarks.benchmark_fish_pipeline \
  --backend lindi \
  --output-root publication/runs-publication \
  --memory-limit "4 GiB" --cellpose-device auto \
  --classification publication \
  --record benchmarks/results/publication-fish-soma-traces.json
```

The harness segments each complete z-plane with actual Cellpose, compares the
persisted labels to direct Cellpose on the same projection, extracts the full
movie into `(time, cell)` traces, checks a direct NumPy reference subset, and
records a no-recomputation resume. This is a long network experiment, not a
quickstart smoke test. The `"4 GiB"` target covers the `cpsam` network on CPU;
on a CUDA host `"2 GiB"` suffices.

## Dual-channel candidates

The dual-channel example requires experimentally verified NWB series names.
Its simple detector produces review candidates, not validated cell identities.
Detection is optional and guarded by an explicit memory budget.

See the repository's `examples/README.md` for complete commands and caveats.
