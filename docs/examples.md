# Examples

The repository examples are executable analyses, not pseudocode.

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
  --classification publication \
  --record benchmarks/results/publication-fish-soma-traces.json
```

The harness segments each complete z-plane with actual Cellpose, compares the
persisted labels to direct Cellpose on the same projection, extracts the full
movie into `(time, cell)` traces, checks a direct NumPy reference subset, and
records a no-recomputation resume. This is a long network experiment, not a
quickstart smoke test.

## Dual-channel candidates

The dual-channel example requires experimentally verified NWB series names.
Its simple detector produces review candidates, not validated cell identities.
Detection is optional and guarded by an explicit memory budget.

See the repository's `examples/README.md` for complete commands and caveats.
