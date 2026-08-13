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
`(y, x, z)` temporal median. It touches 1,450 native image-plane chunks. Preserve
completed results and use new `--output` and `--preview` paths for changed runs.

The retained case-study measurement is in
`benchmarks/results/fish-case-study-2026-08-13.json`.

## Dual-channel candidates

The dual-channel example requires experimentally verified NWB series names.
Its simple detector produces review candidates, not validated cell identities.
Detection is optional and guarded by an explicit memory budget.

See the repository's `examples/README.md` for complete commands and caveats.
