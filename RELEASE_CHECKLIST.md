# Software release checklist

## Repository and software

- [x] Maintainer selects an OSI-approved license and adds `LICENSE` plus matching
  `pyproject.toml`, `CITATION.cff`, and `.zenodo.json` metadata (BSD-3-Clause;
  consistency machine-checked by `tools/check_metadata.py`).
- [x] Author, project URL, classifiers, citation, changelog, contribution,
  conduct, security, and version metadata are present and machine-checked.
- [x] Locked uv environments support Python 3.10 through 3.13 in CI; Python
  3.10 has an explicit `tomli` fallback.
- [x] Network-free tests compare NumPy, NWB-Zarr, and NWB-HDF5 results.
- [x] A real local LINDI/PyNWB bridge test is in the optional-dependency CI job.
- [x] Trace outputs have planning, coordinates, provenance, manifests,
  checksums, verification, and per-partition resume.
- [x] Friendly Cellpose segmentation processes complete 2-D images or complete
  y/x planes and prevents accidental unreconciled y/x tiles.
- [ ] Public GitHub matrix, optional LINDI, real Cellpose, docs, package, and
  CodeQL workflows pass on the release-candidate commit.
- [ ] Docker image builds in an environment with a working daemon and its base
  image digest is retained.

## Validation evidence

- [x] Deterministic trace tests compare against direct NumPy with exact output,
  including a label spanning multiple source chunks.
- [x] Trace planning is tested to read no movie values and to skip empty source
  chunks.
- [x] Deterministic interruption, resume, corruption detection, and selective
  repair are covered by a retained local benchmark harness.
- [x] An opt-in test runs actual Cellpose and compares exact labels with direct
  Cellpose on the same deterministic projection.
- [ ] Retain the successful real Cellpose job log/model cache identity for the
  release candidate; mocked tests alone are not publication evidence.
- [ ] Run and retain the clean end-to-end DANDI fish projection → Cellpose →
  whole-movie trace record with memory, transfer, wall time, checksums,
  numerical validation, integrity, and resume.
- [ ] Run and retain the clean PyNWB + LINDI + Dask trace baseline over the exact
  same masks; run the remfile baseline if used in the manuscript table.
- [ ] Complete expert/manual biological assessment, or state explicitly in the
  manuscript that only software-path equivalence was established.
- [x] Preserve the legacy fish record as `historical` and the dirty current
  projection measurement as development evidence, not publication evidence.

## Final benchmark run-book

The detailed evidence ledger was retired from the public tree (removed in
`04633da`; its final version is readable at `7e54b6c:PUBLICATION_READINESS.md`).
The exact clean-run sequence it prescribed, in order — every publication-classified
runner refuses a dirty Git tree:

```bash
# 0. Gates green, tree committed; record the commit for the manuscript.
uv run pytest -q && uv run ruff check . && uv run basedpyright
git status --porcelain && git rev-parse HEAD

# 1. Local correctness, resume, integrity (publication classification).
uv run python benchmarks/benchmark_projection.py \
    --record benchmarks/results/publication-local-projection.json \
    --classification publication
uv run python benchmarks/benchmark_resume_integrity.py \
    --record benchmarks/results/publication-resume-integrity.json \
    --classification publication

# 2. Component memory attribution (include cuda probes on a GPU host).
uv run python benchmarks/memory_attribution.py \
    --record benchmarks/results/publication-memory-attribution.json

# 3. Resource scaling (local fixture) and second-dataset smoke, three
#    repetitions each.
uv run python -m benchmarks.benchmark_resource_scaling \
    --fixture-root /tmp/nf-scaling --output-root /tmp/nf-scaling-out \
    --frames 192 --repetitions 3 \
    --record benchmarks/results/publication-resource-scaling.json \
    --classification publication
uv run python -m benchmarks.benchmark_dandi_smoke \
    --dandiset "DANDI:000223@0.260528.0906" \
    --asset cc499fe1-fe23-42aa-8db0-0e689970fb89 \
    --neurodata-type TwoPhotonSeries --frames 96 \
    --memory-limit "2 GiB" --expect-axes time,y,x --backend remfile \
    --repetitions 3 --output-root /tmp/nf-smoke \
    --record benchmarks/results/publication-dandi-smoke-000223.json \
    --classification publication

# 4. Archive-scale fish pipeline (hours, ~230 GB read). Output root must NOT
#    be publication/runs, which holds retained development evidence.
uv run python benchmarks/benchmark_fish_pipeline.py \
    --output-root publication/runs-publication \
    --record benchmarks/results/publication-fish-soma-traces-remfile.json \
    --backend remfile --memory-limit "4 GiB" --cellpose-device auto \
    --block-size 1048576 --cache-size-mib 64 \
    --classification publication

# 5. Same workflow over LINDI for transport independence.
uv run python benchmarks/benchmark_fish_pipeline.py \
    --output-root publication/runs-publication-lindi \
    --record benchmarks/results/publication-fish-soma-traces-lindi.json \
    --backend lindi --memory-limit "4 GiB" --cellpose-device auto \
    --classification publication

# 6. Fair baselines over the retained masks from step 4, configured for
#    parity: identical block/cache, and --time-chunk equal to the NeuroFlow
#    record's execution.trace_plan.time_window.
uv run python benchmarks/benchmark_fish_trace_baseline.py \
    --labels publication/runs-publication/fish-cellpose.zarr \
    --reference-traces publication/runs-publication/fish-traces.zarr \
    --output /tmp/nf-baseline-traces.zarr \
    --record benchmarks/results/publication-fish-remfile-dask-traces.json \
    --backend remfile --block-size 1048576 --cache-size-mib 64 \
    --time-chunk <time_window from step 4's record> \
    --classification publication
uv run python benchmarks/benchmark_fish_trace_baseline.py \
    --labels publication/runs-publication/fish-cellpose.zarr \
    --reference-traces publication/runs-publication/fish-traces.zarr \
    --output /tmp/nf-baseline-traces-lindi.zarr \
    --record benchmarks/results/publication-fish-lindi-dask-traces.json \
    --backend lindi \
    --time-chunk <time_window from step 4's record> \
    --classification publication
```

Notes: `--memory-limit "2 GiB"` in steps 4-5 is refused when `cpsam` runs on
CPU, by design — use 4 GiB or `--cellpose-device cuda`. Block size, cache
size, and temporal window are identical on both sides of the step-6
comparison and recorded in both records (`transport_configuration` /
`configuration_parity`); both sides consume the identical retained masks and
run their accumulation single-threaded, so neither tool holds a concurrency
advantage.

## Release and archival actions

- [ ] Freeze the public API and create signed tag `v0.1.0` from the exact clean
  commit used for publication experiments.
- [ ] Create a GitHub release and archive the tagged software in Zenodo and/or
  Software Heritage.
- [ ] Add assigned software DOI and release date to citation/archive metadata.
- [x] Record DANDI:000350 version, DOI, asset, object, and primary article in the
  reproducibility guide.
