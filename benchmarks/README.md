# Reproducible benchmarks

Benchmark records use `neuroflow.benchmarking` schema version 1 and include
software/Git identity, environment, source, partition policy, available I/O and
memory measurements, checksums, numerical validation, resume information, and
explicit `null` values for unavailable measurements.

Results have one of three classifications:

- `publication`: a retained current-engine experiment intended for the paper;
- `current`: a current-engine smoke or development measurement;
- `historical`: preserved evidence that must not support current-engine claims.

## Network-free experiments

```bash
uv run python -m benchmarks.benchmark_projection \
  --classification publication \
  --output benchmarks/results/publication-local-projection.json
uv run python -m benchmarks.benchmark_scaling \
  --sizes 128,256,512 --frames 16 --repetitions 3 \
  --classification publication \
  --output benchmarks/results/publication-scaling.json
uv run python -m benchmarks.benchmark_resume_integrity \
  --classification publication \
  --output benchmarks/results/publication-resume-integrity.json
```

The projection benchmark compares the same temporal median with direct NumPy,
direct Dask, and NeuroFlow. It is a correctness and overhead experiment; it
does not assume that NeuroFlow should beat in-memory NumPy on small arrays.
Scaling runs use a fresh process per size so peak RSS high-water marks are not
carried over from a larger earlier run.

`baselines.py` contains fair direct PyNWB/HDMF-Zarr, direct Dask, LINDI, and
source-chunk-oriented trace implementations over the same selection and
operation. Install the optional
LINDI path with `uv sync --locked --dev --extra baselines`. Report versions,
selection, cache state, network context, and numerical equivalence for every
comparison; LINDI is a remote-access baseline, not an intentionally handicapped
competitor.

## Archive experiment

Run the projection-only current engine through the schema-writing wrapper:

```bash
bash benchmarks/run_fish_case_study.sh publication/runs/fish-projection
```

This public DANDI experiment should run once from a documented network context
with a fresh result path. Retain its result rather than repeatedly transferring
archive data for favorable timings. Numerical validation remains pending until
an independent reference comparison is retained.

The flagship publication command is instead:

```bash
uv run python -m benchmarks.benchmark_fish_pipeline \
  --backend remfile --classification publication \
  --record benchmarks/results/publication-fish-soma-traces.json
```

It records projection, direct-equivalent real Cellpose, bounded whole-movie
traces, a direct NumPy reference subset, integrity, and completed-result resume.
`benchmark_fish_trace_baseline` runs the manual PyNWB + remfile/LINDI + Dask
trace workflow over the exact same masks. Both publication runners reject a
dirty Git tree.

`results/fish-case-study-2026-08-13.json` is explicitly historical because it
predates the current engine. `results/local-summary.json` uses a legacy summary
format. Neither is silently converted into publication evidence.
