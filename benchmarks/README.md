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

`benchmark_lindi_equivalence` is the cheap transport-independence check. It
persists the same NumPy temporal median over the same few frames and z-planes of
the same asset through remfile and through LINDI, then compares the two reopened
Zarr outputs for exact equality:

```bash
uv run python -m benchmarks.benchmark_lindi_equivalence \
  --output-root /tmp/lindi-equivalence \
  --record benchmarks/results/current-lindi-equivalence.json
```

Each backend gets its own subprocess so the peak RSS values are independent.
`bytes_read` is measured for remfile and stays `null` for LINDI, which exposes
no transport counter. The slice is small on purpose: this is an equivalence
claim, not a throughput claim.

`benchmark_dandi_smoke` is the generality check: the ordinary public workflow
(discover with `NWBQuery(neurodata_type=...)`, inspect inferred axes and
physical chunks, preflight a plan, persist a bounded temporal mean, verify,
compare against a plain h5py + NumPy reference) on a dataset chosen entirely
from the command line, with no dataset-specific code:

```bash
uv run python -m benchmarks.benchmark_dandi_smoke \
  --dandiset "DANDI:000223@0.260528.0906" \
  --asset cc499fe1-fe23-42aa-8db0-0e689970fb89 \
  --neurodata-type TwoPhotonSeries --frames 96 \
  --expect-axes time,y,x \
  --output-root /tmp/dandi-smoke \
  --record benchmarks/results/current-dandi-smoke-000223.json
```

`--expect-axes` turns the axis inference into an assertion instead of a hidden
assumption. The record separates `engine_phase_peak_rss_bytes` from the
whole-process peak, because the harness's own independent reference computation
loads raw frames into the same process.

`results/fish-case-study-2026-08-13.json` is explicitly historical because it
predates the current engine. `results/local-summary.json` uses a legacy summary
format. Neither is silently converted into publication evidence.
