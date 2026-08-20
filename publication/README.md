# Publication experiment bundle

This directory is the paper-facing index for NeuroFlow's retained evidence.
Every numerical value in a table or figure must originate in a structured JSON
record under `benchmarks/results/`; do not enter benchmark values by hand.

Evidence is deliberately separated into three classes:

- **publication**: produced by the current engine with the publication schema;
- **current**: a current-engine development or smoke run;
- **historical**: preserved context that predates the current engine and cannot
  support current performance claims.

The network-free experiments are:

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
uv run python publication/generate_tables.py
uv run python -m benchmarks.plot_scaling \
  benchmarks/results/publication-scaling.json \
  publication/figures/bounded-memory-scaling.svg
```

The archive experiment is intentionally manual because it traverses a large
public movie and timing depends on network location and cache state. The
primary configuration is `configs/fish-soma-traces.json`; it covers projection,
real Cellpose, whole-movie traces, direct software/numerical comparisons,
integrity, and resume. See `experiments/archive_fish.md`. The projection-only
configuration remains useful development evidence but is not the north-star
scientific result.

No expert biological validation, DOI deposition, or external publication is
represented as complete here. The relevant protocol and expected artifacts are
documented under `experiments/` and `expected_outputs/`.

`manuscript_draft.md` is a Technical Note scaffold. Bracketed evidence gates
must be filled only from clean retained records; they are intentionally not
silently replaced with development measurements.
