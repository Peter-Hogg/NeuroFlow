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
  --output benchmarks/results/publication-scaling.json
uv run python -m benchmarks.benchmark_resume_integrity \
  --output benchmarks/results/publication-resume-integrity.json
uv run python publication/generate_tables.py
uv run python -m benchmarks.plot_scaling \
  benchmarks/results/publication-scaling.json \
  publication/figures/bounded-memory-scaling.svg
```

The archive experiment is intentionally manual because it transfers public
data and its timing depends on network location and cache state. See
`configs/archive-fish.json` and `experiments/archive_fish.md`.

No expert biological validation, DOI deposition, or external publication is
represented as complete here. The relevant protocol and expected artifacts are
documented under `experiments/` and `expected_outputs/`.
