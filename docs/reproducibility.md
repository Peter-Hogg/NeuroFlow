# Reproducibility

## Network-free verification

```bash
uv sync --locked --dev
uv run ruff check .
uv run basedpyright
uv run pytest --cov=neuroflow --cov=neuroflow_cellpose \
  --cov=neuroflow_pynapple --cov-fail-under=80
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
```

## Archive-scale case study

```bash
TIMEFORMAT=$'\nElapsed: %R seconds'
time uv run python -m examples.dandi_fish_projection
```

NeuroFlow intentionally refuses to reuse an output whose provenance describes
a different workflow. Preserve that result and pass new `--output` and
`--preview` paths; do not delete or overwrite a result used in an analysis.

Record the execution date, machine, CPU, RAM, operating system, network context,
DANDI identifier and version, asset ID, command, elapsed time, peak RSS, result
size, and verification status. Avoid repeatedly benchmarking a public archive;
retain the derived result and use resume for follow-up checks.

The retained 2026-08-13 case-study record is
`benchmarks/results/fish-case-study-2026-08-13.json`.

## Data citation

The fish example uses the following immutable public dataset:

- Dandiset: `DANDI:000350`
- Version: `0.240822.1759`
- DOI: <https://doi.org/10.48324/dandi.000350/0.240822.1759>
- License: CC-BY-4.0
- Asset ID: `4f898ff7-6084-4e84-a449-f05811c1d951`
- NWB object: `/acquisition/NeuronOnePhotonSeries`
- Primary article DOI: <https://doi.org/10.1016/j.cell.2019.05.050>

Do not infer experimental identity or biological interpretation from image
appearance. Users are responsible for citing the dataset and relevant primary
experimental study in work derived from it.
