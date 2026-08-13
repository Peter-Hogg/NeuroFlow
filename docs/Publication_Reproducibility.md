# Publication reproducibility protocol

## Network-free verification

```bash
uv sync --locked --dev
uv run ruff check .
uv run basedpyright
uv run pytest --cov=neuroflow --cov=neuroflow_cellpose \
  --cov=neuroflow_pynapple --cov-fail-under=80
uv build
```

## Archive-scale case study

```bash
TIMEFORMAT=$'\nElapsed: %R seconds'
time uv run python -m examples.dandi_fish_projection \
  --output examples/_output/paper-fish-projection.zarr \
  --preview examples/_output/paper-fish-projection-z14.png
```

NeuroFlow intentionally refuses to reuse an output whose provenance describes
a different workflow. Preserve that result and pass new `--output` and
`--preview` paths; do not delete or overwrite a result used in an analysis.

Record execution date, machine, CPU, RAM, operating system, network location,
DANDI identifier/version, asset ID, command, elapsed time, peak RSS, result size,
and verification status. Do not repeatedly benchmark the public archive; retain
the derived result and use resume for follow-up checks.

## Release record

The manuscript must cite an immutable tagged release archived with a DOI. Attach
benchmark JSON, expected outputs, the figure-generating command, and environment
lock file to the archive or associated GigaDB/Code Ocean record.
