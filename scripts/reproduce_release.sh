#!/usr/bin/env bash
set -euo pipefail

uv sync --locked --dev
uv run ruff check .
uv run basedpyright
uv run python scripts/check_docs.py
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv run pytest \
  --cov=neuroflow \
  --cov=neuroflow_cellpose \
  --cov=neuroflow_pynapple \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=80
uv run python -m benchmarks.run_repetitions \
  --output benchmarks/results/local-summary.json
uv build
