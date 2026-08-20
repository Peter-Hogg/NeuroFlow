#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-publication/runs/fish-projection}"
mkdir -p benchmarks/results publication/runs

/usr/bin/time -v -o benchmarks/results/publication-fish-resource-usage.txt \
  uv run python -m benchmarks.benchmark_archive \
  --result "${output_root}.zarr" \
  --preview "${output_root}-z14.png" \
  --record benchmarks/results/publication-fish-projection.json \
  | tee benchmarks/results/publication-fish-run.txt

du -sb "${output_root}.zarr" "${output_root}-z14.png" \
  > benchmarks/results/publication-fish-output-sizes.txt
