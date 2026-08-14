#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-examples/_output/case-study-fish-projection}"
mkdir -p benchmarks/results

/usr/bin/time -v -o benchmarks/results/fish-resource-usage.txt \
  uv run python -m examples.dandi_fish_projection \
  --output "${output_root}.zarr" \
  --preview "${output_root}-z14.png" \
  | tee benchmarks/results/fish-run.txt

du -sb "${output_root}.zarr" "${output_root}-z14.png" \
  > benchmarks/results/fish-output-sizes.txt
