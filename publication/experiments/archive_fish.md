# Current-engine archive experiment

Run this once from a network context that will be reported with the result:

```bash
uv run python -m benchmarks.benchmark_archive \
  --frames 50 --tile-y 256 --tile-x 256 --max-workers 1 \
  --cache-size-mib 64 --block-size 262144 \
  --result publication/runs/fish-projection.zarr \
  --preview publication/runs/fish-projection-z14.png \
  --record benchmarks/results/publication-fish-projection.json
```

Use a fresh process and result path. Before running, record whether the host,
proxy, operating system, or filesystem may cache HTTP responses. Afterward,
retain the JSON record, output, preview, command log, and `/usr/bin/time -v`
output. Validate the projection against an independently computed reference;
the archive script deliberately records numerical validation as pending until
that comparison exists.

The 2026-08-13 fish case-study JSON remains historical evidence because it
predates the current expression and provenance engine. It must not be pooled
with the current benchmark.
