# Flagship archive fish experiment

Run from a clean immutable release-candidate checkout and a network context
that will be reported with the result:

```bash
uv sync --locked --dev --extra cellpose --extra lindi
uv run python -m benchmarks.benchmark_fish_pipeline \
  --backend remfile \
  --memory-limit "2 GiB" \
  --projection-frames 50 \
  --validation-frames 1 \
  --cellpose-model cpsam \
  --classification publication \
  --record benchmarks/results/publication-fish-soma-traces.json
```

The runner refuses publication classification if Git is dirty. It retains:

- the 323 GB logical movie identity and physical chunks;
- projection transfer, memory, wall time, output, checksum, and integrity;
- actual Cellpose version/model/settings, object count, output checksum, and an
  exact direct-Cellpose comparison on every projection plane;
- bounded whole-movie `(time, cell)` traces, source chunks touched, transfer,
  peak RSS, wall time, output size, checksum, and integrity;
- a direct plane-wise NumPy comparison on the leading movie frame;
- a second identical trace call proving all completed partitions are resumed
  without recomputation.

Before running, record whether the host, proxy, operating system, or filesystem
may cache HTTP responses. Preserve the outputs and logs rather than rerunning
for favorable timing. Large Zarr outputs remain ignored; upload them as run
artifacts and retain the small JSON/preview evidence.

Run the manual LINDI/Dask comparison over the exact same masks:

```bash
uv run python -m benchmarks.benchmark_fish_trace_baseline \
  --backend lindi \
  --labels publication/runs/fish-cellpose.zarr \
  --reference-traces publication/runs/fish-traces.zarr \
  --output publication/runs/fish-lindi-dask-traces.zarr \
  --record benchmarks/results/publication-fish-lindi-dask-traces.json \
  --classification publication
```

Repeat with `--backend remfile` when the full four-configuration comparison is
required. The manual baseline intentionally lacks resume, integrity manifests,
automatic memory planning, and output provenance; those differences must be
reported alongside wall time and memory rather than reduced to a speed contest.

The retained projection-only development record has `classification: current`
and `dirty: true`. The 2026-08-13 legacy-engine record remains historical.
Neither is final publication evidence.
