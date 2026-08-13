# Reproducible benchmarks

Run the deterministic network-free benchmark with:

```bash
uv run python -m benchmarks.benchmark_projection \
  --output benchmarks/results/local-projection.json
```

Repeat runs should be retained as separate JSON records. Report medians and
distributions rather than selecting a single favorable run. The live fish
example is the archive-scale case study; use a fresh output path and record the
DANDI version, asset ID, wall time, peak RSS, output size, and network context.

The local benchmark compares direct NumPy with NeuroFlow, verifies the durable
result, and records numerical error, wall time, peak RSS, and source/result disk
use. It is a correctness and overhead benchmark, not a claim that NeuroFlow
should outperform in-memory NumPy on small arrays.

Run five local repetitions and create a summary with:

```bash
uv run python -m benchmarks.run_repetitions
```

Run the public DANDI case study once, capturing GNU Time resource statistics,
with `bash benchmarks/run_fish_case_study.sh`. Reuse its output for later checks
instead of repeatedly transferring archive data.
