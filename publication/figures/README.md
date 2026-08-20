# Generated figures

Figures in this directory must be generated from retained structured results.
For example:

```bash
uv run python -m benchmarks.plot_scaling \
  benchmarks/results/publication-scaling.json \
  publication/figures/bounded-memory-scaling.svg
```

The SVG generator has no plotting-library dependency and labels peak RSS as a
measured process high-water mark, not a hard memory limit.
