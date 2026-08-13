# Getting started

## Installation

NeuroFlow requires Python 3.10 or newer. For a source checkout:

```bash
git clone https://github.com/Peter-Hogg/NeuroFlow.git
cd NeuroFlow
uv sync --dev
```

Optional integrations are deliberately separate:

```bash
uv sync --dev --extra cellpose
uv sync --dev --extra pynapple
```

## A bounded local workflow

Start with the network-free example, which creates a small NWB-Zarr file and
demonstrates execution, verification, lazy reopening, and resume:

```bash
uv run python -m examples.local_nwb_zarr
```

Before executing a large analysis, inspect its source and plan:

```python
import neuroflow

movie = neuroflow.load("session.nwb.zarr", name="movie")
print(movie.axes, movie.shape)
```

Use named slicing and a declared memory limit. Numerical I/O starts only when
an operation executes or a lazy result is explicitly computed.

## Result safety

NeuroFlow refuses to resume an output whose workflow identity differs. Keep the
old result as evidence and choose a fresh output path. Overwrite is explicit and
refuses filesystem roots and other protected broad local targets.
