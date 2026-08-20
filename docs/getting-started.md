# Getting started

## Installation

NeuroFlow requires Python 3.10 or newer. For a source checkout:

```bash
git clone https://github.com/Peter-Hogg/NeuroFlow.git
cd NeuroFlow
uv sync --locked --dev
```

Optional integrations are deliberately separate:

```bash
uv sync --dev --extra cellpose
uv sync --dev --extra pynapple
uv sync --locked --dev --extra baselines
```

## A bounded local workflow

Start with the network-free example, which creates a small NWB-Zarr file and
demonstrates execution, verification, lazy reopening, and resume:

```bash
uv run python -m examples.local_nwb_zarr
```

Before executing a large analysis, inspect its source and plan:

```python
import numpy as np
import neuroflow

movie = neuroflow.load("session.nwb.zarr", name="movie")
print(movie.axes, movie.shape)
projection = np.median(movie[:50], axis="time")
workflow = projection.to_spec(
    "projection.zarr",
    chunks=(256, 256),
    memory_limit="2 GiB",
)
workflow.to_json("projection-workflow.json")
print(workflow.plan().summary())
result = neuroflow.reproduce(workflow)
assert result.workflow.verify().valid
```

Use named or contiguous NumPy slicing and a declared memory limit. Arithmetic,
supported NumPy calls, metadata inspection, and `repr()` remain lazy. Numerical
I/O starts only at `.compute()` or `.persist()`.

The workflow record can be checked and rerun without importing the analysis
script:

```bash
uv run neuroflow plan projection-workflow.json
uv run neuroflow reproduce projection-workflow.json \
  --output projection-reproduction.zarr
uv run neuroflow verify projection-reproduction.zarr
```

## Result safety

NeuroFlow refuses to resume an output whose workflow identity differs. Keep the
old result as evidence and choose a fresh output path. Overwrite is explicit and
refuses filesystem roots and other protected broad local targets. An output
also cannot equal, contain, or be nested within any active input path.
