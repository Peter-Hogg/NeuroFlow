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
uv sync --locked --dev --extra cellpose
uv sync --locked --dev --extra pynapple
uv sync --locked --dev --extra lindi
uv sync --locked --dev --extra baselines
```

Without `uv`, install from the checkout with pip: `pip install .` (add extras
as `pip install ".[cellpose,lindi]"`).

## Memory limits in one paragraph

Every `memory_limit` is a **total process-memory target**, not a per-array
allowance: the planner subtracts a measured process floor (interpreter,
libraries, caches) and fits partitioned work into what remains. Defaults are
1 GiB for computations, 2 GiB for persists, and 4 GiB for Cellpose
segmentation. If a target cannot fit — for example `"2 GiB"` with the `cpsam`
model on CPU, whose loaded network alone holds ~2 GiB — NeuroFlow refuses up
front with the reason and the fix, rather than failing mid-run. You do not
need to understand the planner: run the commands below, and if a refusal
appears, do what it says (usually raise the target or switch device).

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
assert result.verify().valid
```

Use named or contiguous NumPy slicing and a declared memory limit. Arithmetic,
supported NumPy calls, metadata inspection, and `repr()` remain lazy. Numerical
I/O starts only at `.compute()` or `.persist()`.

## Archive-scale soma traces

The flagship path keeps infrastructure choices out of the scientific code:

```python
import numpy as np
import neuroflow
from neuroflow.selection import NWBQuery

source = neuroflow.open_dandi(
    "DANDI:000350@0.240822.1759", backend="lindi"
)
movie = neuroflow.NeuroArray(
    source,
    source.select(
        NWBQuery(
            asset="4f898ff7-6084-4e84-a449-f05811c1d951",
            name="NeuronOnePhotonSeries",
        )
    ),
)
projection = np.median(movie[:50], axis="time").astype("float32").persist(
    "projection.zarr", memory_limit="2 GiB"
)
labels = projection.cellpose(
    output="labels.zarr", pretrained_model="cpsam", memory_limit="4 GiB"
)
plan = movie.plan_traces(labels, memory_limit="2 GiB")
print(plan.summary())
traces = movie.extract_traces(
    labels, output="traces.zarr", memory_limit="2 GiB"
)
```

The trace result is oriented `(time, cell)` and includes `timestamps` and
`cell_ids`. NeuroFlow chooses the time window from the memory target and source
chunks; `time_chunk` remains an expert override. The `"4 GiB"` segmentation
target accommodates the `cpsam` network on CPU; on a CUDA GPU, where the
weights live in VRAM, `"2 GiB"` suffices.

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
