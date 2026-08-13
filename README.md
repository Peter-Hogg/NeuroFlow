# NeuroFlow

NeuroFlow is a lazy execution and interoperability layer for applying existing
Python tools to archive-scale NWB data. It separates data access, partition
planning, execution, and durable output persistence; scientific algorithms stay
in external libraries.

Version 0.1 implements lazy local and DANDI NWB-Zarr discovery, semantic
selection, temporal and spatial partitions, explicit Dask execution, resumable
partition manifests, provenance, Zarr arrays, and Parquet tables. See
[`docs/NeuroFlow_API_Specification.md`](docs/NeuroFlow_API_Specification.md) and
[`docs/NeuroFlow_Architecture_Decisions.md`](docs/NeuroFlow_Architecture_Decisions.md).

## Example

```python
import numpy as np
import neuroflow
from neuroflow.adapters import ArrayOutput, FunctionAdapter
from neuroflow.partition import TimeWindowPlan
from neuroflow.selection import NWBQuery
from neuroflow.storage import ZarrOutput

source = neuroflow.open_source("/data/session.nwb.zarr")
movie = source.select(NWBQuery(name="whole_brain_movie"))
adapter = FunctionAdapter(
    function=lambda block: np.asarray(block, dtype="float32"),
    input_kind="array",
    output=ArrayOutput("float32"),
    splittable_axes=("time",),
)
result = neuroflow.run(
    source=source,
    selection=movie,
    adapter=adapter,
    partition=TimeWindowPlan(size=1000, overlap=100),
    output=ZarrOutput("analysis.zarr"),
)
print(result.plan.summary())
result.execute()
```

Creating the source, selection, plan, or result handle does not execute numerical
work. Execution begins only at `result.execute()` or with `execute=True`.

Regularly sampled series support sample- or duration-based temporal windows.
Irregular timestamps are exposed with `selection.as_dask_timestamps()` and remain
lazy; timestamp-aligned planning is rejected until a future planner can do so
without silently materializing the full coordinate vector.

Outputs use `mode="create"`, `"overwrite"`, or `"append"`. Existing managed
outputs are resumed only when their deterministic workflow provenance matches;
unmanaged paths and stale provenance are rejected unless overwrite was explicitly
requested.

```bash
neuroflow inspect /data/session.nwb.zarr
neuroflow status analysis.zarr
```

## Tiled segmentation

The second vertical slice supports external segmentation functions without
placing a segmentation algorithm in core. Each function invocation receives one
bounded tile including its declared halo and returns local labels plus an object
table:

```python
from neuroflow.adapters import SegmentationFunctionAdapter
from neuroflow.partition import SpatialTilePlan
from neuroflow.storage import SegmentationOutput

adapter = SegmentationFunctionAdapter(
    function=my_segmentation_function,
    name="my-segmenter",
    version="1",
    requires_overlap={"y": 32, "x": 32},
)
result = neuroflow.run(
    source=source,
    selection=movie,
    adapter=adapter,
    partition=SpatialTilePlan(
        tile_shape=(512, 512),
        halo=(32, 32),
        axes=("y", "x"),
    ),
    output=SegmentationOutput("segmentation-result"),
)
result.execute()
```

NeuroFlow trims halos, assigns collision-free global IDs, writes dense labels to
Zarr and object rows to partitioned Parquet, and commits one completion manifest
only after both outputs succeed. Results are marked `unmerged`: reconciling
scientific objects across tile boundaries remains an explicit adapter merge step,
never an implicit assumption by core.

## Integrity and repair

Every partition manifest includes output checksums. Audit a result without reading
numerical payloads, or perform a full bounded checksum audit:

```python
result.verify(checksums=False)  # manifests and output presence
result.verify()                 # reads one partition at a time
```

```bash
neuroflow verify analysis.zarr --no-checksums
neuroflow verify analysis.zarr
```

Resume validates completed partition checksums before skipping them. Missing or
corrupt outputs are recomputed from their bounded input slices; valid partitions
are not rerun.

## Optional Cellpose integration

Cellpose is not imported by or required for NeuroFlow core:

```bash
uv sync --extra cellpose
```

```python
from neuroflow_cellpose import CellposeAdapter

adapter = CellposeAdapter(
    pretrained_model="cpsam_v2",
    gpu=True,
    diameter=30.0,
    cellprob_threshold=0.0,
    halo={"y": 32, "x": 32},
)
```

The adapter constructs Cellpose models lazily inside worker threads, caches one
model per thread, forwards declared Cellpose 4.x evaluation parameters, converts
masks into label/object outputs, and records the installed Cellpose version.
Model weights may be downloaded by Cellpose on first use; NeuroFlow does not own
or bundle those weights. Users must select a model explicitly and are responsible
for checking the model and dataset licenses for their use case.

## Optional Pynapple integration

Pynapple is also isolated from core:

```bash
uv sync --extra pynapple
```

```python
from neuroflow_pynapple import PynappleAdapter
from neuroflow.partition import TimeWindowPlan
from neuroflow.storage import ParquetOutput

adapter = PynappleAdapter(
    function=my_pynapple_analysis,
    parameters={"window": 0.5},
)
result = neuroflow.run(
    source=source,
    selection=time_series,
    adapter=adapter,
    partition=TimeWindowPlan(size="60 s", overlap="5 s"),
    output=ParquetOutput("session-analysis"),
)
```

Each task receives a bounded `Tsd`, `TsdFrame`, or `TsdTensor`. Regular NWB
timestamps are reconstructed from rate metadata; irregular timestamp arrays are
sliced alongside the corresponding data partition. User-function outputs are
normalized into partitioned Parquet tables, and the installed Pynapple version is
recorded in provenance.

## Development

```bash
uv sync --dev
uv run pytest
uv run ruff check .
```
