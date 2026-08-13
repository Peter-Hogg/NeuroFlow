# NeuroFlow

**Run ordinary Python analyses on NWB recordings that are too large—or too
remote—to load all at once.**

NeuroFlow opens an NWB object lazily, divides it into scientifically meaningful
pieces, sends one bounded NumPy array at a time to your function, and saves each
result durably. Your analysis code stays small; NeuroFlow handles remote access,
Dask execution, provenance, resume, and integrity checks.

```text
DANDI or local NWB → choose a series → plan bounded work → your function
                                                            ↓
                                      verified Zarr / Parquet result
```

- Inspect and select data before reading its numerical payload.
- Give your function a normal NumPy array, not a framework-specific object.
- Resume interrupted jobs from verified partitions instead of starting over.
- Read local or DANDI-hosted NWB-Zarr and NWB-HDF5 data.

NeuroFlow is an early `0.1` project. Its strongest path is chunked NWB-Zarr;
remote HDF5 support is intentionally conservative and described below.

## Try it in two minutes

Install the development environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
```

Start with the fully local example. It creates a tiny NWB-Zarr recording, runs
four chunks through a custom function, verifies the output, and tests resume:

```bash
uv run python -m examples.local_nwb_zarr
```

For a real recording, run the DANDI example:

```bash
uv run python -m examples.dandi_hdf5
```

It uses Dask to request one `128 × 128` image block and writes a viewable
`examples/_output/dandi-hdf5-preview.png`. It then processes the selected
`1 × 512 × 512` max projection as one bounded task, writes a Zarr result,
verifies it, and reopens it lazily. The complete remote NWB file is never saved
locally or loaded into memory.

The recording is pinned for reproducibility and the default numerical workload
is about 1 MiB. Internet access is required. See [the examples guide](examples/README.md)
for the exact asset, generated files, and options.

For a real calcium movie, the fish example uses a plain NumPy temporal median
through NeuroFlow to project 50 frames across all 29 z-planes of Misha Ahrens'
150 GB whole-brain zebrafish recording:

```bash
uv run python -m examples.dandi_fish_projection
```

NeuroFlow partitions the physical reads by z-plane, writes the full
`888 × 2048 × 29` result as a y/x-tiled Zarr array, records provenance, supports
resume, and verifies every partition. It also saves z-plane 14 as a PNG preview.
The complete remote recording is never downloaded.

For the common path, NeuroFlow also exposes a named-axis, NumPy-like API. The
same durable engine handles partitioning, Dask execution, Zarr output, resume,
and verification:

```python
movie = neuroflow.load("DANDI:000350@0.240822.1759", name="NeuronOnePhotonSeries")
projection = movie.isel(time=slice(0, 50)).median(
    "time",
    output="projection.zarr",
    chunks=(256, 256, 1),
    max_workers=2,
)
cells = projection.segment(
    cellpose_adapter,
    output="cells",
    tile_shape=(1,),
    axes=("z",),
    max_workers=1,
)
```

Persisted arrays are composable inputs through `neuroflow.open_array()`. Dense
labels can be passed to `movie.extract_traces(...)`, which reads bounded time
windows and z-planes rather than materializing the movie or a cell-by-voxel
matrix.

## Bring your own function

The same API can be pointed at an existing NWB-Zarr dataset:

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

## What “lazy” means here

- **NWB-Zarr:** numerical arrays remain object-store-backed Zarr arrays. Native
  chunks are preserved and Dask can fetch independent chunks lazily.
- **NWB-HDF5:** local files remain h5py datasets. Remote files are opened through
  PyNWB's recommended `remfile` transport with a bounded 64 MiB in-memory cache;
  fsspec remains available with `storage_options={"transport": "fsspec"}`. No
  complete-file download or eager array conversion is performed. Metadata
  discovery may still require many range requests because HDF5 metadata is not
  object-store-native. See PyNWB's official
  [streaming guide](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html).
- HDF5 datasets may be contiguous (`native_chunks=None`). Dask can still divide
  them into logical blocks, but those blocks are not physical HDF5 chunks and
  may require more range requests. The threaded scheduler is supported;
  process and distributed schedulers are rejected because open h5py/file
  handles are not generally serializable.
- The remote server must honor byte-range requests. Opening fails explicitly if
  a seekable range reader cannot be established. Users should choose bounded
  partition plans and inspect `result.plan` before execution.
- PyNWB datasets stay lazy only while their underlying IO handle is open.
  NeuroFlow therefore owns the PyNWB, h5py, and transport handles together and
  closes them in order when the source context ends. Compute all lazy reads
  before calling `source.close()`.

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
uv run basedpyright
```

## Testing

Every push and pull request runs the network-free test suite on GitHub Actions:
linting, static type checking, and pytest with coverage. The suite covers local
Zarr and HDF5 selection, mocked DANDI routing, partition planning, bounded
execution, persistence, checksums, repair and resume, segmentation, optional
adapter boundaries, and example configuration. The current suite has 38 tests,
measures 84% statement coverage, and enforces an 80% coverage floor in CI.

The live DANDI example is deliberately **not** part of CI. Archive availability
and network conditions should not make ordinary builds flaky; run it separately
with the command above when validating remote integration.

Design rationale and the formal API live in [`docs/`](docs/). Those documents
are useful when extending NeuroFlow; this README is the place to start when
using it.
