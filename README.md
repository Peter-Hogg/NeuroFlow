# NeuroFlow

**Run ordinary Python analyses on NWB recordings that are too large—or too
remote—to load all at once.**

NeuroFlow opens an NWB object lazily and supports a documented subset of NumPy
without reading the numerical payload. At an explicit persistence boundary it
divides the expression into bounded tasks, sends one NumPy block at a time
through the fused operation, and saves each result durably. NeuroFlow handles
remote access, Dask execution, provenance, resume, and integrity checks.

```text
DANDI or local NWB → choose a series → plan bounded work → your function
                                                            ↓
                                      verified Zarr / Parquet result
```

- Inspect and select data before reading its numerical payload.
- Write common arithmetic, ufuncs, casts, and reductions with NumPy syntax.
- Give custom adapters a normal NumPy array, not a framework-specific object.
- Resume interrupted jobs from verified partitions instead of starting over.
- Read local or DANDI-hosted NWB-Zarr and NWB-HDF5 data.

NeuroFlow is an early `0.1` project. Chunked NWB-Zarr is the simplest path;
remote NWB-HDF5 can use either remfile or the optional LINDI backend.

The [complete user guide and API reference](https://peter-hogg.github.io/NeuroFlow/)
build from `docs/` with Sphinx. After
GitHub Pages is enabled with **GitHub Actions** as its source, the documentation
workflow publishes every successful `master` build.

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
through NeuroFlow to project 50 frames across all 29 z-planes of a 323 GB
logical whole-brain zebrafish recording:

```bash
uv run python -m examples.dandi_fish_projection
```

NeuroFlow partitions the physical reads by z-plane, writes the full
`888 × 2048 × 29` result as a y/x-tiled Zarr array, records provenance, supports
resume, and verifies every partition. It also saves z-plane 14 as a PNG preview.
The complete remote recording is never downloaded.

For an immediately viewable 29-plane reference stack, the companion example
saves a bounded temporal median as a `(z, y, x)` NumPy array and optionally
opens it in napari:

```bash
uv run python -m examples.dandi_fish_reference_volume --no-view
```

Its default uses 9 frames and a `512 x 512` crop. The crop bounds output and
median memory, but—because the HDF5 chunks contain complete image planes—it
does not avoid transferring the full native chunk for each selected time/z
pair. See the [examples guide](examples/README.md) for resource details and
optional TIFF/napari usage.

For the common path, `NeuroArray` implements a deliberately supported NumPy
subset. Building this expression performs no numerical reads; `.persist()` is
the explicit bounded execution boundary:

```python
import numpy as np
import neuroflow
from neuroflow.selection import NWBQuery

fish = neuroflow.open_dandi(
    "DANDI:000350@0.240822.1759", backend="lindi"
)
selected = fish.select(
    NWBQuery(
        asset="4f898ff7-6084-4e84-a449-f05811c1d951",
        name="NeuronOnePhotonSeries",
    )
)
movie = neuroflow.NeuroArray(fish, selected)

expression = np.median(movie[:50], axis="time").astype("float32")
# Optional preflight: partitioning, task count, source-chunk and
# transport-level read estimates, all before any numerical I/O.
print(expression.plan("projection.zarr", memory_limit="2 GiB").summary())
projection = expression.persist("projection.zarr", memory_limit="2 GiB")
masks = projection.cellpose(
    pretrained_model="cpsam",
    output="fish-cellpose.zarr",
    # Segmentation asks for more than the other stages because `memory_limit`
    # is a total process target and one loaded `cpsam` network is ~1.9 GiB
    # resident on CPU. Pass `gpu=True` to move the weights into VRAM instead.
    memory_limit="4 GiB",
)
print(movie.plan_traces(masks, memory_limit="2 GiB").summary())
traces = movie.extract_traces(
    masks,
    output="fish-traces.zarr",
    memory_limit="2 GiB",
)
assert neuroflow.open_result("fish-traces.zarr").verify().valid
```

`memory_limit` is an approximate **total process-memory target** — the number a
laptop user means by "stay under 2 GiB" — not a per-task allowance. The planner
charges measured process overhead once, sizes partitions and concurrency from
the remainder, and refuses work that cannot fit rather than overrunning
silently. It is a planning target, not an OS-enforced cap; every run records
planned and measured peak RSS side by side. See
[the memory-semantics section](https://peter-hogg.github.io/NeuroFlow/High_Level_API.html)
for the decomposition and its measured basis.

Persisted arrays are composable inputs through `neuroflow.open_array()`, which
requires a complete result and verifies partition checksums by default. Dense
labels can be passed to `movie.extract_traces(...)`, which reads source-aligned
spatial chunks in bounded time windows, skips chunks containing no soma, and
stores a `(time, cell)` array. It never materializes the movie or a
cell-by-voxel matrix. `np.asarray(movie)`, iteration, truth testing, unsupported NumPy
functions, and general array broadcasting fail explicitly instead of silently
loading data. See the
[supported operation table](https://peter-hogg.github.io/NeuroFlow/High_Level_API.html).

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
- **NWB-HDF5:** local files remain lazily sliceable datasets. DANDI HDF5 can use
  `backend="remfile"` or, after `uv sync --extra lindi`, `backend="lindi"`.
  `backend="auto"` currently retains remfile as the conservative default;
  fsspec remains available for direct URLs with
  `storage_options={"transport": "fsspec"}`. No
  complete-file download or eager array conversion is performed. Metadata
  discovery may still require many range requests because HDF5 metadata is not
  object-store-native. See PyNWB's official
  [streaming guide](https://pynwb.readthedocs.io/en/stable/tutorials/advanced_io/streaming.html).
- LINDI manages its own transport and cache. The current bridge does not expose
  byte-transfer counters, so benchmark records use `null` rather than inventing
  a value. The analysis, planning, persistence, and verification semantics are
  otherwise backend-independent.
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

Outputs use `mode="create"` or `"overwrite"`. With the default `resume=True`,
an existing managed output is resumed only when its deterministic workflow
provenance matches. Unmanaged paths and stale provenance are rejected unless
overwrite was explicitly requested. Append semantics are not supported.

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
labels = projection.cellpose(
    pretrained_model="cpsam",
    output="labels.zarr",
    memory_limit="4 GiB",
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
adapter boundaries, and example configuration. The suite enforces an 80%
statement-coverage floor and reports current totals on every supported CI job.

The live DANDI example is deliberately **not** part of CI. Archive availability
and network conditions should not make ordinary builds flaky; run it separately
with the command above when validating remote integration.

Design rationale and the formal API live in [`docs/`](docs/). Those documents
are useful when extending NeuroFlow; this README is the place to start when
using it.
