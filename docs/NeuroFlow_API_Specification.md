# NeuroFlow API Specification
**Status:** Draft 0.1  
**Audience:** coding agents, maintainers, reviewers, and early contributors  
**Purpose:** define the smallest stable public API for an out-of-core execution and interoperability framework for archive-scale NWB analysis.

---

## 1. Product definition

NeuroFlow allows existing Python analysis libraries to operate on very large NWB datasets stored locally or on DANDI without requiring full dataset downloads or full in-memory materialization.

NeuroFlow owns:

- NWB and DANDI object discovery
- lazy data access
- partition planning
- Dask graph construction
- resource declarations
- durable intermediate and final outputs
- resumability
- provenance
- lazy result loading

External libraries own:

- segmentation
- registration
- event detection
- tuning calculations
- decoding
- model inference
- scientific interpretation

The core package must not implement domain-specific scientific algorithms.

---

## 2. Non-negotiable invariants

1. Library code must not call `.compute()` implicitly.
2. Opening, inspecting, selecting, and planning must remain lazy.
3. No operation may require full-dataset materialization unless explicitly declared by an adapter.
4. Large task results must be written directly to durable storage rather than returned through the Dask scheduler.
5. External libraries must be optional dependencies.
6. Every persisted result must contain source and execution provenance.
7. Interrupted jobs must be resumable at deterministic partition boundaries.
8. APIs are added only after at least two concrete workflows require the abstraction.
9. Storage layout and compute partitioning must be treated as separate concepts.
10. The first implementation target is NWB-Zarr on DANDI; other backends must not distort the initial design.

---

## 3. Package layout

```text
neuroflow/
    __init__.py
    source/
        base.py
        dandi.py
        local.py
    selection/
        query.py
        nwb_types.py
    partition/
        base.py
        time.py
        spatial.py
        session.py
    adapters/
        base.py
        numpy.py
    execution/
        graph.py
        runner.py
        resources.py
    storage/
        base.py
        zarr.py
        parquet.py
        manifest.py
    results/
        base.py
        array.py
        table.py
        segmentation.py
        traces.py
    provenance/
        model.py
        hashing.py
    diagnostics/
        plan.py
        estimates.py
```

Integrations such as Cellpose and Pynapple may initially live in separate optional modules or packages:

```text
neuroflow_cellpose/
neuroflow_pynapple/
```

They should not become required dependencies of `neuroflow-core`.

---

## 4. Core source API

### 4.1 `open_source`

```python
def open_source(
    source: str | Path | "SourceSpec",
    *,
    version: str | None = None,
    storage_options: dict[str, object] | None = None,
) -> "NWBSource":
    ...
```

Accepted source examples:

```python
open_source("DANDI:000123")
open_source("DANDI:000123@0.240101.1234")
open_source("/data/session.nwb.zarr")
open_source("s3://bucket/session.nwb.zarr")
```

Opening a source must:

- resolve metadata only
- avoid reading numerical datasets
- preserve immutable Dandiset version information when available
- expose assets and NWB object metadata
- avoid silently selecting an asset when multiple assets match

### 4.2 `NWBSource`

```python
class NWBSource(Protocol):
    @property
    def identity(self) -> "SourceIdentity": ...

    def assets(self) -> "AssetCollection": ...

    def select(
        self,
        query: "NWBQuery",
    ) -> "Selection": ...

    def inspect(self) -> "SourceSummary": ...
```

`inspect()` may retrieve metadata, shapes, dtypes, native chunks, compression metadata, and timestamps. It must not read full numerical arrays.

---

## 5. Semantic selection API

Users should select data by NWB meaning whenever possible, not by raw Zarr paths.

```python
movie = source.select(
    NWBQuery(
        neurodata_type="TwoPhotonSeries",
        name="whole_brain_movie",
    )
)
```

### 5.1 `NWBQuery`

```python
@dataclass(frozen=True)
class NWBQuery:
    neurodata_type: str | type | None = None
    name: str | None = None
    path: str | None = None
    asset: str | None = None
    subject: str | None = None
    session_id: str | None = None
    where: Mapping[str, object] | None = None
```

Rules:

- Exact path selection is supported as an escape hatch.
- Ambiguous selection raises `AmbiguousSelectionError`.
- Missing selection raises `ObjectNotFoundError`.
- The selected object retains NWB metadata, dimensional meaning, timestamps, and source identity.

### 5.2 `Selection`

```python
class Selection:
    metadata: "SelectionMetadata"

    def as_dask_array(
        self,
        *,
        chunks: "ChunkSpec | Literal['native', 'auto']" = "auto",
    ) -> "dask.array.Array":
        ...

    def plan(
        self,
        partition: "PartitionPlan",
    ) -> "ExecutionPlan":
        ...
```

---

## 6. Partition API

A partition defines the scientific unit of execution. It is distinct from native Zarr chunks.

### 6.1 Base protocol

```python
class PartitionPlan(Protocol):
    def build(
        self,
        selection: Selection,
    ) -> Sequence["Partition"]:
        ...

    def validate(
        self,
        selection: Selection,
    ) -> "ValidationReport":
        ...
```

### 6.2 Time windows

```python
TimeWindowPlan(
    size="60 s",
    overlap="5 s",
    align_to="timestamps",
)
```

or

```python
TimeWindowPlan(
    size=1000,
    overlap=100,
    units="samples",
)
```

### 6.3 Spatial tiles

```python
SpatialTilePlan(
    tile_shape=(64, 512, 512),
    halo=(8, 64, 64),
    axes=("z", "y", "x"),
)
```

### 6.4 Asset/session mapping

```python
AssetPlan(filter={"neurodata_type": "TwoPhotonSeries"})
SessionPlan(group_by=("subject_id", "session_id"))
```

### 6.5 Partition identity

Every partition must have a deterministic identity derived from:

- immutable source identity
- selected NWB object
- partition plan parameters
- partition coordinates
- adapter version
- relevant analysis parameters

This identity is used for resumability and provenance.

---

## 7. Adapter API

The adapter describes how an external analysis library consumes partitions and produces outputs.

### 7.1 Minimal protocol

```python
class AnalysisAdapter(Protocol):
    name: str
    version: str

    def requirements(self) -> "AdapterRequirements":
        ...

    def prepare(
        self,
        partition: "LoadedPartition",
        context: "TaskContext",
    ) -> object:
        ...

    def run(
        self,
        prepared: object,
        context: "TaskContext",
    ) -> "TaskOutput":
        ...

    def persist(
        self,
        output: "TaskOutput",
        writer: "PartitionWriter",
        context: "TaskContext",
    ) -> "PartitionManifest":
        ...
```

### 7.2 Optional merge protocol

```python
class MergeableAdapter(AnalysisAdapter, Protocol):
    def boundary_summary(
        self,
        manifest: "PartitionManifest",
    ) -> "BoundarySummary":
        ...

    def merge(
        self,
        neighbors: Sequence["BoundarySummary"],
        writer: "ResultWriter",
        context: "MergeContext",
    ) -> "MergeManifest":
        ...
```

### 7.3 Requirements declaration

```python
@dataclass(frozen=True)
class AdapterRequirements:
    input_kinds: tuple[str, ...]
    splittable_axes: tuple[str, ...]
    requires_overlap: Mapping[str, int | str]
    output_kinds: tuple[str, ...]
    resources: "ResourceSpec"
    deterministic: bool
    requires_local_path: bool = False
```

### 7.4 Plain-function adapter

The simplest supported user experience:

```python
adapter = FunctionAdapter(
    function=my_numpy_function,
    input_kind="array",
    output=ArrayOutput(dtype="float32"),
)
```

The framework must not claim that arbitrary functions are distributable without declared partition and merge semantics.

---

## 8. Execution API

### 8.1 `run`

```python
result = neuroflow.run(
    source=source,
    selection=movie,
    adapter=adapter,
    partition=TimeWindowPlan(size=1000, overlap=100),
    output=ZarrOutput("s3://bucket/run-001.zarr"),
    scheduler="distributed",
    resume=True,
)
```

### 8.2 Required behavior

`run()` must:

1. resolve and validate the source
2. resolve semantic inputs
3. validate adapter and partition compatibility
4. estimate memory, task count, overlap overhead, and storage writes
5. construct a lazy Dask graph
6. avoid moving large outputs through the scheduler
7. persist task outputs atomically
8. write partition manifests
9. resume completed partitions when requested
10. create a final result manifest
11. return a lazy result handle

Execution starts only after an explicit action such as:

```python
handle = run(..., execute=False)
handle.execute()
```

or an explicitly documented default. There must be no hidden computation during graph construction.

### 8.3 Resources

```python
ResourceSpec(
    cpu=4,
    memory="16 GiB",
    gpu=1,
    local_scratch="50 GiB",
)
```

Adapters may declare worker initialization requirements for expensive models.

---

## 9. Output API

### 9.1 Output specifications

```python
ZarrOutput(
    uri="s3://bucket/run-001.zarr",
    mode="create",
    compressor="default",
)

ParquetOutput(
    uri="s3://bucket/run-001/objects/",
    partition_on=("asset_id", "tile_id"),
)
```

### 9.2 Atomicity

A partition is complete only when:

- all expected arrays/tables are successfully written
- checksums or validation metadata are recorded
- a final success manifest is atomically committed

Partial outputs must not be mistaken for completed partitions.

### 9.3 Canonical result composition

A segmentation result may contain:

```python
SegmentationResult(
    labels=<lazy Dask array backed by Zarr>,
    objects=<lazy Dask DataFrame backed by Parquet>,
    masks=<LazyMaskStore>,
    provenance=<ProvenanceRecord>,
)
```

A trace result may contain:

```python
TraceResult(
    traces=<lazy Dask array backed by Zarr>,
    cells=<lazy table>,
    timestamps=<lazy or compact coordinate>,
    provenance=<ProvenanceRecord>,
)
```

### 9.4 Storage choices

- dense multidimensional arrays: Zarr
- large tables: partitioned Parquet
- sparse or ragged masks: compressed indexed representation
- local compatibility scratch: temporary memmap
- durable project format: never raw memmap alone

---

## 10. Lazy result API

```python
result = neuroflow.open_result("s3://bucket/run-001/")
```

Result handles expose:

```python
result.arrays
result.tables
result.provenance
result.status
result.failed_partitions
result.resume()
```

Access to a subset must not load unrelated output:

```python
result.labels[100:150, 1000:2000, 1000:2000]
result.objects.query("quality_score > 0.8")
result.traces[cell_ids, t0:t1]
```

---

## 11. Provenance schema

Every result records:

- NeuroFlow version
- adapter name and version
- external-library versions
- source Dandiset ID and immutable version
- asset IDs and checksums
- NWB object paths and neurodata types
- partition specification
- native storage chunks
- processing chunks
- overlap
- parameters
- environment/container identity
- scheduler configuration
- execution start/end times
- completed and failed partitions
- random seeds
- output locations and schemas

Provenance must be machine-readable JSON and optionally mirrored into NWB-compatible metadata.

---

## 12. Diagnostics

Before execution:

```python
plan = neuroflow.plan(...)
print(plan.summary())
```

The report should include:

- source size
- selected shape and dtype
- native chunking
- processing partitions
- expected number of tasks
- estimated uncompressed memory per task
- overlap amplification
- expected output size
- likely pathological access patterns
- declared CPU/GPU needs

Warnings should be actionable.

---

## 13. Failure model

Required exceptions include:

```text
SourceResolutionError
UnsupportedBackendError
AmbiguousSelectionError
ObjectNotFoundError
PartitionValidationError
AdapterCompatibilityError
OutputConflictError
IncompletePartitionError
ProvenanceMismatchError
```

Resume must reject stale outputs when source identity, adapter version, parameters, or partition semantics differ unless explicitly overridden.

---

## 14. Initial supported workflows

### Workflow A: remote dF/F or summary computation

- input: NWB-Zarr image series from DANDI
- partition: temporal blocks with optional overlap
- algorithm: external NumPy/scipy function
- output: Zarr array
- result: lazy Dask array

### Workflow B: large-volume segmentation

- input: summary image or volume
- partition: spatial tiles with halos
- algorithm: Cellpose adapter
- output:
  - dense labels in Zarr
  - object table in Parquet
  - optional sparse masks
- merge: boundary-object reconciliation
- result: lazy segmentation handle

### Workflow C: multimodal session analysis

- inputs: units, position, intervals
- partition: assets or sessions
- algorithm: Pynapple adapter
- output: tables and time-series arrays
- result: lazy table/array collection

---

## 15. Explicitly out of scope for version 0.1

- implementing segmentation algorithms
- implementing motion correction
- replacing Suite2p, CaImAn, Cellpose, or Pynapple
- transparent distribution of arbitrary stateful functions
- automatic scientific validation
- automatic optimal chunking for every workload
- mutable editing of source DANDI assets
- universal NWB-HDF5 optimization
- a graphical workflow builder
- Kubernetes-specific cluster management
- a custom scheduler

---

## 16. Acceptance tests for the first release

1. Opening a real DANDI NWB-Zarr asset performs no bulk data read.
2. Constructing a workflow performs no `.compute()`.
3. A fake store records only requested slices.
4. Memory usage remains bounded as input size increases.
5. A failed partition can be rerun without repeating completed partitions.
6. A result can be reopened lazily in a fresh Python process.
7. Provenance uniquely identifies source, adapter, parameters, and output.
8. Cellpose or a stand-in variable-output segmentation adapter writes directly to persistent storage.
9. One single-machine and one distributed-cluster execution use the same analysis definition (design goal; distributed execution is not currently validated).
10. Optional integrations are absent without breaking core imports.

---

## 17. Supported NumPy expression layer

`NeuroArray` implements a finite NumPy protocol surface rather than implicit
conversion. Scalar arithmetic, comparisons, selected ufuncs, casts, contiguous
rank-preserving slices, and the reductions `sum`, `mean`, `min`, `max`,
`median`, scalar `percentile`, and scalar `quantile` build a canonical lazy
expression.

The expression is evaluated only by explicit `.compute()` or `.persist()`.
Implicit `np.asarray`, iteration, truth testing, `out=`, raw ndarray operands,
general broadcasting, fancy indexing, and unsupported NumPy functions must
raise before numerical I/O.

Durable lowering is restricted to expressions whose output partition can be
computed from one bounded source partition. All reduced axes are read whole;
retained axes may be tiled. Non-local broadcast branches such as
`x / x.max()` require a future staged reduction and are rejected rather than
being evaluated with tile-local or repeated-global semantics.
