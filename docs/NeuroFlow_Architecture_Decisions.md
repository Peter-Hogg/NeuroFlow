# NeuroFlow Architecture Decision Record
**Document:** ADR-0001 through ADR-0011  
**Status:** Proposed  
**Audience:** maintainers and coding agents  
**Rule:** Agents may implement within these decisions but must not silently reverse them.

---

## ADR-0001 — Product boundary

### Decision

NeuroFlow is an execution, interoperability, and persistence framework for applying existing Python libraries to archive-scale NWB data.

It is not a scientific-analysis package.

### Rationale

Scientific algorithms already exist in libraries such as Cellpose, Pynapple, scikit-image, PyTorch, Suite2p, and CaImAn. Reimplementing those algorithms would expand validation and maintenance obligations while weakening the central contribution.

The framework should solve the missing systems problem:

```text
NWB/DANDI source
→ semantic selection
→ partition plan
→ external library
→ durable partitioned output
→ lazy downstream result
```

### Consequences

- Core contains no bespoke segmentation, registration, or dF/F algorithm.
- Example notebooks may demonstrate scientific functions.
- Integrations remain optional.
- Scientific correctness remains the responsibility of the called library and user-defined workflow.

---

## ADR-0002 — Dask is the initial execution backend

### Decision

Use Dask for lazy arrays, arbitrary delayed tasks, distributed scheduling, diagnostics, and resource-aware execution.

### Rationale

The target workflows combine:

- chunked dense arrays
- irregular outputs
- laptop-to-cluster execution
- CPU and GPU tasks
- large remote inputs
- workflows that must remain lazy
- integration with Zarr and Parquet

Dask provides Array for shape-aware computation and Delayed/Futures for heterogeneous work. It avoids building a custom scheduler.

### Constraints

- Public scientific APIs must not expose unnecessary Dask internals.
- Dask is an implementation backend, not the project's scientific identity.
- Backend-neutral concepts such as `PartitionPlan`, `Adapter`, and `Result` must remain separable from the Dask graph implementation.

### Revisit when

- a demonstrated workload cannot be expressed reliably in Dask
- scheduler overhead dominates realistic tasks
- another backend materially improves interoperability without fragmenting the API

---

## ADR-0003 — NWB-Zarr on DANDI is the first-class source

### Decision

Optimize version 0.1 for immutable NWB-Zarr assets hosted by DANDI.

### Rationale

The flagship use case is whole-brain zebrafish light-sheet imaging. Zarr provides chunked object-store-native access and works naturally with Dask. Supporting every possible NWB backend at the start would dilute the architecture.

### Consequences

- Local NWB-Zarr is also supported.
- NWB-HDF5 may be supported experimentally but is not allowed to dictate the initial API.
- The source layer must expose backend capability information.
- Documentation must not imply equal performance across formats.

---

## ADR-0004 — Separate storage chunks from processing partitions

### Decision

Native Zarr chunks and scientific processing partitions are separate concepts.

### Rationale

Storage chunks determine physical reads and compression boundaries. Processing partitions determine the unit presented to an external algorithm. A segmentation tile may span many storage chunks; a temporal operation may group many native chunks and require overlap.

### Consequences

Every execution plan records:

- native chunks
- processing partition shape
- overlap/halo
- expected read amplification
- estimated memory per task

The planner warns rather than pretending arbitrary chunk choices are free.

---

## ADR-0005 — Adapter contract instead of algorithm wrappers in core

### Decision

External libraries integrate through a declared adapter contract.

### Rationale

"Arbitrary library support" is only honest when partition, overlap, resource, and output semantics are explicit. A generic function cannot automatically be distributed safely.

### Minimum adapter declarations

- accepted NWB inputs
- partitionable axes
- overlap requirements
- preparation/conversion step
- execution function
- output types
- resource requirements
- persistence method
- merge semantics, when required

### Consequences

- A lightweight `FunctionAdapter` supports simple NumPy-like functions.
- Cellpose and Pynapple integrations validate two distinct execution classes.
- Adapter APIs must remain small enough that a useful integration can be written without adopting a workflow DSL.

---

## ADR-0006 — Durable outputs are first-class

### Decision

Large intermediate and final outputs are written incrementally to durable stores and reopened lazily.

### Rationale

A framework for huge inputs is incomplete if segmentation masks, cell tables, or traces are accumulated in scheduler memory. Million-cell outputs require partitioned persistence.

### Canonical storage

- Zarr: dense arrays, label images, traces
- Parquet: object tables, quality metrics, centroids, bounding boxes
- indexed ragged/sparse representation: masks or variable-length per-object data
- memmap: temporary node-local compatibility only

### Consequences

Tasks return small manifests, not large arrays or tables. Downstream stages consume lazy result handles backed by persisted data.

---

## ADR-0007 — Memmap is a compatibility mechanism, not a project format

### Decision

Support temporary memmaps only for libraries requiring filesystem-backed arrays.

### Rationale

Memmaps work well on one machine but are poor durable distributed outputs:

- awkward concurrent writes
- no compression
- weak object-store compatibility
- no native chunk metadata
- monolithic file assumptions

### Consequences

Adapters may localize one bounded partition to node-local scratch:

```text
remote Zarr slice
→ temporary memmap
→ external legacy library
→ Zarr/Parquet output
→ delete temporary memmap
```

The public result format must not depend on the continued existence of scratch files.

---

## ADR-0008 — Resumability through deterministic partition manifests

### Decision

Each partition writes an atomic completion manifest keyed by deterministic source and workflow identity.

### Rationale

Large jobs will fail. Recomputing successful tiles or sessions is unacceptable. Dask's in-memory task state is not a durable scientific checkpoint.

### Manifest identity includes

- immutable source version
- asset identity
- selected NWB object
- adapter version
- parameters
- partition coordinates
- output schema version

### Consequences

- Resume validates manifests before skipping work.
- Parameter or source changes invalidate incompatible partitions.
- Partial writes are never treated as success.
- Outputs support inspection of complete, failed, and missing partitions.

---

## ADR-0009 — Provenance is mandatory, not optional decoration

### Decision

Every persisted result includes machine-readable provenance.

### Rationale

Remote archive reanalysis must be reproducible and traceable to immutable source assets and exact analysis conditions. This is central to scientific utility and auditability.

### Required provenance

- Dandiset and version
- asset IDs/checksums
- NWB paths/types
- package and adapter versions
- parameters and random seeds
- partition and chunk plans
- scheduler/environment metadata
- output schema and locations
- execution status

### Consequences

A write without provenance is considered incomplete.

---

## ADR-0010 — No hidden execution

### Decision

Selection, planning, and graph construction remain lazy. Execution requires an explicit user action.

### Rationale

Hidden reads or `.compute()` calls undermine user control, reproducibility, memory guarantees, and trust.

### Enforcement

Tests must use instrumented fake stores to detect:

- unexpected reads
- oversized slices
- full-array materialization
- eager execution during import or planning

Agents must not insert `.compute()`, `.persist()`, or eager conversions inside core library functions without an explicit API contract and test.

---

## ADR-0011 — API growth follows validated vertical slices

### Decision

Build a narrow end-to-end path before generalizing.

### First vertical slice

1. resolve one DANDI NWB-Zarr asset
2. select one image series
3. partition it temporally
4. run one plain function
5. write a Zarr result incrementally
6. reopen the result lazily
7. resume an interrupted run
8. record provenance

### Second vertical slice

Spatially tiled segmentation with variable-size outputs and direct persistence.

### Third vertical slice

Session-level multimodal analysis using Pynapple-style objects.

### Consequences

Do not initially build:

- a plugin marketplace
- a workflow language
- automatic adapter generation
- a GUI
- multiple execution backends
- broad cloud-provider abstractions
- a generalized data catalog

An abstraction may be promoted to public API after two vertical slices demonstrate its reuse.

---

# Agent operating rules

Coding agents working on NeuroFlow must:

1. read this ADR and the API specification before modifying code
2. state which ADRs a proposed change touches
3. avoid adding new abstractions unless a current issue requires them
4. provide tests for laziness, bounded reads, and resumability
5. keep optional dependencies isolated
6. avoid broad refactors while implementing a single issue
7. never claim scientific equivalence without a defined validation test
8. report uncertainties rather than silently inventing NWB semantics
9. keep pull requests small and independently reviewable
10. update the ADR only through an explicit architecture-change pull request

# Architecture change process

A proposed reversal must include:

- current decision
- observed failure or limitation
- at least two alternatives
- impact on public API
- migration plan
- new acceptance tests

Until accepted, the existing decision remains binding.
