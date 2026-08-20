# NeuroFlow enables bounded whole-dataset analysis of archive-scale NWB data

> GigaScience Technical Note working draft. Bracketed evidence gates must be
> replaced only from clean retained benchmark records before submission.

## Abstract

Neuroscience archives increasingly contain imaging datasets whose logical size
exceeds the memory and storage available on a researcher's workstation. Remote
range readers expose slices of these datasets, and general schedulers can
execute chunked graphs, but researchers must still design traversal, memory,
persistence, restart, integrity, and provenance machinery. NeuroFlow is a
Python execution layer that translates a deliberately bounded subset of
NumPy-like analysis over NWB arrays into source-aware durable tasks. It uses
PyNWB for data semantics, remfile or LINDI for replaceable remote HDF5 access,
and Dask for chunk execution. The flagship workflow computes a bounded temporal
projection from a versioned DANDI zebrafish movie, applies actual Cellpose to
complete projection planes, and extracts mean fluorescence traces from the
original movie into a compact `(time, cell)` result. [Insert clean full-pipeline
measurements: logical size, transfer, RSS, wall time, somata, trace size.] Direct
Cellpose and NumPy comparisons assess software-path and numerical equivalence;
partition manifests, checksums, and repeated execution assess integrity and
recovery. NeuroFlow is intended to make useful whole-dataset analysis possible
on laptop-scale resources without requiring scientists to write remote chunk
loops or scheduler infrastructure.

## Findings

### Motivation

NWB standardizes the meaning and organization of neurophysiology data, while
DANDI makes versioned NWB assets accessible. LINDI and remfile address efficient
remote HDF5 access, and Dask supplies general chunked scheduling. These layers
do not by themselves decide how a particular whole-dataset analysis should be
partitioned within a memory target, how partial output should survive failure,
or how the result should be verified and attributed. NeuroFlow addresses this
orchestration layer; it does not replace those projects.

### Bounded NumPy-like execution

`NeuroArray` retains named axes, source identity, physical chunks, selection
bounds, dtype, and a canonical expression rather than converting the source to
an in-memory ndarray. Supported arithmetic, ufuncs, casts, and reductions remain
lazy until an explicit `compute` or `persist` boundary. Planning estimates input,
temporary, accumulator, output, scheduler, and cache/reserve allocations.
Unsupported expressions fail during planning instead of silently materializing
the source. Durable partitions own disjoint output regions and are finalized
with atomic manifests and checksums.

### Replaceable remote access

DANDI-hosted NWB-HDF5 assets can be opened through remfile or an optional LINDI
bridge without changing the analysis expression. Array discovery relies on the
capabilities `shape`, `dtype`, `ndim`, `chunks`, and bounded indexing rather than
requiring a concrete h5py dataset. LINDI manages its own caching; NeuroFlow does
not reimplement or expose remfile-specific cache knobs on that backend. Transfer
counters not exposed by LINDI are recorded as unknown.

### Soma trace workflow

The pinned experiment uses DANDI:000350 version 0.240822.1759, asset
`4f898ff7-6084-4e84-a449-f05811c1d951`, and
`/acquisition/NeuronOnePhotonSeries`, a `(3065, 888, 2048, 29)` movie with
323,296,788,480 logical bytes. Fifty frames are reduced to a float32
`(888, 2048, 29)` temporal median. Cellpose processes one complete y/x plane per
durable task so that no unreconciled x/y tile boundary is presented as a cell
identity.

Trace planning scans the compact label image in movie-source-aligned spatial
chunks, records every label intersecting each chunk, and omits chunks without a
soma. Execution chooses a bounded time window from the memory target, fetches an
active movie chunk once, and updates all intersecting label accumulators. It
persists float32 mean traces with timestamps and uint64 global cell identifiers.
The source remains remote and no cell-by-voxel matrix is constructed.

### Validation and workflow comparison

Software equivalence is separated from biological accuracy. The same Cellpose
model and evaluation settings are invoked directly and through NeuroFlow on the
exact persisted projection; labels must agree exactly after removing the
partition namespace. Leading fish frames are reduced independently with
plane-wise NumPy and compared with the persisted traces using recorded absolute
and relative tolerances. Synthetic tests additionally cover labels spanning
source chunks, empty-chunk skipping, interruption, resume, deliberate
corruption, detection, and selective repair.

The workflow comparison includes PyNWB + remfile + Dask, PyNWB + LINDI + Dask,
NeuroFlow + remfile, and NeuroFlow + LINDI over the same masks. Results report
correctness, time, RSS, available transfer counters, code/configuration burden,
manual scheduler/chunk settings, persistence, resume, integrity, and provenance.
The analysis does not assume NeuroFlow will be fastest; bounded recovery and
auditability may justify modest overhead.

## Methods

### Execution and provenance

Workflow identities include versioned source/asset identity, NWB path and
selection, expression or adapter identity, partition plan, resource policy,
output contract, and upstream result checksum identity. Workflow JSON uses an
allowlisted schema and never deserializes arbitrary executable Python. Result
provenance captures NeuroFlow and dependency versions, Python/platform/CPU/RAM,
Git commit and dirty state, backend, parameters, task plan, attempts, available
I/O, wall time, peak RSS, output size, checksum, and integrity status.

### Publication protocol

Publication experiments run from a clean immutable commit in a fresh process.
The runner rejects dirty-tree publication classification. Large Zarr products
are retained as run artifacts rather than committed; small JSON records,
checksums, previews, tables, and exact commands are archived. Paper tables are
generated only from `classification: publication` records by default. Current
and historical records cannot silently enter final tables.

### Scientific scope

Exact agreement with direct Cellpose demonstrates that NeuroFlow transport and
persistence do not alter the model result. It does not establish that detected
objects are biologically correct somata. [Insert expert/manual validation and
protocol outcome, or state that biological accuracy was outside the validated
scope.] Mean fluorescence traces are algorithmically validated; downstream
motion correction, neuropil correction, deconvolution, and interpretation are
outside this Technical Note.

## Availability of source code and requirements

Source code: <https://github.com/Peter-Hogg/NeuroFlow>. Python 3.10–3.13 and uv
are supported by the locked development workflow. [Insert release tag, OSI
license, software DOI, archive URL, and public CI links after completion.]

## Availability of supporting data

The source dataset is DANDI:000350 version 0.240822.1759,
<https://doi.org/10.48324/dandi.000350/0.240822.1759>. Generated benchmark JSON,
workflow records, and compact evidence are indexed under `publication/`.
[Insert immutable artifact/archive identifiers for labels, traces, logs, and
benchmark outputs.]

## Declarations

[Complete competing interests, funding, author contributions, ethics/data-use
statement, and acknowledgements before submission.]
