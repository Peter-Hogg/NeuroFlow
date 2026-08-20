# Concepts

## Selection is semantic and lazy

An NWB selection records the object path, named axes, dtype, native chunks, and
absolute slice bounds. Equal-shaped slices at different positions are distinct
scientific inputs and therefore have different workflow identities.

## Processing chunks and storage chunks differ

Native chunks describe how the archive stores bytes. Processing partitions
describe the bounded arrays passed to an analysis function. Output chunks
describe the persisted Zarr layout. NeuroFlow reports all three rather than
pretending Dask can change an HDF5 file's physical chunks.

## A partition is the unit of recovery

Each completed partition has an atomic manifest and checksum. Resume validates
existing bytes before skipping work. Verification reads one owned partition at
a time and rejects manifest outputs outside the declared result root. Component
names, array-versus-table storage kinds, and canonical table paths are bound to
the provenance schema, so a different file cannot stand in for a declared
array. Current version-2 manifests also record every partition output size;
legacy version-1 manifests remain readable when sizes were not recorded.
`open_array()` verifies a structurally complete result and its checksums by
default, then binds downstream provenance to a canonical digest of the upstream
partition descriptors and checksums. Resume also checks the actual Zarr chunk
layout against the planned layout before it trusts existing partition data.

Trace extraction counts label IDs one label-storage chunk at a time with one
Dask worker. It budgets the chunk-local unique workspace, the retained cell-ID
map, and each movie window before allowing those allocations to grow.

## NumPy expressions are plans, not arrays in RAM

`NeuroArray` records a canonical expression tree. Scalar arithmetic, supported
ufuncs, casts, and compatible reductions are fused into one bounded operation
per source partition. The tree—not a Python function address—is part of the
workflow identity. Equivalent named and positional axes therefore resume the
same work, while changed constants, operations, selections, or dtypes do not.

NumPy cannot know that a remote NWB object is dangerous to materialize.
NeuroFlow therefore refuses implicit conversion, iteration, and truth testing.
`.compute()` is explicit and has a conservative 1 GiB default estimate;
`.persist()` is the preferred boundary for larger results.

## Global scalars are explicit stages

A global scalar `sum`, `mean`, `min`, or `max` used downstream becomes one
deduplicated stage. NeuroFlow partitions the source on native chunk boundaries,
stores checksummed partials, combines them deterministically, and substitutes
the verified scalar into downstream bounded tasks. An interrupted run resumes
valid partials. Unsupported non-scalar broadcasts and global order statistics
fail during planning rather than trigger repeated or unbounded source reads.

## A workflow specification is portable provenance

`WorkflowSpec` is versioned, deterministic JSON for a safe subset of workflows.
It records source identity and selection, canonical allowlisted expression,
partition and memory policy, create-only output, and software metadata. It is
validated rather than executed as code. `workflow.plan()` resolves the source
and reports boundedness and explicit estimated/unknown cost fields before any
numerical payload is read; post-run provenance records measured timing, task,
resume, output, RSS, stage, integrity, Git, and environment information.

## Memory limits are estimates, not process isolation

NeuroFlow limits source partitions, declared adapter memory, concurrency, and
known setup operations. It cannot see allocations hidden inside arbitrary
native libraries. Third-party adapters should declare conservative resources,
and empirical peak memory should still be measured for reported workloads.
