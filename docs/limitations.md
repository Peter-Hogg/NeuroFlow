# Limits and scientific responsibilities

NeuroFlow provides execution and storage guarantees; it does not validate a
biological interpretation.

- Remote HDF5 requires a server supporting byte-range requests and uses threads.
- Logical x/y crops may still transfer complete physical HDF5 chunks.
- Segmentation across NeuroFlow x/y tiles is rejected by the friendly API unless
  explicitly marked unreconciled.
- Cellpose model choice, weights, thresholds, and biological accuracy require
  dataset-specific validation against expert-reviewed annotations.
- Candidate detectors in examples are quality-control tools, not classifiers.
- Memory budgets are conservative estimates, not operating-system hard limits.
- Trace extraction bounds label discovery and movie windows, but a dataset with
  pathologically many distinct nonzero label IDs is rejected rather than
  building an unbounded in-memory cell map.
- Checksum verification caps each partition output at 2 GiB. Use smaller
  partitions for larger results; verification never streams an unbounded table
  object merely because a manifest names it.
- Network transfer metrics based on `Content-Length` describe observed response
  payloads; they are not a billing statement from DANDI.
- A reopened result can be verified and read, but resuming execution requires
  reconstructing the original adapter function.
- NumPy compatibility is intentionally finite. Raw ndarray operands, general
  array broadcasting, transpose/reshape, fancy indexing, `out=`, mutable
  operations, and unlisted NumPy functions are rejected.
- Global scalar `sum`, `mean`, `min`, and `max` can feed a downstream expression
  through a persisted bounded reduction stage, so `movie / movie.max()` is
  supported. The stage reads native-aligned bounded partitions once and its
  scalar is reused by all downstream partitions. Non-scalar broadcasts such as
  `movie - movie.mean("time")`, global median/percentile, variance, standard
  deviation, and arbitrary multi-input DAGs remain unsupported and fail while
  the expression is planned.
- Exact median and percentile require every reduced axis in one processing
  partition. They remain bounded when retained axes can be tiled; a memory
  limit rejects unsafe global cases before numerical I/O.
- DANDI selections bind provenance to the versioned asset identity. Local HDF5
  sources also include file size and modification time, while local Zarr uses a
  lightweight metadata marker. Those markers are not content hashes of every
  numerical chunk. For a mutable direct source, pass an immutable application
  version such as `neuroflow.load(path, version="acquisition-2026-08-13")` and
  change it whenever the source changes. Generic remote URLs likewise need a
  stable `version=` or immutable/versioned URL for source-change-safe resume.

For reproducible reporting, record exact source versions, asset identifiers,
selections, commands, software versions, random seeds, numerical tolerances,
hardware, network context, validation protocols, and excluded data.
