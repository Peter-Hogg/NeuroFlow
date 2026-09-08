# Changelog

All notable changes to NeuroFlow are documented here. The project follows
[Semantic Versioning](https://semver.org/) and keeps an
[Keep a Changelog](https://keepachangelog.com/) style history.

## [0.1.0] - 2026-09-08

First public release, accompanying the NeuroFlow publication.

### Added

- Lazy local and DANDI-hosted NWB-Zarr and NWB-HDF5 access.
- Named-axis selections and NumPy-like temporal reductions.
- Lazy NumPy protocol support for scalar arithmetic, selected ufuncs, casts,
  named-axis reductions, scalar percentiles, and explicit compute/persist
  boundaries.
- Conservative expression memory estimates and refusal of implicit array
  conversion, iteration, unsupported broadcasting, and mutable NumPy outputs.
- A 1 GiB default estimate for explicit in-memory compute and a 2 GiB per-task
  default for durable expression persistence.
- Bounded Dask execution with durable Zarr and Parquet outputs.
- Partition provenance, checksums, verification, repair, and resume.
- A first-class optional LINDI backend for DANDI NWB-HDF5 sources.
- Source-chunk-oriented trace planning with automatic time windows, empty-chunk
  skipping, durable resume, integrity, and `(time, cell)` coordinates.
- Tiled segmentation contracts and optional Cellpose integration, including a
  plane-safe `projection.cellpose(...)` convenience path, an opt-in real
  Cellpose equivalence test, and the end-to-end fish publication harness.
- `neuroflow_cellpose.clear_model_cache()` to release cached networks (host
  and GPU memory) between pipeline phases.
- Optional Pynapple integration, with an opt-in equivalence test against the
  real pynapple package.
- Composable persisted arrays and bounded fluorescence trace extraction.
- Manual PyNWB + remfile/LINDI + Dask trace baselines over identical masks.
- The region-traces example: full-length traces from one region at ~13% of the
  movie's bytes, with the transfer predicted by the plan.
- Software-release metadata and reproducibility assets; `py.typed` markers so
  downstream type checkers see the annotations; top-level re-exports of the
  common workflow nouns (`NWBQuery`, partition plans, outputs,
  `FunctionAdapter`); a Windows CI leg.

### Changed

- Workflow identity now includes the selected DANDI asset, complete partition
  descriptors, and canonical expressions. Downstream persisted-array identity
  is bound to verified upstream partition checksums, sizes, and schema.
- Parallel array partitions must own disjoint, chunk-aligned Zarr regions.
- Persisted arrays require complete results and checksum verification when
  reopened by default; resume also validates the actual Zarr chunk layout.
- Output paths may not overlap active inputs, and unsupported append mode is
  rejected instead of behaving like create/resume.
- Remote fish and dual-channel examples now express projections with
  `np.median(...)` directly and cast their compact visualization outputs to
  `float32` explicitly.
- The legacy `median(..., output=...)` convenience route now follows NumPy dtype
  semantics and stores its array as `result`, matching `.median(...).persist()`.
  Existing code that relied on the former implicit `float32` cast or `median`
  component should add `.astype("float32")` and select the new component.
- New manifests record partition byte sizes, and checksum verification refuses
  any single output above its bounded verification limit.
- Manifest verification binds every component to its declared array/table kind
  and canonical location; finalization refuses missing, failed, mismatched, or
  uncommitted manifests.
- Trace extraction discovers label IDs sequentially by storage chunk, budgets
  unique-ID and per-window workspaces, and returns a content-identified array.
- Provenance preserves the original execution record and appends each resume
  attempt with its own policy, environment, timestamps, status, and error.
- Remote-array discovery now uses a minimal sliceable-array capability check
  instead of requiring a literal `h5py.Dataset`.
- Staged global reductions budget against the memory remaining after process
  overhead, matching every other planner path.
- Transports without response hooks (fsspec, LINDI) report transfer as
  unmeasured rather than recording zero bytes as a measurement.
- `open_source` parses `DANDI:` identifiers with the same grammar as
  `open_dandi` and errors clearly instead of falling through to a filesystem
  lookup; the NWB-HDF5 open error distinguishes local from remote failures.
- The `cellpose` extra is bounded to `>=4.1,<5`: the adapter passes 4.x
  evaluation keywords.

### Removed

- Unused hard dependencies `xarray` and `rich`.
- The never-implemented adapter merge contract (`MergeableAdapter`,
  `BoundarySummary`, `MergeContext`, `MergeManifest`), the unused
  `ProvenanceRecord`, `TraceResult`, `SegmentationResult`, and
  `ARRAY_NEURODATA_TYPES` placeholders, `AdapterRequirements.requires_local_path`,
  and the ignored `AssetPlan.filter` / `SessionPlan.group_by` parameters.
  Workflow specs carrying the removed partition parameters are refused with an
  explicit error.

### Fixed

- `import neuroflow` no longer fails on Windows: the Unix-only `resource`
  module is optional and peak RSS is recorded as null where unmeasurable.
- CPU-selected Cellpose device provenance no longer reports a CUDA toolkit
  version.

[0.1.0]: https://github.com/Peter-Hogg/NeuroFlow/releases/tag/v0.1.0
