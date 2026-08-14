# Changelog

All notable changes to NeuroFlow are documented here. The project follows
[Semantic Versioning](https://semver.org/) and keeps an
[Keep a Changelog](https://keepachangelog.com/) style history.

## [Unreleased]

### Added

- Software-release metadata and reproducibility assets.
- Lazy NumPy protocol support for scalar arithmetic, selected ufuncs, casts,
  named-axis reductions, scalar percentiles, and explicit compute/persist
  boundaries.
- Conservative expression memory estimates and refusal of implicit array
  conversion, iteration, unsupported broadcasting, and mutable NumPy outputs.
- A 1 GiB default estimate for explicit in-memory compute and a 2 GiB per-task
  default for durable expression persistence.

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

## [0.1.0] - 2026-08-13

### Added

- Lazy local and DANDI-hosted NWB-Zarr and NWB-HDF5 access.
- Named-axis selections and NumPy-like temporal reductions.
- Bounded Dask execution with durable Zarr and Parquet outputs.
- Partition provenance, checksums, verification, repair, and resume.
- Tiled segmentation contracts and optional Cellpose integration.
- Optional Pynapple integration.
- Composable persisted arrays and bounded fluorescence trace extraction.

[Unreleased]: https://github.com/Peter-Hogg/NeuroFlow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Peter-Hogg/NeuroFlow/releases/tag/v0.1.0
