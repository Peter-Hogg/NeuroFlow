# Software release checklist

## Repository and software

- [ ] Maintainer selects an OSI-approved license and adds `LICENSE` plus matching
  `pyproject.toml`, `CITATION.cff`, and `.zenodo.json` metadata.
- [x] Author, project URL, classifier, citation, changelog, contribution, conduct,
  and security metadata are present.
- [x] Network-free tests compare NumPy, NWB-Zarr, and NWB-HDF5 results.
- [x] Trace outputs have provenance, coordinates, manifests, checksums, repair,
  resume, and `open_result()` support.
- [x] Friendly segmentation prevents accidental use of unreconciled y/x
  tiles.
- [x] Workflow concurrency and estimated memory limits are enforced.
- [x] Wheel and source distribution build and the wheel installs cleanly.
- [ ] GitHub matrix workflows pass on the public repository.
- [ ] Docker image builds in CI or another environment with a working daemon.

## Validation evidence

- [x] Deterministic reference data and backend-equivalence tests are included.
- [x] Repeated local benchmark harness compares NumPy, direct Dask, and NeuroFlow.
- [x] Run and retain five deterministic local benchmark repetitions.
- [x] Run and retain one fresh archive-scale DANDI case-study record.
- [ ] Validate real Cellpose output against manual or expert-reviewed annotations.
- [x] Validate extracted traces against an independent direct NumPy reference.
- [x] Local benchmark records machine, versions, command, seed, and tolerances.
- [x] Archive-scale record includes available network context and transfer statistics;
  unavailable peak RSS and detailed network fields are explicitly identified.

## Release and archival actions

- [ ] Freeze the public API and create signed tag `v0.1.0`.
- [ ] Create a GitHub release and archive the tagged software in Zenodo or
  Software Heritage.
- [ ] Add any assigned software DOI to `CITATION.cff` and release metadata.
- [x] Record the DANDI:000350 version, DOI, asset, object, and primary article in
  the reproducibility guide.
