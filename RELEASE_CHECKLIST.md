# Publication release checklist

## Repository and software

- [ ] Maintainer selects an OSI-approved license and adds `LICENSE` plus matching
  `pyproject.toml`, `CITATION.cff`, and `.zenodo.json` metadata.
- [x] Author, project URL, classifier, citation, changelog, contribution, conduct,
  and security metadata are present.
- [x] Network-free tests compare NumPy, NWB-Zarr, and NWB-HDF5 results.
- [x] Trace outputs have provenance, coordinates, manifests, checksums, repair,
  resume, and `open_result()` support.
- [x] Friendly segmentation prevents accidental publication of unreconciled y/x
  tiles.
- [x] Workflow concurrency and estimated memory limits are enforced.
- [x] Wheel and source distribution build and the wheel installs cleanly.
- [ ] GitHub matrix workflows pass on the public repository.
- [ ] Docker image builds in CI or another environment with a working daemon.

## Scientific evidence

- [x] Deterministic reference data and backend-equivalence tests are included.
- [x] Repeated local benchmark harness compares NumPy, direct Dask, and NeuroFlow.
- [x] Run and retain five deterministic local benchmark repetitions.
- [ ] Run one fresh archive-scale DANDI case study and retain resource logs.
- [ ] Validate real Cellpose output against manual or expert-reviewed annotations.
- [x] Validate extracted traces against an independent direct NumPy reference.
- [x] Local benchmark records machine, versions, command, seed, and tolerances.
- [ ] Archive-scale record includes the network context and transfer statistics.

## Archival and manuscript actions

- [ ] Freeze the public API and create signed tag `v0.1.0`.
- [ ] Create GitHub release and archive it in Zenodo, Software Heritage, or GigaDB.
- [ ] Add the resulting DOI to citation and manuscript files.
- [ ] Register bio.tools and RRID identifiers and add them to metadata.
- [ ] Publish a Code Ocean capsule or equivalent reproducible environment.
- [ ] Cite DANDI:000350 version `0.240822.1759`, its creators, asset, and relevant
  primary publication in the manuscript reference list.
- [ ] Complete author contributions, funding, competing interests, and journal
  template sections.
