# Software release checklist

## Repository and software

- [ ] Maintainer selects an OSI-approved license and adds `LICENSE` plus matching
  `pyproject.toml`, `CITATION.cff`, and `.zenodo.json` metadata.
- [x] Author, project URL, classifiers, citation, changelog, contribution,
  conduct, security, and version metadata are present and machine-checked.
- [x] Locked uv environments support Python 3.10 through 3.13 in CI; Python
  3.10 has an explicit `tomli` fallback.
- [x] Network-free tests compare NumPy, NWB-Zarr, and NWB-HDF5 results.
- [x] A real local LINDI/PyNWB bridge test is in the optional-dependency CI job.
- [x] Trace outputs have planning, coordinates, provenance, manifests,
  checksums, verification, and per-partition resume.
- [x] Friendly Cellpose segmentation processes complete 2-D images or complete
  y/x planes and prevents accidental unreconciled y/x tiles.
- [ ] Public GitHub matrix, optional LINDI, real Cellpose, docs, package, and
  CodeQL workflows pass on the release-candidate commit.
- [ ] Docker image builds in an environment with a working daemon and its base
  image digest is retained.

## Validation evidence

- [x] Deterministic trace tests compare against direct NumPy with exact output,
  including a label spanning multiple source chunks.
- [x] Trace planning is tested to read no movie values and to skip empty source
  chunks.
- [x] Deterministic interruption, resume, corruption detection, and selective
  repair are covered by a retained local benchmark harness.
- [x] An opt-in test runs actual Cellpose and compares exact labels with direct
  Cellpose on the same deterministic projection.
- [ ] Retain the successful real Cellpose job log/model cache identity for the
  release candidate; mocked tests alone are not publication evidence.
- [ ] Run and retain the clean end-to-end DANDI fish projection → Cellpose →
  whole-movie trace record with memory, transfer, wall time, checksums,
  numerical validation, integrity, and resume.
- [ ] Run and retain the clean PyNWB + LINDI + Dask trace baseline over the exact
  same masks; run the remfile baseline if used in the manuscript table.
- [ ] Complete expert/manual biological assessment, or state explicitly in the
  manuscript that only software-path equivalence was established.
- [x] Preserve the legacy fish record as `historical` and the dirty current
  projection measurement as development evidence, not publication evidence.

## Release and archival actions

- [ ] Freeze the public API and create signed tag `v0.1.0` from the exact clean
  commit used for publication experiments.
- [ ] Create a GitHub release and archive the tagged software in Zenodo and/or
  Software Heritage.
- [ ] Add assigned software DOI and release date to citation/archive metadata.
- [x] Record DANDI:000350 version, DOI, asset, object, and primary article in the
  reproducibility guide.
