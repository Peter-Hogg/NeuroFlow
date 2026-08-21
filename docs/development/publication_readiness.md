# Publication-readiness engineering record

This document records the local baseline and the scope of the release-candidate
work. It distinguishes observed evidence from infrastructure that still needs
to be run elsewhere.

## Initial baseline — 2026-08-20

Repository commit before changes: `5f85f669133d64525dca91c59de27effd95f8a3c`
(`master`). The worktree was initially clean.

- Python support declared in package metadata and CI: 3.10, 3.11, 3.12, and
  3.13. The available local interpreter was CPython 3.12.3.
- The untouched suite collected 170 tests. Its distributed `LocalCluster` test
  could not bind a local scheduler socket inside the restricted execution
  sandbox; the remaining targeted baseline tests did not expose a product
  failure. This environmental limitation is not recorded as a passing full
  suite. Final evidence was recorded in the root evidence ledger, retired from
  the public tree in `04633da` (last version: `7e54b6c:PUBLICATION_READINESS.md`).
- Ruff: passed on the untouched tree.
- basedpyright: passed with 0 errors, warnings, or notes.
- Sphinx: passed with warnings treated as errors after directing PyNWB's cache
  to a writable temporary directory.
- Markdown link check: passed.
- Coverage: not established at baseline because the full run was interrupted
  by the local-socket restriction; no percentage is inferred.
- Package build: the first sandboxed attempt could not use uv's read-only
  managed-Python directory. This was an environment limitation, not a package
  build result. A uv environment was subsequently created from `uv.lock` with
  CPython 3.12 for implementation and final verification.
- Docker: not tested because no Docker executable/daemon is installed in this
  environment.
- Retained benchmarks: five historical local projection repetitions and one
  2026-08-13 DANDI fish record were present. The fish record explicitly says it
  predates the current NumPy-expression engine and lacks several measurements.

## Publication-critical gaps identified

The initial repository already enforced explicit execution boundaries,
partition-level persistence, checksum verification and repair, remote HDF5
constraints, and conservative NumPy support. The main missing evidence or
machinery was:

- a safe, portable, versioned workflow file and reproduction command;
- bounded multi-stage global reductions reusable by downstream partitions;
- structured planning and post-run environment/execution reports;
- a common publication benchmark schema and current-engine experiment suite;
- synchronized release metadata checks, an explicit license decision point,
  release-readiness automation, and publication artifact organization.

External archive-scale runs, competitor measurements, expert biological
validation, license approval, DOI deposition, releases, and journal actions are
outside this local engineering run and must not be marked complete without
their own retained evidence.

## Revised flagship implementation

The updated mission promotes the projection → actual Cellpose → whole-movie
soma trace workflow above the projection-only demonstration. The evidence
ledger that tracked this work was retired from the public tree in `04633da`
(last version: `7e54b6c:PUBLICATION_READINESS.md`); retained benchmark records
live under `benchmarks/results/` and the final run-book in
`RELEASE_CHECKLIST.md`. In particular,
the repository now has optional LINDI transport, capability-based HDF5 array
discovery, source-chunk-aware trace planning/execution, direct Cellpose and
NumPy comparison paths, a full fish runner, and a manual LINDI/Dask baseline.
Those harnesses are not substitutes for clean retained experiment records.
