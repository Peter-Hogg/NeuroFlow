# External deposit instructions

These actions require maintainer accounts and cannot be completed by a local
build alone.

## GitHub and Zenodo

1. Select the project license and update the four license metadata locations in
   `RELEASE_CHECKLIST.md`.
2. Merge a clean release commit after all GitHub workflows pass.
3. Create signed tag `v0.1.0` and a GitHub release from `CHANGELOG.md`.
4. Enable the repository in the maintainer's Zenodo account and archive the
   release.
5. Add the assigned DOI to `CITATION.cff`, `.zenodo.json`, README, and the
   manuscript availability statement.
6. Optionally submit the same immutable package and benchmark records to GigaDB
   or Software Heritage.

## bio.tools and RRID

Register NeuroFlow at <https://bio.tools/> using `paper/biotools.json` as the
starting payload. Submit the resulting software record to SciCrunch Resource
Identification Portal, then add the bio.tools identifier and RRID to
`CITATION.cff`, `pyproject.toml`, and `paper/AVAILABILITY.md`.

## Code Ocean

Create a capsule from the tagged repository, using the supplied `Dockerfile` or
the locked uv environment. Run `bash scripts/reproduce_release.sh` as the
reproduction command and publish its test, coverage, benchmark, and distribution
artifacts. Add the capsule DOI to the manuscript.
