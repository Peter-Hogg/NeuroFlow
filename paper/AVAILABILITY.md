# Availability statements — draft

## Source code and requirements

- Project name: NeuroFlow
- Project home: https://github.com/Peter-Hogg/NeuroFlow
- Operating systems: platform independent; validated CI platforms must be listed
  from the release workflow
- Programming language: Python 3.10 or newer
- Requirements: recorded in `pyproject.toml` and locked in `uv.lock`
- License: pending maintainer selection of an OSI-approved license
- Archived release DOI: pending release deposit
- RRID and bio.tools identifier: pending registration

## Data availability

Synthetic test data are generated transparently by `tests/conftest.py`, with an
expected projection in `tests/data/projection_reference.json`. The archive-scale
case study uses the immutable DANDI identifier and asset recorded by
`examples/dandi_fish_projection.py`. The manuscript reference list must cite the
Dandiset accession/version and its creators. Benchmark JSON and figure outputs
will be deposited with the release in an appropriate DOI-bearing repository.
