# Reproducibility

NeuroFlow uses `uv.lock` as the dependency record. The repository targets
Python 3.10 through 3.13; the publication image uses Python 3.10.

## Recreate and verify the development environment

```bash
git clone https://github.com/Peter-Hogg/NeuroFlow.git
cd NeuroFlow
uv python install 3.10
uv sync --locked --dev --python 3.10
uv run ruff check .
uv run basedpyright
uv run pytest --cov=neuroflow --cov=neuroflow_cellpose \
  --cov=neuroflow_pynapple --cov-fail-under=80
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
```

`python tools/check_release.py` performs the lightweight maintainer audit and
reports external or human decisions separately as manual actions.

## Portable workflow records

A `WorkflowSpec` is canonical JSON describing an allowlisted NumPy expression,
source identity and selection, partition policy, resource budget, create-only
Zarr output, schema version, and NeuroFlow version. It never contains pickled
functions or executable Python.

```python
import json
import neuroflow

movie = neuroflow.load("session.nwb.zarr", name="movie")[:50]
workflow = (movie / movie.max()).to_spec(
    "normalized.zarr",
    chunks=(256, 256),
    memory_limit="1 GiB",
)
workflow.to_json("workflow.json")
print(json.dumps(workflow.plan().to_dict(), indent=2))
result = neuroflow.reproduce(workflow)
assert result.verify().valid
```

The equivalent command-line flow is:

```bash
neuroflow plan workflow.json
neuroflow reproduce workflow.json
neuroflow environment
```

Reproduction validates the schema and canonical expression, resolves only
known source and partition kinds, checks pinned source identity, refuses output
symlinks and implicit overwrite, and fails clearly when a required backend is
unavailable. A direct remote URL should include a stable `version`; a mutable
local file's lightweight identity is not a full content hash.

## Docker reproduction image

The Dockerfile pins `uv` and performs a locked, non-editable install. It accepts
the Python base as a build argument so a publication run can pin a registry
digest without hard-coding a digest that has not been verified by the
maintainer:

```bash
docker build \
  --build-arg PYTHON_IMAGE=python@sha256:<verified-python-3.10-slim-digest> \
  --build-arg NEUROFLOW_GIT_SHA="$(git rev-parse HEAD)" \
  --build-arg NEUROFLOW_GIT_DIRTY="$(test -z "$(git status --porcelain)" && echo false || echo true)" \
  -t neuroflow-publication .
docker run --rm neuroflow-publication environment
docker run --rm --entrypoint pytest neuroflow-publication -q
```

Resolve and record the digest on the experiment date. Docker was not available
on every development host, so a successful image build must be retained by CI
or the publication runner rather than inferred from the Dockerfile.

## Network-free publication experiments

```bash
uv run python -m benchmarks.benchmark_projection \
  --classification publication \
  --output benchmarks/results/publication-local-projection.json
uv run python -m benchmarks.benchmark_scaling \
  --sizes 128,256,512 --frames 16 --repetitions 3 \
  --output benchmarks/results/publication-scaling.json
uv run python -m benchmarks.benchmark_resume_integrity \
  --output benchmarks/results/publication-resume-integrity.json
uv run python publication/generate_tables.py
```

Every current record includes the environment and Git state. Peak RSS is a
process high-water mark, not a hard limit; unavailable byte-transfer data is
`null`, not estimated after the fact.

## Archive-scale case study

```bash
bash benchmarks/run_fish_case_study.sh publication/runs/fish-projection
```

Use a fresh path and record cache/network context. Preserve the result and use
resume for follow-up checks instead of repeatedly transferring public data.
The retained 2026-08-13 case-study record is historical because it predates the
current execution engine. It cannot support current-engine performance claims.

## Data citation

The fish experiment pins:

- Dandiset `DANDI:000350`, version `0.240822.1759`;
- DANDI DOI <https://doi.org/10.48324/dandi.000350/0.240822.1759>;
- CC-BY-4.0 asset `4f898ff7-6084-4e84-a449-f05811c1d951`;
- `/acquisition/NeuronOnePhotonSeries`;
- primary article <https://doi.org/10.1016/j.cell.2019.05.050>.

Software equivalence does not establish biological interpretation. Follow the
separate protocol in `../publication/experiments/scientific_validation_protocol.md`
before making segmentation or trace-validity claims.
