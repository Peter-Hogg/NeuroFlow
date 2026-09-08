# NeuroFlow

NeuroFlow provides a small named-axis API for running NumPy functions on NWB
data while managing lazy access, bounded Dask execution, durable outputs,
provenance, resume, and verification.

```python
import numpy as np
import neuroflow

movie = neuroflow.load("session.nwb.zarr", name="movie")  # axes: time, y, x
lazy_projection = np.sqrt(np.median(movie[:50], axis="time") + 1)
projection = lazy_projection.persist(
    "projection.zarr",
    chunks=(256, 256),
    memory_limit="2 GiB",
)
assert projection.workflow.verify().valid
```

Run `python -m examples.local_nwb_zarr` from a checkout to create a small
`session.nwb.zarr` this snippet works against.

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
concepts
workflows
High_Level_API
examples
limitations
reproducibility
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

Maintainer records — release decisions, publication status, and historical
design documents — live in the repository under `docs/development/` and are
deliberately not part of this site; they are not a second user-facing API
contract.
