# NeuroFlow

NeuroFlow provides a small named-axis API for running NumPy functions on NWB
data while managing lazy access, bounded Dask execution, durable outputs,
provenance, resume, and verification.

```python
import numpy as np
import neuroflow

movie = neuroflow.load("session.nwb.zarr", name="calcium_movie")
lazy_projection = np.sqrt(np.median(movie[:50], axis="time") + 1)
projection = lazy_projection.persist(
    "projection.zarr",
    chunks=(256, 256, 1),
    memory_limit="2 GiB",
)
assert projection.workflow.verify().valid
```

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
:caption: Reference and design

api
NeuroFlow_Architecture
NeuroFlow_Architecture_Decisions
```

Development and publication status are tracked in
`development/publication_readiness.md`; historical design notes are indexed in
`development/history/README.md` and are not a second user-facing API contract.
