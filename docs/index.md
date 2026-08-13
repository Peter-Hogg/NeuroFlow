# NeuroFlow

NeuroFlow lets neuroscience analyses look like ordinary NumPy while it manages
lazy NWB access, bounded Dask execution, durable outputs, provenance, resume,
and verification.

```python
import neuroflow

movie = neuroflow.load("session.nwb.zarr", name="calcium_movie")
projection = movie.isel(time=slice(0, 50)).median(
    "time",
    output="projection.zarr",
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
High_Level_API
examples
limitations
Publication_Reproducibility
```

```{toctree}
:maxdepth: 2
:caption: Reference and design

api
NeuroFlow_Architecture
NeuroFlow_Architecture_Decisions
```
