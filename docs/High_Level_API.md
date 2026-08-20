# NumPy-compatible named arrays

`NeuroArray` is NeuroFlow's user-facing lazy array. It implements a deliberate
NumPy subset while retaining NWB axis names, source identity, selection bounds,
and native chunk metadata.

```python
import numpy as np
import neuroflow

movie = neuroflow.load(source, name="NeuronOnePhotonSeries")
bounded = movie[:50]

# No numerical reads: this only builds a canonical expression.
projection = np.sqrt(np.median(bounded, axis="time") + 1)

# Bounded execution begins here.
result = projection.persist(
    "projection.zarr",
    chunks=(256, 256, 1),
    max_workers=2,
    memory_limit="2 GiB",
)
assert result.workflow.verify().valid
```

## Supported operations

The compatibility boundary is explicit. Unsupported calls raise `TypeError`
while the expression is being built; NeuroFlow never falls back to converting
the source into a NumPy array.

| Category | Supported surface |
|---|---|
| Metadata | `.shape`, `.axes`, `.dtype`, `.ndim`, `.size`, `.nbytes`, `len`, `repr`, `np.shape`, `np.ndim`, `np.size` |
| Selection | `.isel(axis=slice(...))` and rank-preserving contiguous `array[...]` slices with one ellipsis |
| Arithmetic | `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `+`/`-`, `abs` and their scalar reverse forms |
| Comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| Ufuncs | `sqrt`, `exp`, `expm1`, `log`, `log1p`, `sin`, `cos`, `tan`, `minimum`, `maximum`, `isfinite`, `isinf`, `isnan` |
| Reductions | `sum`, `mean`, `min`, `max`, `median`, scalar `percentile`, scalar `quantile` as methods or NumPy functions |
| Dtypes | `.astype(...)`; NumPy 2 scalar-promotion and reduction-dtype rules are preserved |

Reductions accept an axis name, positive or negative integer, tuple containing
names and/or integers, or `None`. Axis forms are normalized to names in
provenance, so `movie.mean("time")` and `np.mean(movie, axis=0)` describe the
same workflow. `keepdims=True` is supported. `sum` and `mean` accept `dtype=`.

```python
temporal_mean = movie.mean("time")
joint_sum = np.sum(movie, axis=("time", "x"), dtype="float64")
middle_signal = np.percentile(movie, 50, axis="time", keepdims=True)
```

Percentile and quantile currently require one scalar `q`; a vector would add an
unnamed dimension and is rejected.

## Operand and broadcasting rules

Python and NumPy scalars broadcast normally, including NumPy 2's distinction
between weak Python scalars and typed NumPy scalars. Two `NeuroArray` operands
may be combined when they refer to the same source selection and have identical
shapes and axes.

Raw ndarrays, differently bounded selections, and general array broadcasting
are rejected. A global 0-D `sum`, `mean`, `min`, or `max` of the same selection
is the one deliberate broadcast exception. It becomes a bounded persisted
stage and its verified scalar is reused by every downstream partition.

```python
scaled = (movie + 1.0) / 2                 # supported
combined = movie[:50] + movie[50:100]     # rejected: different selections
normalized = movie / movie.max()           # supported: staged scalar
centered = movie - movie.mean("time")      # rejected: non-scalar broadcast
```

The last expression requires a named-axis array broadcast, which the expression
engine does not implement. Use an explicit custom multi-input analysis stage.
NeuroFlow will not compute a tile-local mean or repeatedly reread the complete
remote array and pretend either behavior is equivalent to NumPy.

## Explicit execution boundaries

`compute()` materializes an in-memory NumPy array and defaults to a conservative
1 GiB working-memory estimate:

```python
small = movie[:5, :128, :128].mean("time").compute()
```

Pass a different `memory_limit` explicitly when appropriate. The check happens
before numerical I/O. `compute()` uses Dask's threaded scheduler so open HDF5
handles are not sent to processes.

`persist()` is the normal boundary for large expressions and defaults to a
conservative 2 GiB per-task memory limit. Pass `memory_limit=` explicitly when
the workload needs a different bound. It:

- keeps every reduction axis whole inside a task;
- tiles retained axes using native chunks and requested output-chunk alignment;
- fuses compatible elementwise work with the reduction;
- rejects a task whose estimated input, outputs, intermediates, and exact
  quantile workspace exceed the declared memory limit;
- writes each partition directly to Zarr;
- prevents parallel partitions from sharing a writable Zarr chunk;
- records the canonical expression, selected asset, absolute bounds, NumPy
  version, partition layout, resource policy, and checksums;
- persists bounded partials for global scalar stages and resumes only valid
  partials before deterministically combining them;
- resumes only partitions whose identity and checksum still match.

The legacy convenience form remains available:

```python
result = movie[:50].median(
    "time",
    output="projection.zarr",
    chunks=(256, 256, 1),
    memory_limit="2 GiB",
)
```

It is equivalent to `movie[:50].median("time").persist(...)`.
Unlike the original 0.1 helper, it follows NumPy's output dtype and stores the
component as `result`. Add `.astype("float32")` before `persist()` when that is
the scientific and storage contract you intend.

## Materialization safeguards

The following operations intentionally fail before reading source data:

```python
np.asarray(movie)       # use movie.compute()
np.array(movie)         # use movie.compute()
for frame in movie:     # select a bounded slice, then compute
    ...
bool(movie)             # reduce explicitly, then compute
np.concatenate([...])   # not in the supported function table
np.add(movie, 1, out=...)  # mutation is not supported
```

Integer indexing, `newaxis`, stepped slices, boolean masks, and fancy indexing
are also rejected because the first release preserves array rank and named-axis
identity.

## Specialized stages

Persisted arrays can re-enter a workflow with `neuroflow.open_array()`. Public
reopening requires complete result/provenance records and verifies every
partition checksum before treating the output as a new source. The resulting
source identity includes the canonical partition descriptors and recorded
checksums, so changed upstream bytes cannot silently resume downstream work.
`open_array(..., verify=False)` is an explicit trusted fast path only for an
output that successfully finished in the current process; `.persist()` uses
that path internally to avoid immediately rereading the bytes it just wrote.

Use
`projection.segment(...)` for a Cellpose-style segmentation stage and
`movie.extract_traces(labels, ...)` for bounded trace extraction. Specialized
adapters remain explicit boundaries rather than pretending an external model is
a NumPy ufunc.
