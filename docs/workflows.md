# Planning, workflow records, and reproduction

## Plan before reading numerical data

`NeuroArray.plan()` and `WorkflowSpec.plan()` open source metadata and build the
same bounded task graph used by execution without reading the numerical
payload. The report includes source and selection metadata, physical chunks,
partition count and maximum shape, estimated task memory, logical bytes read,
physical chunks touched and uncompressed bytes when determinable, output size,
resource assumptions, staged reductions, warnings, and boundedness reasons.

Each quantity is marked `estimated` or `unknown` in `ExecutionPlan.to_dict()`.
Physical-chunk bytes are not presented as measured network transfer: compression,
HTTP requests, range coalescing, and caches can differ.

```python
import json
import neuroflow

movie = neuroflow.load("session.nwb.zarr", name="movie")
workflow = (movie[:100] / movie[:100].max()).to_spec(
    "normalized.zarr",
    chunks=(10, 256, 256),
    memory_limit="1 GiB",
)
print(json.dumps(workflow.plan().to_dict(), indent=2))
```

## What `WorkflowSpec` records

Schema version 1 supports one canonical, allowlisted NumPy-expression workflow
over DANDI, local NWB-HDF5, NWB-Zarr, or a verified persisted NeuroFlow array.
It records:

- versioned source URI, DANDI asset or persisted component, and available
  source checksum/version;
- NWB path, absolute slice bounds, axes, shape, dtype, and physical chunks;
- canonical expression nodes and derived dtype/shape/axes metadata;
- adapter identity and output contract;
- spatial tile, time window, asset, or session partition policy;
- create-only Zarr output and scheduler, resume, workers, and memory limit;
- generating NeuroFlow version and original workflow identity/status.

Serialization is deterministic JSON. Unknown keys, duplicate JSON keys,
unsupported schema versions, excessive depth/size, arbitrary callables,
non-allowlisted operations, object dtypes, mismatched derived metadata, changed
source identity, output symlinks, and implicit overwrite all fail explicitly.
There is no pickle or expression `eval` path.

Schema migrations dispatch on `schema_version`. A future version must add an
explicit reader/migrator; version 1 never guesses how to interpret newer files.

## After execution

The result provenance records UTC start/end timestamps, wall time, completed,
computed and resumed/skipped tasks, completed partitions, output bytes, peak
RSS, available source bytes-read counters, stage results, integrity status,
source identity, Git SHA/dirty state, Python/platform/CPU/RAM, and dependency
versions. Hostname is excluded unless explicitly requested from the environment
capture API, and environment variables, credentials, and source URLs are not
copied into the machine record.

Every persisted partition and staged partial has an atomic manifest and
checksum. Resume trusts only valid owned outputs; corruption is reported and
only invalid work is recomputed.
