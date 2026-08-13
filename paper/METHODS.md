# Methods — working draft

## Software design

NeuroFlow separates semantic NWB selection, physical source access, scientific
partition planning, NumPy-compatible adapters, Dask task execution, and durable
result storage. Source inspection and workflow planning are metadata-only.
Numerical reads begin only during explicit execution.

NWB-Zarr arrays remain fsspec/Zarr-backed. Local NWB-HDF5 arrays remain h5py
datasets, while remote DANDI NWB-HDF5 assets use the PyNWB-recommended remfile
seekable HTTP range reader with a configurable bounded memory cache. Logical
Dask partitions do not alter physical HDF5 chunks; the planner reports native
and processing chunks separately.

Each task receives one bounded NumPy array. Array outputs are written to Zarr,
tables to Parquet, and segmentation outputs to a composite dense-label/table
store. A partition is complete only after its data and SHA-256 checksum are
persisted and an atomic manifest is committed. Resume validates existing bytes
before skipping work. Corrupt partitions are recomputed independently.

## Named-axis operations

The high-level `NeuroArray` API maps named slices and reductions onto the lower
execution model. Reduction schemas explicitly declare removed axes and output
chunks. Persisted arrays retain axes and may become inputs to later workflows.

## Segmentation and traces

The friendly segmentation API rejects cross-y/x NeuroFlow tiling unless the user
explicitly requests an unreconciled result. The demonstrated safe workflow uses
complete image planes and delegates internal image tiling to Cellpose. Trace
extraction partitions the movie by time, streams one z-plane at a time, sums all
voxels belonging to each global label across planes, and divides by global voxel
counts. Trace windows have independent manifests and checksums and contain time
and cell coordinates.

## Correctness and performance evaluation

Synthetic deterministic NWB-Zarr and NWB-HDF5 recordings are compared with
direct NumPy expected values. The benchmark harness compares direct NumPy,
direct Dask over the NWB selection, and NeuroFlow, reporting numerical error,
wall time, peak resident memory, source size, result size, and verification.
The archive-scale case study uses the immutable DANDI version and asset recorded
in the example source. Repeated benchmark records and complete machine/network
metadata must accompany the final manuscript.

For biological segmentation validation, expert-reviewed instance masks should
be compared with `neuroflow.compare_segmentations`. It performs one-to-one
maximum-IoU object matching and reports precision, recall, F1, matched-object
IoU, and foreground Dice. The annotation protocol, blinded review procedure,
IoU threshold, excluded planes, and all individual-plane metrics must accompany
the manuscript; those empirical results cannot be inferred from unit tests.
