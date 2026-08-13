# Concepts

## Selection is semantic and lazy

An NWB selection records the object path, named axes, dtype, native chunks, and
absolute slice bounds. Equal-shaped slices at different positions are distinct
scientific inputs and therefore have different workflow identities.

## Processing chunks and storage chunks differ

Native chunks describe how the archive stores bytes. Processing partitions
describe the bounded arrays passed to an analysis function. Output chunks
describe the persisted Zarr layout. NeuroFlow reports all three rather than
pretending Dask can change an HDF5 file's physical chunks.

## A partition is the unit of recovery

Each completed partition has an atomic manifest and checksum. Resume validates
existing bytes before skipping work. Verification reads one owned partition at
a time and rejects manifest outputs outside the declared result root.

## Memory limits are estimates, not process isolation

NeuroFlow limits source partitions, declared adapter memory, concurrency, and
known setup operations. It cannot see allocations hidden inside arbitrary
native libraries. Third-party adapters should declare conservative resources,
and empirical peak memory should still be measured for publication workloads.
