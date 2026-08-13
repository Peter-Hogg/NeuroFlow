# Agent Development Plan

## Milestone 1

Open an NWB-Zarr dataset from DANDI and expose a lazy Dask-backed
object.

## Milestone 2

Execute arbitrary NumPy functions over time or spatial blocks.

## Milestone 3

Persist outputs as chunked Zarr arrays and Parquet tables.

## Milestone 4

Implement the first adapter (Cellpose or Pynapple).

## Milestone 5

Demonstrate an end-to-end zebrafish workflow.

## Non-negotiable engineering rules

-   Never call compute() inside library code.
-   Never materialize entire datasets.
-   Core package contains no scientific algorithms.
-   Every pipeline records provenance.
-   Resume interrupted jobs.
-   Design APIs only after two concrete use cases require them.
