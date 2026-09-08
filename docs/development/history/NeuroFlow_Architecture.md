# Architecture Notes

## Core modules

-   neuroflow.io
    -   DANDI
    -   NWB
    -   Zarr
    -   local filesystem
-   neuroflow.execution
    -   Dask Array
    -   Delayed
    -   Distributed execution
-   neuroflow.adapters
    -   Cellpose
    -   Pynapple
    -   Generic NumPy
    -   PyTorch
-   neuroflow.storage
    -   Zarr
    -   Parquet
    -   Provenance

## Adapter contract

Each adapter should define:

-   input selection
-   partition strategy
-   overlap requirements
-   execution function
-   merge strategy
-   output schema

The framework owns orchestration. The external library owns scientific
computation.
