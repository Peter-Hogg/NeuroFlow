# Project Brief: NeuroFlow

## Working title

**NeuroFlow** --- A scalable execution framework for archive-scale NWB
analysis.

## Executive summary

NeuroFlow is not another neuroscience analysis library. It is an
execution and interoperability layer that allows existing Python
neuroscience tools to operate on extremely large NWB datasets stored
locally or on DANDI without requiring complete downloads or loading
entire datasets into memory.

The framework separates four concerns:

1.  Data access (NWB/DANDI)
2.  Execution (Dask)
3.  Scientific algorithms (Cellpose, Pynapple, scikit-image, PyTorch,
    etc.)
4.  Persistent outputs (Zarr, Parquet, NWB metadata)

Researchers write ordinary analysis functions or lightweight adapters.
NeuroFlow constructs lazy execution graphs, partitions work, manages
overlap, schedules execution from laptop to cluster, records provenance,
and stores outputs in scalable formats.

The flagship use case is whole-brain zebrafish light-sheet imaging where
movies, segmentations, and extracted traces are too large for
conventional RAM-based workflows.

## Core principles

-   No mandatory local staging of datasets.
-   No scientific algorithms in the core package.
-   Lazy execution until explicitly requested.
-   Outputs remain lazily accessible.
-   Adapter-based integration with existing neuroscience libraries.
-   Reproducible provenance attached to every analysis.
