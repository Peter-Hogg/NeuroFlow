# Expected retained outputs

Each publication experiment should retain:

- a benchmark JSON conforming to benchmark schema version 1;
- the canonical `WorkflowSpec` JSON when the experiment uses a serializable
  NumPy-expression workflow;
- result provenance, partition manifests, and checksums;
- console log and exact command;
- environment record, cache state, and network context;
- numerical comparison JSON with tolerances and reference implementation;
- paper tables and figures regenerated only from retained JSON.

The archive fish run additionally expects a verified Zarr projection, a
middle-plane PNG preview, real Cellpose labels and object table, compact
`(time, cell)` traces, direct Cellpose equivalence, a direct NumPy trace subset,
whole-workflow measurements, and completed-result resume evidence. A fair
manual LINDI/Dask record over the same masks is required for final submission.
Scientific segmentation validation additionally expects frozen model identity,
blinded annotations, metrics, reviewer checklist, and adjudication notes. These
files are intentionally absent until the experiments are actually performed.
