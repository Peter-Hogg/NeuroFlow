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

The archive fish run additionally expects a verified Zarr projection and a
middle-plane PNG preview. Scientific segmentation validation additionally
expects frozen model identity, blinded annotations, metrics, reviewer
checklist, and adjudication notes. These files are intentionally absent until
the experiments are actually performed.
