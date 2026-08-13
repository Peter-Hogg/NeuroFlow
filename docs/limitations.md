# Limits and scientific responsibilities

NeuroFlow provides execution and storage guarantees; it does not validate a
biological interpretation.

- Remote HDF5 requires a server supporting byte-range requests and uses threads.
- Logical x/y crops may still transfer complete physical HDF5 chunks.
- Segmentation across NeuroFlow x/y tiles is rejected by the friendly API unless
  explicitly marked unreconciled.
- Cellpose model choice, weights, thresholds, and biological accuracy require
  dataset-specific validation against expert-reviewed annotations.
- Candidate detectors in examples are quality-control tools, not classifiers.
- Memory budgets are conservative estimates, not operating-system hard limits.
- Network transfer metrics based on `Content-Length` describe observed response
  payloads; they are not a billing statement from DANDI.
- A reopened result can be verified and read, but resuming execution requires
  reconstructing the original adapter function.

For publication, report exact source versions, asset identifiers, selections,
commands, software versions, random seeds, numerical tolerances, hardware,
network context, validation protocols, and excluded data.
