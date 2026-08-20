# Project-name collision assessment

Assessment date: 2026-08-20. This is a discoverability review, not trademark
advice, and it does not authorize a rename.

`NeuroFlow` has meaningful collisions in both neuroscience and software:

- a [CVPR 2026 visual neural encoding/decoding project](https://github.com/MichaelMaiii/NeuroFlow)
  uses the identical repository name in a closely related neuroscience field;
- a [2026 browser-based mouse-brain atlas workflow](https://guangweizhang.com/tool-neuroflow.html)
  uses the identical name for another neuroscience analysis tool;
- [older Read the Docs pages](https://neuroflow.readthedocs.io/en/latest/)
  describe a different Python project as NeuroFlow;
- `neuroflow-sdk` is used for an [EEG/BCI platform](https://www.piwheels.org/project/neuroflow-sdk/);
- [NeuroFlow](https://www.neuroflow.com/) is also an established behavioral
  health technology company.

The collision risk is high for paper search, web search, citations, package
support, and spoken references. The exact `neuroflow` project URL returned no
PyPI page during this assessment, but availability must be checked again at
release time and does not remove the broader naming risk.

## Maintainer decision

Before the first archival release, decide whether to retain the name or adopt a
more distinctive one. If retaining it, consistently use a qualified phrase
such as **“NeuroFlow for bounded NWB analysis”** in the paper title, abstract,
README, metadata, and search keywords. If renaming, do it before minting a DOI
and coordinate the repository, import package, distribution, CLI, docs URL,
citation metadata, container labels, and provenance compatibility policy.
