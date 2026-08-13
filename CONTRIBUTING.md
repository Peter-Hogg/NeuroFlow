# Contributing to NeuroFlow

Bug reports, documentation improvements, tests, and focused pull requests are
welcome. Open an issue before a large API or storage-format change so its
scientific semantics can be agreed before implementation.

## Development setup

```bash
git clone https://github.com/Peter-Hogg/NeuroFlow.git
cd NeuroFlow
uv sync --locked --dev
uv run pytest
uv run ruff check .
uv run basedpyright
```

Tests must not depend on live archive availability. Add small synthetic NWB test
data and deterministic expected outputs to the network-free suite. Live DANDI
checks belong in explicitly invoked examples or release validation.

New array operations must declare axis transformations, bounded-read behavior,
output schema, resume semantics, and provenance identity. Scientific adapters
must document required overlap and whether cross-partition reconciliation is
implemented.

By contributing, you agree that your contribution will be distributed under
the project's selected open-source license.
