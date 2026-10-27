"""Environment provenance must degrade to null, not crash, off POSIX."""

from __future__ import annotations

import os

import pytest

from neuroflow.provenance import environment


def test_total_ram_is_null_where_os_has_no_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Windows' os module has no sysconf; the 0.1.0 Windows CI leg crashed here.
    monkeypatch.delattr(os, "sysconf", raising=False)
    assert environment._total_ram_bytes() is None  # pyright: ignore[reportPrivateUsage]
