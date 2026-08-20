"""Privacy-conscious, machine-readable execution environment capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

from neuroflow._version import __version__

ENVIRONMENT_SCHEMA_VERSION = "1"
DEPENDENCIES = (
    "dask",
    "distributed",
    "fsspec",
    "h5py",
    "hdmf-zarr",
    "neuroflow",
    "numpy",
    "pandas",
    "pyarrow",
    "pynwb",
    "remfile",
    "s3fs",
    "scipy",
    "xarray",
    "zarr",
)
OPTIONAL_BACKENDS = ("cellpose", "pynapple", "lindi")


def capture_environment(*, include_hostname: bool = False) -> dict[str, object]:
    """Capture portable software/hardware facts without environment variables.

    Hostname is opt-in, and command output, credentials, URLs, and the complete
    process environment are never retained.
    """
    dependencies: dict[str, str | None] = {}
    for package in (*DEPENDENCIES, *OPTIONAL_BACKENDS):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    git = _git_state()
    result: dict[str, object] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "neuroflow_version": __version__,
        "git": git,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "description": platform.platform(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "model": _cpu_model(),
        },
        "total_ram_bytes": _total_ram_bytes(),
        "dependencies": dependencies,
    }
    if include_hostname:
        result["hostname"] = platform.node() or None
    return result


def _git_state() -> dict[str, object]:
    explicit_sha = os.environ.get("NEUROFLOW_GIT_SHA")
    explicit_dirty = os.environ.get("NEUROFLOW_GIT_DIRTY")
    if explicit_sha:
        return {
            "commit": explicit_sha,
            "dirty": (
                explicit_dirty.lower() in {"1", "true", "yes"}
                if explicit_dirty is not None
                else None
            ),
            "source": "environment",
        }
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
        return {"commit": commit or None, "dirty": bool(status), "source": "git"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "source": "unavailable"}


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[-1].strip() or None
    except OSError:
        pass
    value = platform.processor().strip()
    return value or None


def _total_ram_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return pages * page_size
    except (OSError, ValueError):
        pass
    return None
