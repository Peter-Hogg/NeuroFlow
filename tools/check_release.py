"""Audit automated release invariants and report unresolved manual gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from neuroflow.benchmarking import validate_benchmark_record
from tools.check_metadata import metadata_errors


@dataclass(frozen=True)
class ReleaseAudit:
    errors: tuple[str, ...]
    manual_actions: tuple[str, ...]


def audit_release(root: Path | None = None) -> ReleaseAudit:
    repository = root or Path(__file__).resolve().parents[1]
    errors = list(metadata_errors(repository))
    required = (
        "README.md",
        "CITATION.cff",
        ".zenodo.json",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "RELEASE_CHECKLIST.md",
        "uv.lock",
        ".github/workflows/ci.yml",
        ".github/workflows/release-validation.yml",
        ".github/workflows/publication-benchmarks.yml",
        "benchmarks/benchmark_fish_pipeline.py",
        "benchmarks/benchmark_fish_trace_baseline.py",
        "tests/test_cellpose_real.py",
    )
    for relative in required:
        if not (repository / relative).is_file():
            errors.append(f"required release file is missing: {relative}")

    results = repository / "benchmarks" / "results"
    if results.is_dir():
        for path in sorted(results.glob("*.json")):
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid benchmark JSON {path.name}: {exc}")
                continue
            if isinstance(value, dict) and "benchmark_schema_version" in value:
                try:
                    validate_benchmark_record(value)
                except ValueError as exc:
                    errors.append(f"invalid benchmark record {path.name}: {exc}")

    manual: list[str] = []
    if not (repository / "LICENSE").is_file():
        manual.append("select an OSI-approved license and synchronize metadata")
    fish_record = results / "publication-fish-soma-traces.json"
    if not _is_clean_publication_record(fish_record):
        manual.append("retain a clean publication-classified full fish pipeline record")
    lindi_record = results / "publication-fish-lindi-dask-traces.json"
    if not _is_clean_publication_record(lindi_record):
        manual.append("retain a clean publication-classified LINDI/Dask baseline")
    manual.extend(
        (
            "confirm public CI, real Cellpose, docs, package, and Docker jobs",
            "complete or explicitly scope expert biological validation",
            "tag and archive the exact release candidate, then add its DOI",
        )
    )
    return ReleaseAudit(tuple(errors), tuple(manual))


def _is_clean_publication_record(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(value, dict) or value.get("classification") != "publication":
        return False
    environment = value.get("environment")
    if not isinstance(environment, dict):
        return False
    git = environment.get("git")
    return isinstance(git, dict) and git.get("dirty") is False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail for external or maintainer-controlled release gates",
    )
    args = parser.parse_args()
    audit = audit_release()
    for item in audit.errors:
        print(f"FAIL: {item}")
    for item in audit.manual_actions:
        print(f"MANUAL: {item}")
    if not audit.errors:
        print("PASS: automated repository release invariants")
    if audit.errors or (args.strict and audit.manual_actions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
