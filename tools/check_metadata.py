"""Check version, citation, and unresolved-license metadata consistency."""

from __future__ import annotations

import json
import re
from pathlib import Path

import tomllib


def metadata_errors(root: Path | None = None) -> list[str]:
    repository = root or Path(__file__).resolve().parents[1]
    project = tomllib.loads((repository / "pyproject.toml").read_text())["project"]
    version = str(project["version"])
    errors: list[str] = []
    version_file = (repository / "neuroflow" / "_version.py").read_text()
    docs_conf = (repository / "docs" / "conf.py").read_text()
    citation = (repository / "CITATION.cff").read_text()
    zenodo = json.loads((repository / ".zenodo.json").read_text())

    expected = {
        "neuroflow/_version.py": _match(version_file, r'__version__\s*=\s*"([^"]+)"'),
        "docs/conf.py": _match(docs_conf, r'release\s*=\s*"([^"]+)"'),
        "CITATION.cff": _match(citation, r"(?m)^version:\s*([^\s]+)\s*$"),
    }
    for location, actual in expected.items():
        if actual != version:
            errors.append(
                f"version mismatch: pyproject.toml={version!r}, {location}={actual!r}"
            )
    if zenodo.get("title") != project.get("description") and not zenodo.get("title"):
        errors.append(".zenodo.json must contain a title")

    license_file = repository / "LICENSE"
    project_license = project.get("license")
    citation_license = _match(citation, r"(?m)^license:\s*([^\s]+)\s*$")
    zenodo_license = zenodo.get("license")
    if license_file.exists():
        declared = [project_license, citation_license, zenodo_license]
        if any(value is None for value in declared):
            errors.append(
                "LICENSE exists but pyproject.toml, CITATION.cff, and .zenodo.json "
                "do not all declare it"
            )
        elif len({str(value) for value in declared}) != 1:
            errors.append("license identifiers disagree across release metadata")
    else:
        declarations = (project_license, citation_license, zenodo_license)
        if any(value is not None for value in declarations):
            errors.append("metadata declares a license but root LICENSE is absent")
        if not (repository / "docs" / "development" / "license_decision.md").is_file():
            errors.append(
                "unresolved license requires "
                "docs/development/license_decision.md"
            )
    if re.search(r"(?m)^date-released:", citation):
        errors.append(
            "CITATION.cff must not claim date-released before an external release"
        )
    return errors


def _match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip("\"'") if match else None


def main() -> None:
    errors = metadata_errors()
    if errors:
        raise SystemExit("\n".join(f"FAIL: {item}" for item in errors))
    print("PASS: package, citation, and license-decision metadata are consistent")


if __name__ == "__main__":
    main()
