"""Fail when a relative Markdown link points to a missing repository path."""

from __future__ import annotations

import re
from pathlib import Path

LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for document in sorted(root.rglob("*.md")):
        if any(part.startswith(".") for part in document.relative_to(root).parts):
            continue
        for target in LINK.findall(document.read_text()):
            value = target.split("#", 1)[0].strip().strip("<>")
            if not value or "://" in value or value.startswith("mailto:"):
                continue
            resolved = (document.parent / value).resolve()
            if not resolved.exists():
                errors.append(f"{document.relative_to(root)}: missing {target}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
