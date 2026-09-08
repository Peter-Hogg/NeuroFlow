"""Sphinx configuration for the NeuroFlow documentation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

project = "NeuroFlow"
author = "Peter W. Hogg"
copyright = "2026, Peter W. Hogg"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
myst_enable_extensions = ["colon_fence", "deflist"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Maintainer records and historical design documents: tracked in the
    # repository, deliberately absent from the public site.
    "development/**",
]
html_theme = "furo"
html_title = "NeuroFlow documentation"
