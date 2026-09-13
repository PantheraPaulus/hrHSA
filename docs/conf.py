"""Sphinx configuration for the hrRSF documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Package location
# ---------------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"

sys.path.insert(0, str(SOURCE_DIRECTORY))


# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "hrHSA"
author = "Paul Kasko and Ralph Kühn"
copyright = "Kasko & Kühn, 2026"
release = "0.1.0"


# ---------------------------------------------------------------------------
# Sphinx extensions
# ---------------------------------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.images"
]

myst_fence_as_directive = ["mermaid"]
mermaid_output_format = "raw"

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_title = "hrHSA"

html_theme_options = {
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_nav_level": 2,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/YOUR-GITHUB-NAME/YOUR-REPOSITORY",
            "icon": "fa-brands fa-github",
        }
    ],
}

html_sidebars = {
    "**": [],
}

html_static_path = ["_static"]

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")