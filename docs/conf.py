import os
import sys


project = "emidm"
author = "OJWatson"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
]

# myst-nb configuration
# Don't execute notebooks during build (they should have saved outputs)
nb_execution_mode = "off"
source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".md": "myst-nb",
}

# Suppress non-critical warnings
suppress_warnings = ["myst.header"]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"

try:
    import pydata_sphinx_theme  # noqa: F401
except ModuleNotFoundError:
    html_theme = "alabaster"

if html_theme == "pydata_sphinx_theme":
    html_theme_options = {
        "github_url": "https://github.com/OJWatson/emidm",
        "navbar_align": "content",
        "show_toc_level": 2,
    }

html_static_path = ["_static"]

# Make package importable for autodoc
sys.path.insert(0, os.path.abspath("../src"))
