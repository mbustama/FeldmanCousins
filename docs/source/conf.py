# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Tell Sphinx where to find your Python package
sys.path.insert(0, os.path.abspath('../../'))

project = 'PyFC'
copyright = '2026, Mauricio Bustamante'
author = 'Mauricio Bustamante'

# Read the version from the installed distribution rather than hardcoding it here.
# This was hardcoded, and it went stale: it still said '0.1.0' while the package was at
# 0.10.0, so every page of the published documentation carried a version nine releases
# out of date. A second copy of a number that must match `pyproject.toml` will eventually
# disagree with it, and nothing about a docs build fails when it does.
#
# Note the argument is the DISTRIBUTION name (`PyFeldmanCousins`), not the import package
# (`pyfc`); asking for the wrong one raises PackageNotFoundError and would silently pin
# the docs at the fallback below. Both CI (`pages.yml` runs `pip install -e ".[docs]"`)
# and the documented local workflow install the package first, so the fallback should only
# ever be seen when building docs from an uninstalled checkout -- in which case a visibly
# wrong version is the correct outcome, rather than a plausible but invented one.
from importlib.metadata import PackageNotFoundError  # noqa: E402
from importlib.metadata import version as _distribution_version  # noqa: E402

try:
    release = _distribution_version('PyFeldmanCousins')
except PackageNotFoundError:
    release = '0.0.0+unknown'
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add extensions for pulling docstrings and supporting NumPy/Google style formats
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

# Mock heavy scientific imports so Sphinx doesn't crash if they aren't 
# installed in the documentation build environment (e.g., GitHub Actions CI).
autodoc_mock_imports = [
    'numpy', 
    'scipy', 
    'ultranest', 
    'numba', 
    'tqdm', 
    'matplotlib'
]

bibtex_bibfiles = ['refs.bib']
templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the HTML theme to the standard scientific layout
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# --- HTML Theme Options (Sidebar and GitHub Links) ---

# Uncomment and point to your logo file if you have one in docs/source/_static/
html_logo = "_static/pyfc_logo.png"

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'navigation_depth': 4,
    'vcs_pageview_mode': 'edit',
}

html_context = {
    "display_github": True,
    "github_user": "mbustama",
    "github_repo": "FeldmanCousins",
    "github_version": "main",
    "conf_py_path": "/docs/source/", # Adjust to "/docs/" if that is where conf.py lives
}
