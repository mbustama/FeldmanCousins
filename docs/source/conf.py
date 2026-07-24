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
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add extensions for pulling docstrings and supporting NumPy/Google style formats
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Set the HTML theme to the standard scientific layout
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
