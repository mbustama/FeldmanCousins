.. PyFC documentation master file

PyFC: A Python Framework for Feldman-Cousins Confidence Intervals
=================================================================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :alt: Python 3.8+

.. image:: https://github.com/mbustama/FeldmanCousins/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/mbustama/FeldmanCousins/actions
   :alt: CI Tests

.. important::
   **Important Links:**

   * `GitHub Repository <https://github.com/mbustama/FeldmanCousins>`_
   * `Example Jupyter Notebook <https://github.com/mbustama/FeldmanCousins/blob/main/examples/pyfc_tutorial.ipynb>`_

**PyFC** is a rigorous, high-performance frequentist statistical analysis framework for Python. It automates the construction of classical confidence intervals and regions using the unified Feldman-Cousins approach, seamlessly transitioning between one-sided upper limits and two-sided bounds while guaranteeing exact frequentist coverage.

Designed for high-energy physics, astrophysics, and general parametric modeling, PyFC handles both binned (histogram) and unbinned (event-by-event) data, integrates multiple optimization strategies (SciPy, UltraNest, Grid), and utilizes highly parallelized Monte Carlo pseudo-experiment generation.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   configuration
   methodology
   outputs
   references

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`