.. PyFC documentation master file

PyFC: A Python Framework for Feldman-Cousins Confidence Intervals
=================================================================

.. image:: https://github.com/mbustama/FeldmanCousins/actions/workflows/pytest.yml/badge.svg
   :target: https://github.com/mbustama/FeldmanCousins/actions/workflows/pytest.yml
   :alt: CI Tests

.. image:: https://github.com/mbustama/FeldmanCousins/actions/workflows/lint.yml/badge.svg
   :target: https://github.com/mbustama/FeldmanCousins/actions/workflows/lint.yml
   :alt: Code Quality

.. image:: https://codecov.io/gh/mbustama/FeldmanCousins/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/mbustama/FeldmanCousins
   :alt: Coverage

.. image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg
   :target: https://mbustama.github.io/FeldmanCousins/
   :alt: Documentation

.. image:: https://img.shields.io/pypi/v/PyFeldmanCousins.svg
   :target: https://pypi.org/project/PyFeldmanCousins/
   :alt: PyPI

.. image:: https://img.shields.io/badge/License-GPLv3+-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3+

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
   :target: https://github.com/astral-sh/ruff
   :alt: Code style: ruff

.. important::
   **Important Links:**

   * `GitHub Repository <https://github.com/mbustama/FeldmanCousins>`_
   * `Example Notebooks <https://github.com/mbustama/FeldmanCousins/tree/main/examples>`_ (see also :doc:`tutorials` for a guided tour)

**PyFC** is a rigorous, high-performance frequentist statistical analysis framework for Python. It automates the construction of classical confidence intervals and regions using the unified Feldman-Cousins approach, seamlessly transitioning between one-sided upper limits and two-sided bounds while guaranteeing exact frequentist coverage.

Designed for high-energy physics, astrophysics, and general parametric modeling, PyFC handles both binned (histogram) and unbinned (event-by-event) data, integrates multiple optimization strategies (SciPy, UltraNest, Grid, and a Hybrid of the two), and utilizes highly parallelized Monte Carlo pseudo-experiment generation.

Salient Features
-----------------

* **Unified binned and unbinned analysis**: Natively supports Poisson binned data (including arbitrary N-dimensional, even non-contiguous, histograms) and Extended Unbinned Maximum Likelihood (EUML) formulations.
* **Exact coverage**: Empirically derives the Profile Likelihood Ratio (PLR) test statistic distribution via dynamically generated Monte Carlo pseudo-experiments (toys), correctly reporting disconnected accepted regions as separate intervals rather than silently merging them.
* **Advanced optimizers**: Supports gradient-based L-BFGS-B (SciPy), nested sampling (UltraNest), brute-force grid scanning, and a Hybrid strategy that uses UltraNest's robustness for the one-time global fit and SciPy's speed for the many conditional fits.
* **Joint/simplex-constrained parameters**: ``bounds_func`` and ``constraints`` mechanisms express physical constraints between parameters (e.g. a simplex ``a + b <= 1``) that an independent per-parameter box can't -- no fragile hand-rolled penalty function required.
* **Massive parallelization**: GIL-bypassing via NumPy/Numba C-extensions for binned thread pooling, and ``ProcessPoolExecutor`` for unbinned continuous functions, with statistically-validated adaptive early-stopping (``adaptive_toys``) and batched dispatch (``toy_batch_size``) to cut wasted compute without introducing bias.
* **Finite Monte Carlo corrections**: Implements Argüelles, Schneider & Yuan's marginalized Poisson-Gamma mixture likelihood (``L_Eff``, :cite:p:`Arguelles2019izp`) to account for finite simulation statistics -- not the (different, profiled-nuisance-parameter) Barlow-Beeston likelihood, despite the similar "Poisson-Gamma" framing.
* **Optional 2D grid sparsification**: ``sparsify_grid`` (opt-in, off by default) uses Bivariate Spline interpolation and edge-tracing to skip toy generation deep inside/outside 2D contours -- a real speedup where it applies cleanly, though it doesn't yet scale reliably much past ~20x20 nodes per axis (see :doc:`methodology`).
* **State checkpointing**: "Warm start" capability saves binary state matrices after each parameter/pair scan completes, so an interrupted run (e.g. a Slurm walltime kill) resumes instead of restarting from scratch -- see :doc:`outputs`.
* **Interactive configuration**: Built-in CLI for generating serialized JSON experiment configurations -- see :doc:`configuration`.

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   tutorials
   configuration
   methodology
   outputs
   changelog
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