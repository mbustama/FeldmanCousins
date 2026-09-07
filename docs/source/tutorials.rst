Tutorial Notebooks
===================

The ``examples/`` directory in the repository contains eight runnable Jupyter notebooks,
numbered ``01``-``08`` in the order we'd suggest reading them. Each one is self-contained
(states its own imports and mock data) and ends with a "Next steps" pointer to the notebooks
that naturally follow it, so you can also jump straight to whichever topic matches your own
project.

.. note::
   These pages are static documentation and do not execute the notebooks inline. Follow a
   link below to view (or download and run) the notebook on GitHub.

.. list-table::
   :widths: 8 30 42 20
   :header-rows: 1

   * - #
     - Notebook
     - What it covers
     - Read this if...
   * - 01
     - `pyfc_quickstart_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/01_pyfc_quickstart_tutorial.ipynb>`_
     - A full binned model and a full unbinned model, end to end, with real output. The runnable counterpart to :doc:`quickstart`.
     - **Start here.** This is the on-ramp for a brand-new project -- copy whichever of the two models matches your data.
   * - 02
     - `pyfc_high_dimensional_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/02_pyfc_high_dimensional_tutorial.ipynb>`_
     - Scaling ``compute_fc_intervals`` to 5 and then 10 free parameters; why 1D scans stay cheap while 2D contours grow combinatorially (:math:`\binom{n}{2}` pairs).
     - Your model has more than 2-3 free parameters.
   * - 03
     - `pyfc_non_contiguous_data_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/03_pyfc_non_contiguous_data_tutorial.ipynb>`_
     - Passing arbitrarily-shaped, even non-contiguous (e.g. transposed), N-D binned ``data`` arrays straight into PyFC, with a proof of correctness.
     - Your histogram isn't a plain 1D array, or you build it with a different axis order than PyFC expects.
   * - 04
     - `pyfc_algorithmic_features_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/04_pyfc_algorithmic_features_tutorial.ipynb>`_
     - The ``sparsify_grid``, ``smooth_1d``/``smooth_2d``, and ``use_finite_mc_correction_binned`` knobs, each demonstrated on vs. off -- plus how disconnected accepted intervals get reported.
     - You need to speed up a 2D scan, polish a plot, or your MC templates have limited statistics.
   * - 05
     - `pyfc_strategy_comparison_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/05_pyfc_strategy_comparison_tutorial.ipynb>`_
     - Benchmarking ``"grid"``/``"scipy"``/``"hybrid"``/``"ultranest"`` on the same model, then building a fully custom Matplotlib figure directly from PyFC's saved ``.json``/``.npz`` output (no dependency on ``generate_corner_plot``).
     - You're choosing an optimizer strategy, or you want a publication-quality figure beyond PyFC's built-in plot.
   * - 06
     - `pyfc_joint_constraints_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/06_pyfc_joint_constraints_tutorial.ipynb>`_
     - Reproducing and fixing the flat-gradient optimizer trap that comes from a joint (non-box) constraint like a simplex (:math:`f_e + f_\mu \leq 1`), using ``bounds_func`` and ``constraints``.
     - Two or more of your parameters are linked by an inequality that a per-parameter ``[lo, hi]`` box can't express.
   * - 07
     - `pyfc_checkpointing_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/07_pyfc_checkpointing_tutorial.ipynb>`_
     - A genuinely-running ``compute_fc_intervals`` call, killed mid-analysis with ``SIGTERM`` (mimicking a Slurm walltime kill), then correctly resumed with ``warm_start=True`` -- with a checkpoint inspection and a determinism check proving nothing was silently recomputed or corrupted.
     - You're running on a preemptible/walltime-limited cluster, or just want to see the checkpointing "Salient Feature" actually happen.
   * - 08
     - `pyfc_gaussian_priors_tutorial.ipynb <https://github.com/mbustama/FeldmanCousins/blob/main/examples/08_pyfc_gaussian_priors_tutorial.ipynb>`_
     - Adding *soft* constraints on nuisance parameters with ``extra_nll``, including a closed-form check of PyFC's :math:`-2\ln L` convention, a degeneracy that a 3% prior on the background visibly tightens, **correlated (pairwise and higher-dimensional) priors** built with ``pyfc.priors`` -- with the marginal-vs-conditional distinction worked through as running code -- and one-sided/derived-quantity penalties a table of ``(centre, width)`` can't express.
     - A nuisance parameter in your model was measured elsewhere and you want that external result to constrain the fit.

``01``-``03`` build on each other and are worth reading in order for a first project;
``04``-``08`` are independent, narrower deep-dives you can read in any order once you need
that specific capability.

See :doc:`quickstart` for the non-notebook version of the basic workflow, and
:ref:`joint-simplex-constraints` for the reference documentation that notebook ``06`` walks
through hands-on. Notebooks ``06`` and ``08`` are the two halves of the constraint story --
hard geometry and soft priors respectively -- and are best read together.
