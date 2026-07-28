Configuration
=============

Using generate_config.py
------------------------
PyFC provides an interactive command-line tool to help users build their analysis configuration files securely and without typos. To interactively generate a configuration file from anywhere, simply run:

.. code-block:: bash

   python -m pyfc.generate_config

The script will prompt you with questions regarding your likelihood type, number of toys, parallelization preferences, and smoothing options. It validates your inputs and writes a file (by default, ``fc_config.json``) to your current working directory.

Configuration Parameters
------------------------
.. list-table:: 
   :widths: 20 45 20 15
   :header-rows: 1

   * - Parameter
     - Description
     - Allowed Values
     - Default
   * - ``likelihood_type``
     - Evaluates models via Poisson bins or Extended Unbinned Maximum Likelihood.
     - ``"binned"``, ``"unbinned"``
     - ``"binned"``
   * - ``cl``
     - Confidence Levels determining exact frequentist coverage integration targets.
     - List of floats ``(0.0, 1.0)``
     - ``[0.68, 0.90]``
   * - ``n_toys``
     - Monte Carlo pseudo-experiments generated per parameter space point.
     - Integer :math:`> 0`
     - ``500``
   * - ``strategy``
     - Optimizer used for finding global and conditional likelihood minima.
     - ``"scipy"``, ``"ultranest"``, ``"hybrid"``, ``"grid"``
     - ``"scipy"``
   * - ``scipy_method``
     - Overrides the ``scipy.optimize.minimize`` method. Only meaningful for ``strategy="scipy"``/``"hybrid"``. ``null`` lets PyFC pick automatically: ``"L-BFGS-B"`` by default, or ``"SLSQP"`` if ``constraints`` are supplied programmatically (see :ref:`joint-simplex-constraints`).
     - ``null``, ``"L-BFGS-B"``, ``"SLSQP"``, ``"trust-constr"``
     - ``null``
   * - ``use_finite_mc_correction_binned``
     - Shifts Poisson likelihood to a Negative Binomial to account for finite simulation stats.
     - ``True``, ``False``
     - ``True``
   * - ``compute_1D_intervals``
     - Toggles 1D limits mapping.
     - ``True``, ``False``
     - ``True``
   * - ``compute_2D_intervals``
     - Toggles joint 2D contour scanning and edge tracing.
     - ``True``, ``False``
     - ``True``
   * - ``num_cores``
     - Thread/process count for parallel toy generation. ``null`` maps to max hardware threads.
     - Integer :math:`\geq 0`
     - ``8``
   * - ``verbose``
     - Logging detail level.
     - ``0`` (Silent), ``1``, ``2`` (Debug)
     - ``1``
   * - ``warm_start``
     - Checkpoints interim state to ``.npz`` files to recover from preemptions.
     - ``True``, ``False``
     - ``True``
   * - ``param_names``
     - Labels mapping the physical parameters for plotting outputs. Supports raw LaTeX (e.g. ``[r"$\Phi$", r"$\gamma$"]``).
     - List of strings
     - ``["param1", "param2", ...]``
   * - ``smooth_1d``
     - If True, applies default Gaussian kernel smoothing to 1D limit profiles in plots.
     - ``True``, ``False``
     - ``True``
   * - ``smooth_2d``
     - If True, applies default interpolation smoothing to final 2D contour graphics.
     - ``True``, ``False``
     - ``True``
   * - ``adaptive_toys``
     - Dynamically stops toy generation early once a grid point's accept/reject verdict is statistically settled (99.9% confidence, never before 100 toys). Only for ``strategy`` in ``"scipy"``/``"ultranest"``/``"hybrid"``; no effect for ``"grid"``.
     - ``True``, ``False``
     - ``True``
   * - ``toy_batch_size``
     - Chunk size for submitting/collecting toys from the executor (bounds peak memory for ``likelihood_type="unbinned"``; also the granularity of ``adaptive_toys``' stopping checks). Same total ``n_toys`` either way. Only for ``strategy`` in ``"scipy"``/``"ultranest"``/``"hybrid"``; no effect for ``"grid"``.
     - Integer :math:`> 0`
     - ``200``
   * - ``sparsify_grid``
     - Traces contour perimeters in 2D space to skip resolving deep interior/exterior nodes.
     - ``True``, ``False``
     - ``False``
   * - ``n_restarts``
     - Number of distinct starting points tried per DATA fit (unconditional + every 1D/2D conditional fit), keeping whichever converges to the lowest NLL. Only affects ``strategy="scipy"``'s DATA fit, not the MC toy fits or other strategies.
     - Integer :math:`> 0`
     - ``1``
   * - ``neighbor_seeding``
     - Seeds each 1D/2D scan grid point's DATA fit from an adjacent, already-evaluated grid point's profiled parameters instead of always starting from the bounds midpoint. Only affects ``strategy="scipy"``'s DATA fit, not the MC toy fits or other strategies.
     - ``True``, ``False``
     - ``True``
   * - ``save_log``
     - Pipes output directly to a persistent text log file.
     - ``True``, ``False``
     - ``True``
   * - ``save_directory``
     - Directory path where final results, plots, and checkpoints reside.
     - String (path)
     - ``"output/example_fc_output"``
   * - ``output_file``
     - Prefix for the serialized ``.npz`` and ``.json`` result data structures.
     - String
     - ``"fc_results"``