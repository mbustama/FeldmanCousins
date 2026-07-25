Outputs & Checkpointing
=======================

The Checkpoint Engine
---------------------
Feldman-Cousins calculations are highly resource-intensive and often run on shared HPC clusters subject to preemption limits. If your script is interrupted, simply run it again. PyFC will detect the ``checkpoint_fc.npz`` file, verify that your parameter grids match exactly, and seamlessly resume toy generation from the point of interruption.

.. warning::
   **Critical Restart Warning**: While PyFC verifies your grid dimensions on restart, it does *not* rigorously verify hyperparameter modifications. If you abort a run and alter settings like ``strategy``, ``likelihood_type``, or ``n_toys``, you **must** manually delete the ``checkpoint_fc.npz`` file before running again. Otherwise, the system will permanently merge structurally corrupted pseudo-experiment p-values into your finalized thresholds.

Stored Results
--------------
Upon successful completion, outputs are saved to your directory (default: ``output/example_fc_output/``):

* **``fc_results.json``**: Contains structural metadata, evaluated bounds, exact confidence levels, and the global unconditional best-fit coordinate.
* **``fc_results.npz``**: A highly compressed NumPy archive containing exact parameter matrices and boolean masks.

.. list-table:: Available ``.npz`` Keys (Where ``{i}`` and ``{j}`` are 1-based indices)
   :widths: 30 50 20
   :header-rows: 1

   * - Key
     - Description
     - Shape
   * - ``grid_p{i}``
     - 1D parameter space meshgrid evaluating parameter ``i``.
     - ``(N,)``
   * - ``1d_t_data_p{i}``
     - 1D PLR test statistic evaluated on the real data.
     - ``(N,)``
   * - ``1d_accepted_p{i}_{cl}``
     - 1D boolean limits profiling other parameters.
     - ``(N,)``
   * - ``2d_t_critical_p{i}p{j}_{cl}``
     - 2D threshold surfaces for the combination.
     - ``(N, M)``
   * - ``2d_accepted_p{i}p{j}_{cl}``
     - 2D boolean masks (True = inside contour).
     - ``(N, M)``