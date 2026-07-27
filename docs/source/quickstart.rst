Quick Start Guide
=================

.. important::
   The examples below use the API introduced in v0.10.0 (this branch), which is not yet
   published to PyPI as of this writing -- ``pip install PyFeldmanCousins`` currently installs
   an older release with a different ``compute_fc_intervals`` signature (``S_model``/
   ``B_model`` instead of ``pdf_components``). If these examples raise ``TypeError:
   compute_fc_intervals() missing ... 'S_model' and 'B_model'`` or ``got an unexpected keyword
   argument 'pdf_components'``, your installed version predates this change -- install this
   branch directly instead (see :doc:`installation`'s "Local Installation from Source" option)
   until a new PyPI release ships.

.. note::
   Because PyFC evaluates abstract :math:`N`-dimensional grids, you **must** supply Python functions instructing the framework how to mathematically map a coordinate in parameter space to your physical expectations.

.. note::
   If two or more of your parameters are subject to a **joint** constraint (e.g. a simplex like ``a + b <= 1``) rather than independent per-parameter bounds, see :ref:`joint-simplex-constraints` for the ``bounds_func``/``constraints`` mechanisms before reaching for a hand-rolled penalty function.

1. Binned Models
----------------
For a binned analysis, fixed templates are ordinary NumPy arrays referenced via closure (module-global or nested-function capture) from your ``compute_rates_func`` -- they are no longer passed in as function arguments. You must write a ``compute_rates_func`` that combines them with your varied parameters to yield the total expected bin counts (:math:`\mu`) and variances (:math:`\sigma^2`). ``data``/``mu``/``sigma2`` may be **any shape**, not just 1D -- a genuine 2D ``(E, cos_theta)`` histogram is fully supported and evaluated directly, with no need to flatten it yourself first.

.. code-block:: python

   import numpy as np
   import json
   from numba import njit
   from pyfc.orchestrator import compute_fc_intervals

   # A) Fixed templates as module-level constants, referenced via closure
   S_template = np.array([0.1, 0.5, 2.0, 5.0])
   B_template = np.array([15.0, 5.0, 1.0, 0.1])

   # B) Define the physics mapper function
   @njit(fastmath=True, nogil=True)
   def my_compute_rates_binned(params, S_sumw2, B_sumw2):
       """ Maps parameters to expected counts (mu) and simulated variances (sigma2). """
       mu = (params[0] * params[1]) * S_template + params[2] * B_template
       sigma2 = ((params[0] * params[1])**2) * S_sumw2 + (params[2]**2) * B_sumw2
       return mu, sigma2

   # C) Setup Data and Grids
   grids = [np.linspace(1e-9, 1e-7, 20), np.linspace(2.0, 3.0, 15), np.linspace(0.8, 1.2, 10)]
   observed_counts = np.array([20, 7, 2, 0])

   with open('fc_config.json', 'r') as f:
       config = json.load(f)

   # D) Execute
   results = compute_fc_intervals(
       data=observed_counts,
       grids=grids,
       compute_rates_func=my_compute_rates_binned,
       **config
   )

2. Unbinned Models
------------------
For unbinned data, ``pdf_components`` is a **list** of Python functions that each evaluate a PDF over exact kinematic coordinates (signal, background, or any number of further components -- no longer limited to exactly two). Each is evaluated exactly once per fit and the resulting arrays are reused across every subsequent NLL evaluation in that fit. You must provide a ``compute_rates_func`` that consumes the pre-evaluated ``probs`` list to yield the overall expected integral and localized probability density, alongside a ``generate_toy_func`` that handles parametric bootstrapping.

.. code-block:: python

   import numpy as np
   from scipy.stats import norm, expon

   # A) Define PDF structures
   def s_pdf(x): return norm.pdf(x, loc=5.0, scale=1.0)
   def b_pdf(x): return expon.pdf(x, scale=2.0)

   # B) Define the physics mapper function
   def my_compute_rates_unbinned(params, probs):
       """ Returns total extended integral and unnormalized per-event density. """
       s_probs, b_probs = probs[0], probs[1]
       expected_total = params[0] * params[1] + params[2]
       if len(s_probs) == 0 and len(b_probs) == 0:
           return expected_total, np.array([])
       p_events = params[0] * params[1] * s_probs + params[2] * b_probs
       return expected_total, p_events

   # C) Define the toy generator
   def my_generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool):
       """ Handles generating mock event geometries based on physical parameters. """
       n_sig = np.random.poisson(true_params[0] * true_params[1])
       n_bkg = np.random.poisson(true_params[2])
       parts = []
       if n_sig > 0: parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
       if n_bkg > 0: parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
       return np.concatenate(parts) if parts else np.array([])

   # D) Execute
   results = compute_fc_intervals(
       data=observed_events,
       grids=grids,
       compute_rates_func=my_compute_rates_unbinned,
       generate_toy_func=my_generate_unbinned_toy,
       pdf_components=[s_pdf, b_pdf],
       **config
   )

.. note::
   :func:`np.random.choice` (used above) only works for a **1D** MC pool.
   If your events are N-D feature vectors (``mc_pool.shape == (n_events,
   n_features)``), resample by index instead:
   ``idx = np.random.choice(len(mc_pool), size=n, replace=True); toy_events
   = mc_pool[idx]``. Silently applying ``np.random.choice`` directly to an
   N-D array samples along the wrong axis and does not raise an error.