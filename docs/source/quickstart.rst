Quick Start Guide
=================

.. note::
   Because PyFC evaluates abstract :math:`N`-dimensional grids, you **must** supply Python functions instructing the framework how to mathematically map a coordinate in parameter space to your physical expectations.

1. Binned Models
----------------
For a binned analysis, ``S_model`` and ``B_model`` are passed as standard NumPy arrays representing fixed templates. You must write a ``compute_rates_func`` that combines them with your varied parameters to yield the total expected bin counts (:math:`\mu`) and variances (:math:`\sigma^2`).

.. code-block:: python

   import numpy as np
   import json
   from numba import njit
   from pyfc.orchestrator import compute_fc_intervals

   # A) Define the physics mapper function
   @njit(fastmath=True, nogil=True)
   def my_compute_rates_binned(params, S_template, B_template, S_sigma2, B_sigma2):
       """ Maps parameters to expected counts (mu) and simulated variances (sigma2). """
       mu = (params[0] * params[1]) * S_template + params[2] * B_template
       sigma2 = ((params[0] * params[1])**2) * S_sigma2 + (params[2]**2) * B_sigma2
       return mu, sigma2

   # B) Setup Data and Arrays
   grids = [np.linspace(1e-9, 1e-7, 20), np.linspace(2.0, 3.0, 15), np.linspace(0.8, 1.2, 10)]
   S_template = np.array([0.1, 0.5, 2.0, 5.0])
   B_template = np.array([15.0, 5.0, 1.0, 0.1])
   observed_counts = np.array([20, 7, 2, 0])

   with open('fc_config.json', 'r') as f:
       config = json.load(f)

   # C) Execute
   results = compute_fc_intervals(
       data=observed_counts,
       S_model=S_template,
       B_model=B_template,
       grids=grids,
       compute_rates_func=my_compute_rates_binned,
       **config 
   )

2. Unbinned Models
------------------
For unbinned data, ``S_model`` and ``B_model`` are Python functions that evaluate PDFs over exact kinematic coordinates. You must provide a ``compute_rates_func`` to yield the overall expected integral and localized probability density, alongside a ``generate_toy_func`` that handles parametric bootstrapping.

.. code-block:: python

   import numpy as np
   from scipy.stats import norm, expon

   # A) Define PDF structures
   def s_pdf(x): return norm.pdf(x, loc=5.0, scale=1.0)
   def b_pdf(x): return expon.pdf(x, scale=2.0)

   # B) Define the physics mapper function
   def my_compute_rates_unbinned(params, s_probs, b_probs):
       """ Returns total extended integral and unnormalized per-event density. """
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
       S_model=s_pdf,
       B_model=b_pdf,
       grids=grids,
       compute_rates_func=my_compute_rates_unbinned,
       generate_toy_func=my_generate_unbinned_toy,
       **config 
   )