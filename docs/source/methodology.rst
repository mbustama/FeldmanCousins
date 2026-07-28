Methodology & Strategies
========================

Execution Strategies
--------------------
PyFC provides several levers to optimize computation time versus robustness based on the complexity of your likelihood surface.

* **"scipy" (L-BFGS-B)**: Highly recommended for most applications. It utilizes bounding constraints and analytical approximations of the gradient to find likelihood minima incredibly quickly. Bounds are automatically inferred from your ``grids``.
* **"ultranest"**: Utilizes Nested Sampling. Extremely robust against complex, multi-modal likelihood surfaces. Slower, but guarantees finding the global minimum.
* **"hybrid"**: Uses UltraNest to find the global unconditional best-fit (once), and uses SciPy for the thousands of conditional minimizations during profile scanning.
* **"grid"**: Brute-force scanning over the provided parameter nodes. Safest but computationally restrictive.

.. note::
   **Handling Multi-Dimensional Spaces**: PyFC handles arbitrary :math:`N`-dimensional physics models automatically. It profiles (maximizes the likelihood over) all parameters not explicitly being evaluated in the current 1D or 2D slice.

.. _joint-simplex-constraints:

Handling Joint/Simplex-Constrained Parameters
----------------------------------------------
``bounds_list`` (built automatically from your ``grids``) only expresses independent per-parameter box constraints -- ``param_i`` must lie in ``[lo_i, hi_i]``, with no notion of a relationship between two different parameters. Neither L-BFGS-B/SLSQP nor UltraNest's prior transform have any native concept of a *joint* constraint like a simplex (``a + b <= 1``) or a sphere (``a^2 + b^2 <= 1``).

If your physical model has such a constraint and PyFC isn't told about it, the optimizer's default starting guess (the bounds midpoint) can land in the unphysical region, and a naive ``compute_rates_func`` that just returns a degenerate/zero rate there gives gradient-based optimizers nothing to climb out with. **Do not work around this with a hand-rolled smooth penalty function multiplying your rate function** -- it is fragile to tune and reproduces that same flat-gradient trap, just user-side instead of library-side. PyFC provides two purpose-built mechanisms instead, which can be used together.

**Worked example:** a 6-parameter neutrino flavor-fraction fit where ``f_e`` (index 0) and ``f_mu`` (index 1) are subject to ``f_e + f_mu <= 1``, since the derived third fraction ``f_tau = 1 - f_e - f_mu`` must stay non-negative.

**Mechanism 1 -- bounds_func:** use when the constraint only involves the scan's currently-**fixed** test parameter. It tightens a free parameter's own box bound based on whatever other parameter(s) the current scan step has fixed, so the box itself is always physical and the bounds-midpoint starting guess can never land in forbidden territory:

.. code-block:: python

   def simplex_bounds_func(fixed_values, free_indices, default_bounds_list):
       """f_e (index 0) + f_mu (index 1) <= 1."""
       bounds = [default_bounds_list[i] for i in free_indices]
       if 0 in fixed_values and 1 in free_indices:      # f_e fixed, f_mu free
           j = free_indices.index(1)
           lo, hi = bounds[j]
           bounds[j] = (lo, min(hi, 1.0 - fixed_values[0]))
       if 1 in fixed_values and 0 in free_indices:       # f_mu fixed, f_e free
           j = free_indices.index(0)
           lo, hi = bounds[j]
           bounds[j] = (lo, min(hi, 1.0 - fixed_values[1]))
       return bounds

   results, fig = compute_fc_intervals(
       data=data, grids=grids,
       compute_rates_func=my_rates_func,
       bounds_func=simplex_bounds_func,
       # ... other arguments
   )

``bounds_func`` is called as ``bounds_func(fixed_values, free_indices, default_bounds_list)``: ``fixed_values`` is ``{param_index: test_value}`` for whichever parameter(s) the current scan step has fixed (an empty dict during the unconditional fit), ``free_indices`` are the indices being optimized, and ``default_bounds_list`` is the full, untouched ``bounds_list``. Return one ``(lo, hi)`` tuple per entry of ``free_indices``.

*Limitation:* ``bounds_func`` cannot express a constraint between two parameters that are **both** free at the same time (e.g. if ``f_e`` and ``f_mu`` are never the scan's fixed test parameter, such as while profiling over two unrelated parameters). That case needs mechanism 2.

**Mechanism 2 -- constraints:** use for constraints among multiple **simultaneously-free** nuisance parameters. Pass a list of ``scipy.optimize.LinearConstraint``/``NonlinearConstraint`` objects, expressed in the full parameter-vector space -- PyFC projects them down to the free-parameter subspace internally, substituting whichever value(s) the scan currently has fixed:

.. code-block:: python

   from scipy.optimize import LinearConstraint

   # f_e + f_mu <= 1, i.e. 1*f_e + 1*f_mu + 0*(everything else) <= 1
   simplex_constraint = LinearConstraint(
       A=[[1, 1, 0, 0, 0, 0]],   # one row per constraint, one column per full parameter
       lb=-np.inf, ub=1.0,
   )

   results, fig = compute_fc_intervals(
       data=data, grids=grids,
       compute_rates_func=my_rates_func,
       constraints=[simplex_constraint],
       # ... other arguments
   )

Supplying ``constraints`` automatically switches the scipy method from ``L-BFGS-B`` to ``SLSQP`` (the only two of PyFC's methods that support ``constraints`` together with bounds; ``trust-constr`` is also available via ``scipy_method="trust-constr"`` for better-conditioned but slower fits). For the UltraNest path, since nested sampling doesn't use gradients, a violated constraint simply makes that point's log-likelihood a very large negative number -- no method switch needed.

**Which one should I use?** If the constraint only ever couples the scan's currently-fixed test parameter(s) to a free nuisance parameter, ``bounds_func`` is cheaper (the optimizer never has to work near a boundary at all). If your scan fixes some *other* parameter and profiles ``f_e`` and ``f_mu`` together as simultaneously-free nuisance parameters, only ``constraints`` can express that. The two mechanisms compose and can be passed together on the same fit call.

See the README section "Handling Joint/Simplex-Constrained Parameters" for further discussion.

Statistical Mathematics
-----------------------

The code executes the methodology prescribed by Feldman and Cousins :cite:p:`Feldman1997qc` to solve the empty-set problem near physical boundaries. For complex kinematic models with limited statistics, we rely on the continuous Poisson-Gamma mixture formulations detailed by Argüelles et al. :cite:p:`Arguelles2019izp`.

**The Profile Likelihood Ratio**
For a general parameter vector divided into parameters of interest :math:`\boldsymbol{\theta}` and nuisance parameters :math:`\boldsymbol{\nu}`, we construct the Profile Likelihood Ratio (PLR) test statistic :math:`t`:

.. math::

   t_{\text{data}}(\boldsymbol{\theta}) = -2 \ln \frac{\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})}{\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})}~,

where:

* :math:`\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})` is the unconditional Maximum Likelihood Estimate (MLE).
* :math:`\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})` is the conditional MLE, evaluated at a fixed point :math:`\boldsymbol{\theta}_{\text{test}}`.

**Binned Likelihood**
PyFC evaluates the saturated (Baker-Cousins) form of the Negative Log-Likelihood for
binned configurations -- the likelihood ratio relative to the saturated model
(:math:`\mu_i = n_i`) -- dropping the data-only constant :math:`\ln(n_i!)` term:

.. math::

   -\ln \mathcal{L}_{\text{Poisson}} = \sum_{i=1}^{N} \left( \mu_i(\boldsymbol{\theta}) - n_i + n_i \ln\frac{n_i}{\mu_i(\boldsymbol{\theta})} \right)~,

PyFC's implementation carries an extra overall factor of 2 relative to this
expression -- it computes :math:`-2\ln \mathcal{L}_{\text{Poisson}}` directly,
matching the Baker-Cousins/Wilks convention used for the PLR test statistic
:math:`t` itself (see above), so no separate doubling is needed when forming
:math:`t = \text{NLL}_{\text{cond}} - \text{NLL}_{\text{uncond}}`.

**Finite Monte Carlo Correction:**
If enabled, this accounts for limited simulation statistics using the marginalized
effective likelihood :math:`\mathcal{L}_{\text{Eff}}` derived by Argüelles, Schneider
& Yuan :cite:p:`Arguelles2019izp`, with

.. math::

   \alpha_i = \frac{\mu_i^2}{\sigma_i^2} + 1~, \qquad \beta_i = \frac{\mu_i}{\sigma_i^2}~,

.. math::

   \ln \mathcal{L}_{\text{FiniteMC}} = \sum_{i=1}^{N} \left[ \alpha_i \ln \beta_i + \ln \Gamma(n_i + \alpha_i) - (n_i + \alpha_i) \ln(1 + \beta_i) - \ln \Gamma(\alpha_i) \right]~,

with :math:`\text{NLL}_{\text{FiniteMC}} = -2 \ln \mathcal{L}_{\text{FiniteMC}}`. The
"+1" in :math:`\alpha_i` is deliberate: it is what distinguishes this (recommended,
best-coverage) parameterization from the paper's alternative moment-matching-only
choice (:math:`\alpha_i = \mu_i^2/\sigma_i^2`, no "+1"), which the paper's own
coverage tests show performs worse. The data-only constant :math:`\ln(n_i!)` term
is dropped, matching the same convention used for the EUML formula below.

**Extended Unbinned Maximum Likelihood**
For unbinned data, PyFC uses exact coordinates :math:`x_j` of the :math:`M` observed events:

.. math::

   -\ln \mathcal{L}_{\text{EUML}} = N_{\text{expected}}(\boldsymbol{\theta}) - \sum_{j=1}^{M} \ln \lambda(x_j | \boldsymbol{\theta})~,