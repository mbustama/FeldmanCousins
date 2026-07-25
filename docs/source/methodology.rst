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
PyFC evaluates the Negative Log-Likelihood (NLL) for binned configurations:

.. math::

   -\ln \mathcal{L}_{\text{Poisson}} = \sum_{i=1}^{N} \left( \mu_i(\boldsymbol{\theta}) - n_i \ln \mu_i(\boldsymbol{\theta}) \right)~,

**Finite Monte Carlo Correction:**
If enabled, this accounts for limited simulation statistics:

.. math::

   -\ln \mathcal{L}_{\text{FiniteMC}} = \sum_{i=1}^{N} \left( \frac{\mu_i^2}{\sigma_i^2} \ln \left( 1 + \frac{\sigma_i^2}{\mu_i} \right) - n_i \ln \left( \frac{\mu_i}{1 + \sigma_i^2 / \mu_i} \right) \right)~,

**Extended Unbinned Maximum Likelihood**
For unbinned data, PyFC uses exact coordinates :math:`x_j` of the :math:`M` observed events:

.. math::

   -\ln \mathcal{L}_{\text{EUML}} = N_{\text{expected}}(\boldsymbol{\theta}) - \sum_{j=1}^{M} \ln \lambda(x_j | \boldsymbol{\theta})~,