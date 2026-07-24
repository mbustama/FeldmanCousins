# PyFC: A Python Framework for Feldman-Cousins Confidence Intervals

**PyFC** is a rigorous, high-performance frequentist statistical analysis framework for Python. It automates the construction of classical confidence intervals and regions using the unified Feldman-Cousins approach, seamlessly transitioning between one-sided upper limits and two-sided bounds while guaranteeing exact frequentist coverage. 

Designed for high-energy physics, astrophysics, and general parametric modeling, PyFC handles both binned (histogram) and unbinned (event-by-event) data, integrates multiple optimization strategies (SciPy, UltraNest, Grid), and utilizes highly parallelized Monte Carlo pseudo-experiment generation.

## Salient Features
* **Unified Binned and Unbinned Analysis**: Natively supports Poisson binned data and Extended Unbinned Maximum Likelihood (EUML) formulations.
* **Exact Coverage**: Empirically derives the Profile Likelihood Ratio (PLR) test statistic distribution via dynamically generated Monte Carlo pseudo-experiments (toys).
* **Advanced Optimizers**: Supports gradient-based L-BFGS-B (SciPy), nested sampling (UltraNest), and brute-force grid scanning.
* **Massive Parallelization**: GIL-bypassing via NumPy/Numba C-extensions for binned thread pooling, and `ProcessPoolExecutor` for unbinned continuous functions.
* **Finite Monte Carlo Corrections**: Implements the Beeston-Barlow technique, modeling templates as Poisson-Gamma mixtures to account for finite simulation statistics.
* **Dynamic 2D Sparsification**: Employs Bivariate Spline interpolation and edge-tracing algorithms to skip unnecessary toy generation inside or outside 2D contours, radically reducing computational overhead.
* **State Checkpointing**: "Warm start" capability saves binary state matrices periodically to prevent data loss on cluster preemptions.
* **Interactive Configuration**: Built-in CLI for generating serialized JSON experiment configurations.

---

## Table of Contents
1. [Installation](#installation)
2. [File Tree](#file-tree)
3. [Quick Start Guide](#quick-start-guide)
    * [Defining Physics Models](#defining-physics-models)
    * [Method 1: Direct Python Execution](#method-1-direct-python-execution)
    * [Method 2: Execution via JSON Config](#method-2-execution-via-json-config)
4. [Statistical Methodology & Mathematics](#statistical-methodology--mathematics)
    * [The Profile Likelihood Ratio](#the-profile-likelihood-ratio)
    * [Binned Likelihood (Poisson & Finite MC)](#binned-likelihood)
    * [Extended Unbinned Maximum Likelihood](#unbinned-likelihood)
    * [Empirical Coverage (Toy Generation)](#empirical-coverage-and-toy-generation)
    * [2D Contour Sparsification](#2d-contour-sparsification)
5. [Common Recipes and Questions](#common-recipes-and-questions)
6. [How to Cite](#how-to-cite)

---

## Installation

Clone the repository and install via `pip` to ensure all dependencies (NumPy, SciPy, Numba, UltraNest, Corner) are resolved.

    git clone https://github.com/mbustama/FeldmanCousins.git
    cd FeldmanCousins
    pip install -e .

---

## File Tree

* `binned.py` - Core mathematical formulations, NLL definitions, and parallelized grid optimizations for binned (Poisson) data.
* `config.py` - CLI argument parsing and default configuration matrix management.
* `generate_config.py` - Interactive command-line interface for generating robust `fc_config.json` files.
* `optimizers.py` - Wrapper functions mapping likelihood surfaces to SciPy (L-BFGS-B) and UltraNest minimizers.
* `orchestrator.py` - The central execution hub linking models, data, and toy generators to build the FC contours.
* `plotting.py` - Visualization tools for rendering 1D parameter profiles and 2D joint confidence corners.
* `toys.py` - Multiprocessing and multithreading managers for continuous and discrete Monte Carlo pseudo-experiment generation.
* `unbinned.py` - Core mathematical formulations and grid optimizers for the Extended Unbinned Maximum Likelihood.

---

## Quick Start Guide

### Defining Physics Models
The way you define your physics model depends entirely on whether your analysis is **binned** or **unbinned**.

**1. Binned Models**
For binned data, your structural templates should evaluate to 1D arrays of expected rates (e.g., event counts per bin).
    import numpy as np

    # In the binned framework, models can be static templates or functions of parameters.
    # E.g., a simple scaling of predefined Monte Carlo templates:
    S_template = np.array([0.1, 0.5, 2.0, 5.0])
    B_template = np.array([15.0, 5.0, 1.0, 0.1])

**2. Unbinned Models**
For unbinned data, you must provide normalized probability density functions (PDFs) that can evaluate an array of kinematic coordinates (events) and return the probability density at each point. 
    from scipy.stats import norm, expon

    def S_model_pdf(x):
        """Signal PDF: returns the probability density evaluated at array x."""
        return norm.pdf(x, loc=5.0, scale=1.0)

    def B_model_pdf(x):
        """Background PDF: returns the probability density evaluated at array x."""
        return expon.pdf(x, scale=2.0)
*Note: Ensure your unbinned PDF functions can accept vector (NumPy array) inputs for computational efficiency.*

### Method 1: Direct Python Execution
You can import the `compute_fc_intervals` orchestrator directly and pass explicit parameters.

    import numpy as np
    from pyfc.orchestrator import compute_fc_intervals

    # 1. Define physical grids (the parameter space to scan)
    # Let's say param1 is Signal Norm, param2 is Background Norm
    grids = [
        np.linspace(0.0, 5.0, 20), # Param 1
        np.linspace(0.8, 1.2, 10)  # Param 2 (e.g., a constrained nuisance parameter)
    ]

    # 2. Mock Data
    observed_counts = np.array([10, 8, 3, 2])
    S_template = np.array([1.0, 2.0, 3.0, 4.0])
    B_template = np.array([10.0, 5.0, 1.0, 0.1])

    # 3. Execute Feldman-Cousins
    results = compute_fc_intervals(
        data=observed_counts,
        S_model=S_template,
        B_model=B_template,
        grids=grids,
        cl=[0.68, 0.90],
        n_toys=1000,
        strategy="scipy",
        likelihood_type="binned",
        compute_1D_intervals=True,
        compute_2D_intervals=True,
        param_names=["Signal Norm", "Background Norm"]
    )

### Method 2: Execution via JSON Config
You can generate a configuration file via the CLI (`python -m pyfc.generate_config`) and ingest it dynamically.

    import json
    from pyfc.orchestrator import compute_fc_intervals

    # Load the generated JSON
    with open('fc_config.json', 'r') as f:
        config = json.load(f)

    # The orchestrator expects kwargs, so we unpack the dictionary using **
    results = compute_fc_intervals(
        data=observed_counts,
        S_model=S_template,
        B_model=B_template,
        grids=grids,
        **config  # Unpacks n_toys, strategy, num_cores, likelihood_type, etc.
    )

---

## Statistical Methodology & Mathematics

PyFC implements exact classical frequentist intervals. The code executes in excruciating detail the methodology prescribed by Gary Feldman and Robert Cousins (1998) to solve the empty-set problem near physical boundaries.

### The Profile Likelihood Ratio
For a general parameter vector divided into parameters of interest $\boldsymbol{\theta}$ and nuisance parameters $\boldsymbol{\nu}$, we construct the Profile Likelihood Ratio (PLR) test statistic $t$:

$$ t_{\text{data}}(\boldsymbol{\theta}) = -2 \ln \frac{\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})}{\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})} $$

Where:
* $\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})$ is the unconditional Maximum Likelihood Estimate (MLE) over the entire allowed parameter space.
* $\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})$ is the conditional MLE, evaluated at a fixed point $\boldsymbol{\theta}_{\text{test}}$, while profiling (maximizing) out the nuisance parameters $\boldsymbol{\nu}$.

By construction, $t \geq 0$. Lower values indicate excellent agreement between the data and the test hypothesis.

### Binned Likelihood
For binned configurations, the likelihood is the product of independent Poisson probabilities across $N$ bins. Dropping the data-dependent factorial constant, PyFC evaluates the Negative Log-Likelihood (NLL):

$$ - \ln \mathcal{L}_{\text{Poisson}} = \sum_{i=1}^{N} \left( \mu_i(\boldsymbol{\theta}) - n_i \ln \mu_i(\boldsymbol{\theta}) \right) $$

**Finite Monte Carlo Correction (Beeston-Barlow):**
If `use_finite_mc_correction_binned` is True, PyFC acknowledges that the template $\mu_i$ is derived from finite simulation statistics. The pure Poisson distribution is convoluted with a Gamma prior, shifting the likelihood to a Negative Binomial distribution:

$$ - \ln \mathcal{L}_{\text{FiniteMC}} = \sum_{i=1}^{N} \left( \frac{\mu_i^2}{\sigma_i^2} \ln \left( 1 + \frac{\sigma_i^2}{\mu_i} \right) - n_i \ln \left( \frac{\mu_i}{1 + \sigma_i^2 / \mu_i} \right) \right) $$
where $\sigma_i^2$ is the variance (sum of squared MC weights) in bin $i$.

### Extended Unbinned Maximum Likelihood
When binning causes unacceptable information loss (e.g., highly complex kinematics with low event counts), PyFC evaluates the unbinned likelihood. Instead of bin counts, it uses the exact coordinates $x_j$ of the $M$ observed events.

$$ - \ln \mathcal{L}_{\text{EUML}} = N_{\text{expected}}(\boldsymbol{\theta}) - \sum_{j=1}^{M} \ln \lambda(x_j | \boldsymbol{\theta}) $$
where $N_{\text{expected}}$ is the integral of the total rate, and $\lambda(x_j)$ is the non-normalized rate evaluated strictly at the properties of event $j$.

### Empirical Coverage and Toy Generation
Wilks' theorem dictates that $t$ should follow a $\chi^2$ distribution. However, near physical boundaries (e.g., signal $\geq 0$), this theorem explicitly breaks down. PyFC restores exact coverage by deriving the distribution of $t$ empirically at *every* grid point.

1. **Null Hypothesis:** For a fixed test point $\boldsymbol{\theta}_{\text{test}}$, locate the best-fit nuisance parameters $\hat{\hat{\boldsymbol{\nu}}}$.
2. **Pseudo-experiments (Toys):** Generate $N_{\text{toys}}$ fake datasets assuming $(\boldsymbol{\theta}_{\text{test}}, \hat{\hat{\boldsymbol{\nu}}})$ is the true state of nature.
    * *Binned:* Draw from $N$ Poisson distributions.
    * *Unbinned:* Bootstrap discrete events from large user-supplied MC prior arrays (`S_mc_pool`, `B_mc_pool`).
3. **Refit:** Unconditionally and conditionally fit every single toy to calculate $t_{\text{toy}}$.
4. **Critical Value:** Sort the $t_{\text{toy}}$ values. The threshold $t_{\text{critical}}$ for a Confidence Level $\alpha$ (e.g., 0.90) is the value at the $\alpha$-quantile. 

If $t_{\text{data}} \leq t_{\text{critical}}$, the point $\boldsymbol{\theta}_{\text{test}}$ is accepted into the confidence interval.

### 2D Contour Sparsification
To map a 2D contour, generating toys at every point on an $N \times N$ grid is computationally punishing. PyFC implements a heuristic sparsification algorithm:
1. Calculates $t_{\text{data}}$ everywhere.
2. Selects a sparse sub-grid (e.g., every 5th node) and generates complete toy distributions to find exact $t_{\text{critical}}$ values.
3. Fits a `RectBivariateSpline` across the sub-grid to approximate the $t_{\text{critical}}$ surface.
4. Locates the decision boundary (where $t_{\text{data}} \approx t_{\text{critical}}$).
5. Traces this perimeter and evaluates exact toys *only* at the high-resolution grid nodes directly adjacent to the boundary.

---

## Common Recipes and Questions

**Q: How many toys should I use?**
A: For a 68% CL interval (1-sigma), 500-1000 toys are often sufficient. For a 90% or 95% limit, 2000-5000 toys are required to smoothly resolve the tail of the test statistic distribution. Ensure `n_toys` is large enough that $N_{\text{toys}} \times (1 - \alpha) \gg 1$.

**Q: Which optimizer strategy should I choose?**
A: 
*   `"scipy"` (L-BFGS-B): The default. Extremely fast, utilizing gradient information. Best for smooth likelihood surfaces.
*   `"ultranest"`: Slower but virtually immune to local minima. Use this if your parameters are highly correlated or your likelihood surface has multiple distinct valleys.
*   `"grid"`: Brute-force scanning. The safest but slowest option. Often used strictly for 1D projections.

**Q: I have a parameter that represents a systematic uncertainty. How do I profile it?**
A: Simply pass a `np.linspace()` grid for that parameter into the `grids` list. PyFC automatically profiles (maximizes) all parameters in the `grids` list that are not currently fixed as the specific parameters of interest during 1D or 2D conditional scanning.

**Q: What happens if my job gets killed on the cluster?**
A: Ensure `warm_start=True`. PyFC frequently saves the computational state to a `checkpoint_fc.npz` matrix. If you restart the script with the exact same grid geometry, it will load the checkpoint and resume toy generation exactly where it left off.

---

## How to Cite

If you utilize PyFC in your academic work or scientific publications, please cite the framework and link to the source repository:

Bustamante, M. (2026). *PyFC: A Python Framework for Feldman-Cousins Confidence Intervals*. GitHub Repository: https://github.com/mbustama/FeldmanCousins.

*(If applicable, please also cite the original methodology paper: Feldman, G. J., & Cousins, R. D. (1998). Unified approach to the classical statistical analysis of small signals. Physical Review D, 57(7), 3873.)*