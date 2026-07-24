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
3. [Configuration & generate_config.py](#configuration--generate_configpy)
4. [Quick Start Guide](#quick-start-guide)
5. [Configuration Parameters (CLI / JSON)](#configuration-parameters-cli--json)
6. [Execution Strategies & Algorithmic Optimizations](#execution-strategies--algorithmic-optimizations)
7. [Statistical Methodology & Mathematics](#statistical-methodology--mathematics)
8. [Outputs, Plots, and Checkpointing](#outputs-plots-and-checkpointing)
9. [Common Recipes and Questions](#common-recipes-and-questions)
10. [How to Cite](#how-to-cite)

---

## Installation

Clone the repository and install via `pip` to ensure all dependencies (NumPy, SciPy, Numba, UltraNest, Corner) are resolved.

    git clone https://github.com/mbustama/FeldmanCousins.git
    cd FeldmanCousins
    pip install -e .

---

## File Tree

The project structure is organized modularly to separate analytical likelihood math from plotting and execution loops:

    FeldmanCousins/
    ├── pyproject.toml         # Build system and dependency specifications
    ├── README.md              # Project documentation
    └── src/
        └── pyfc/
            ├── __init__.py          # Package initialization and metadata
            ├── binned.py            # Binned NLL math and Numba-accelerated optimizers
            ├── config.py            # CLI argument definitions and JSON parsing
            ├── generate_config.py   # Interactive CLI wizard for creating fc_config.json
            ├── optimizers.py        # Wrapper functions mapping objective functions to SciPy/UltraNest
            ├── orchestrator.py      # The main pipeline executing the Feldman-Cousins algorithm
            ├── plotting.py          # Visualization suite for 1D profiles and 2D contours
            ├── toys.py              # Multiprocessing engines for MC pseudo-experiment generation
            └── unbinned.py          # Extended Unbinned Maximum Likelihood (EUML) formulations

---

## Configuration & generate_config.py

PyFC provides an interactive command-line tool to help users build their analysis configuration files securely and without typos. 

To generate a configuration file, simply run:
    
    python -m pyfc.generate_config

The script will prompt you with questions regarding your likelihood type, number of toys, parallelization preferences, and smoothing options. It validates your inputs and writes a file (by default, `fc_config.json`) to your current working directory.

**Example `fc_config.json` with Default Values:**

    {
        "likelihood_type": "binned",
        "cl": [
            0.68,
            0.9
        ],
        "n_toys": 2000,
        "strategy": "scipy",
        "num_cores": 0,
        "verbose": 1,
        "adaptive_toys": true,
        "toy_batch_size": 200,
        "sparsify_grid": true,
        "warm_start": true,
        "output_file": "fc_results",
        "save_log": false,
        "save_directory": "fc_output",
        "use_finite_mc_correction_binned": true,
        "compute_1D_intervals": true,
        "compute_2D_intervals": true,
        "param_names": [
            "param1",
            "param2"
        ],
        "smooth_1d": false,
        "smooth_2d": false
    }

---

## Quick Start Guide

*Note: The code snippets below are examples of user-defined cases. You will need to substitute the `S_model_func` and `B_model_func` functions, as well as the input grids, with your own specific physical models and parameter configurations.*

### Defining Physics Models
The way you define your physics model depends entirely on whether your analysis is **binned** or **unbinned**.

**1. Binned Models**
For a phenomenological analysis, your models should be parameterized functions (callables) that map your physics parameters (e.g., cross-sections, fluxes) to 1D numpy arrays representing expected event counts per bin.

    import numpy as np

    def S_model_func(flux_norm, spectral_index):
        """
        Calculates the expected signal array across energy bins.
        These parameters will map directly to the parameter grids defined in the orchestrator.
        """
        energy_bins = np.array([10.0, 100.0, 1000.0, 10000.0])
        return flux_norm * (energy_bins ** -spectral_index)

    def B_model_func(bg_norm):
        """
        Calculates the expected background array based on a scaling parameter.
        """
        nominal_atmospheric_bg = np.array([15.0, 5.0, 1.0, 0.1])
        return bg_norm * nominal_atmospheric_bg

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

### Execution via JSON Config
Once you have your models defined and your config generated, you can pass them dynamically into the central orchestrator:

    import json
    import numpy as np
    from pyfc.orchestrator import compute_fc_intervals

    # 1. Define physical grids (the parameter space to scan)
    # These grids correspond sequentially to the inputs of S_model_func and B_model_func
    grids = [
        np.linspace(1e-9, 1e-7, 20), # Param 1: flux_norm
        np.linspace(2.0, 3.0, 15),   # Param 2: spectral_index
        np.linspace(0.8, 1.2, 10)    # Param 3: bg_norm (nuisance parameter)
    ]

    # 2. Mock Data
    observed_counts = np.array([20, 7, 2, 0])

    # 3. Load the generated JSON configuration
    with open('fc_config.json', 'r') as f:
        config = json.load(f)

    # 4. Execute Feldman-Cousins (unpacking the config dictionary via **)
    results = compute_fc_intervals(
        data=observed_counts,
        S_model=S_model_func,
        B_model=B_model_func,
        grids=grids,
        **config 
    )

---

## Configuration Parameters (CLI / JSON)

| Parameter | Description | Allowed Values | Default |
| :--- | :--- | :--- | :--- |
| `likelihood_type` | Evaluates models via Poisson bins or Extended Unbinned Maximum Likelihood. | `"binned"`, `"unbinned"` | `"binned"` |
| `cl` | Confidence Levels determining exact frequentist coverage integration targets. | List of floats `(0.0, 1.0)` | `[0.68, 0.90]` |
| `n_toys` | Monte Carlo pseudo-experiments generated per parameter space point. | Integer `> 0` | `2000` |
| `strategy` | Optimizer used for finding global and conditional likelihood minima. | `"scipy"`, `"ultranest"`, `"hybrid"`, `"grid"` | `"scipy"` |
| `use_finite_mc_correction_binned` | Shifts Poisson likelihood to a Negative Binomial to account for finite simulation stats. | `True`, `False` | `True` |
| `compute_1D_intervals` | Toggles profiling nuisance parameters for 1D parameter limits. | `True`, `False` | `True` |
| `compute_2D_intervals` | Toggles joint 2D contour scanning and edge tracing. | `True`, `False` | `True` |
| `num_cores` | Thread/process count for parallel toy generation. `0` maps to max hardware threads. | Integer `>= 0` | `0` (Max) |
| `verbose` | Logging detail level. | `0` (Silent), `1`, `2` (Debug) | `1` |
| `warm_start` | Checkpoints interim state to `.npz` files to recover from preemptions. | `True`, `False` | `True` |
| `param_names` | Labels mapping the physical parameters for plotting outputs. | List of strings | `["param1", "param2", ...]` |
| `smooth_1d` | Applies Gaussian kernel smoothing to 1D limit profiles in plots. | `True`, `False` | `False` |
| `smooth_2d` | Applies interpolation smoothing to final 2D contour graphics. | `True`, `False` | `False` |
| `adaptive_toys` | Dynamically skips toy generation if a point is definitively excluded. | `True`, `False` | `True` |
| `toy_batch_size` | Chunk size for batched array generation (optimizes memory/speed). | Integer `> 0` | `200` |
| `sparsify_grid` | Traces contour perimeters in 2D space to skip resolving deep interior/exterior nodes. | `True`, `False` | `True` |
| `save_log` | Pipes output directly to a persistent text log file. | `True`, `False` | `False` |
| `save_directory` | Directory path where final results, plots, and checkpoints reside. | String (path) | `"fc_output"` |
| `output_file` | Prefix for the serialized `.npz` and `.json` result data structures. | String | `"fc_results"` |

---

## Execution Strategies & Algorithmic Optimizations

PyFC provides several comprehensive levers to optimize computation time versus robustness based on the complexity of your likelihood surface.

### Optimizer Strategy (`strategy`)
*   **`"scipy"` (L-BFGS-B)**: Highly recommended for most physics applications. It utilizes bounding constraints and analytical approximations of the gradient to find likelihood minima incredibly quickly. It assumes a relatively smooth parameter space.
*   **`"ultranest"`**: Utilizes Nested Sampling. Extremely robust against complex, multi-modal likelihood surfaces where standard gradient minimizers get trapped in local minima. It is much slower than SciPy but guarantees finding the global minimum.
*   **`"hybrid"`**: A balanced approach. Uses UltraNest to find the global unconditional best-fit (which happens only once), and uses SciPy for the thousands of conditional minimizations during the profile scanning and MC toy fitting.
*   **`"grid"`**: Brute-force scanning over the provided parameter nodes. Safest but computationally restrictive. Scales poorly with $N > 2$ dimensions.

### Finite MC Corrections (`use_finite_mc_correction_binned`)
When Monte Carlo templates are generated from limited statistics, treating the expected counts $\mu_i$ as absolute fixed truths leads to overconfidence. Enabling this flag triggers the continuous Poisson-Gamma mixture model formulation (see Mathematics section below) to strictly penalize the likelihood based on the simulated variance ($\sigma_i^2$) in each bin.

### Contour Edge Tracing (`sparsify_grid`)
Calculating $N_{\text{toys}}$ for every node in a $100 \times 100$ 2D grid is computationally wasteful. Enabling `sparsify_grid` activates a heuristic algorithm:
1. Calculates $t_{\text{data}}$ everywhere.
2. Selects a sparse, widely spaced sub-grid and generates complete toy distributions to find exact $t_{\text{critical}}$ values at those sparse nodes.
3. Fits a Scipy `RectBivariateSpline` to interpolate the critical threshold surface.
4. Locates the decision boundary (the "edge" of the contour where $t_{\text{data}} \approx t_{\text{critical}}$).
5. Only evaluates the expensive MC toys on the specific high-resolution cells lying strictly on this perimeter to perfect the contour edge, drastically cutting runtime.

---

## Statistical Methodology & Mathematics

PyFC implements exact classical frequentist intervals. The code executes the methodology prescribed by Gary Feldman and Robert Cousins (1998) to solve the empty-set problem near physical boundaries.

### The Profile Likelihood Ratio
For a general parameter vector divided into parameters of interest $\boldsymbol{\theta}$ and nuisance parameters $\boldsymbol{\nu}$, we construct the Profile Likelihood Ratio (PLR) test statistic $t$:

$$t_{\text{data}}(\boldsymbol{\theta})=-2\ln\frac{\mathcal{L}(\boldsymbol{\theta},\hat{\hat{\boldsymbol{\nu}}}|\text{data})}{\mathcal{L}(\hat{\boldsymbol{\theta}},\hat{\boldsymbol{\nu}}|\text{data})}$$

Where:
* $\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})$ is the unconditional Maximum Likelihood Estimate (MLE) over the entire allowed parameter space.
* $\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})$ is the conditional MLE, evaluated at a fixed point $\boldsymbol{\theta}_{\text{test}}$, while profiling (maximizing) out the nuisance parameters $\boldsymbol{\nu}$.

By construction, $t \geq 0$. Lower values indicate excellent agreement between the data and the test hypothesis.

### Binned Likelihood
For binned configurations, the likelihood is the product of independent Poisson probabilities across $N$ bins. Dropping the data-dependent factorial constant, PyFC evaluates the Negative Log-Likelihood (NLL):

$$-\ln\mathcal{L}_{\text{Poisson}}=\sum_{i=1}^{N}\left(\mu_i(\boldsymbol{\theta})-n_i\ln\mu_i(\boldsymbol{\theta})\right)$$

**Finite Monte Carlo Correction:**
If `use_finite_mc_correction_binned` is True, the pure Poisson distribution is convoluted with a Gamma prior, shifting the likelihood to a Negative Binomial distribution:

$$-\ln\mathcal{L}_{\text{FiniteMC}}=\sum_{i=1}^{N}\left(\frac{\mu_i^2}{\sigma_i^2}\ln\left(1+\frac{\sigma_i^2}{\mu_i}\right)-n_i\ln\left(\frac{\mu_i}{1+\sigma_i^2/\mu_i}\right)\right)$$
where $\sigma_i^2$ is the variance (sum of squared MC weights) in bin $i$.

### Extended Unbinned Maximum Likelihood
When binning causes unacceptable information loss (e.g., highly complex kinematics with low event counts), PyFC evaluates the unbinned likelihood. Instead of bin counts, it uses the exact coordinates $x_j$ of the $M$ observed events.

$$-\ln\mathcal{L}_{\text{EUML}}=N_{\text{expected}}(\boldsymbol{\theta})-\sum_{j=1}^{M}\ln\lambda(x_j|\boldsymbol{\theta})$$
where $N_{\text{expected}}$ is the integral of the total rate, and $\lambda(x_j)$ is the non-normalized rate evaluated strictly at the properties of event $j$.

---

## Outputs, Plots, and Checkpointing

### The Checkpoint Engine (`warm_start`)
Feldman-Cousins calculations are highly resource-intensive and often run on shared HPC clusters subject to preemption limits (e.g., Slurm time limits). If `warm_start` is enabled, PyFC dynamically writes its state to `fc_output/checkpoint_fc.npz` after processing each grid row. 

If your script is interrupted, simply run it again. PyFC will detect the `checkpoint_fc.npz` file, rigorously verify that your newly requested parameter grids match the saved geometry exactly, and seamlessly resume toy generation from the exact point of interruption.

### Stored Results
Upon successful completion, the pipeline outputs final structures directly to your `save_directory` (default: `fc_output/`):
* **`fc_results.npz`**: A highly compressed NumPy archive containing raw matrices for $t_{\text{data}}$, interpolated $t_{\text{critical}}$ surfaces, profiled nuisance parameters, and boolean acceptance masks across all configurations.
* **`fc_results.json`**: A human-readable dictionary summarizing the exact thresholds and limits, properly formatted with standard Python datatypes for web frameworks or cross-language ingestion.
* **`corner_plot.png`**: (And assorted 1D/2D projection images, generated via `plotting.py`) displaying your localized best-fit points alongside the extracted 68% and 90% CL limit contours mapped over your parameter space.

---

## Common Recipes and Questions

**Q: How many toys should I use?**
A: For a 68% CL interval (1-sigma), 500-1000 toys are often sufficient. For a 90% or 95% limit, 2000-5000 toys are required to smoothly resolve the tail of the test statistic distribution. Ensure `n_toys` is large enough that $N_{\text{toys}} \times (1 - \alpha) \gg 1$.

**Q: I have a parameter that represents a systematic uncertainty. How do I profile it?**
A: Simply pass a `np.linspace()` grid for that parameter into the `grids` list. PyFC automatically profiles (maximizes) all parameters in the `grids` list that are not currently fixed as the specific parameters of interest during 1D or 2D conditional scanning.

---

## How to Cite

If you utilize PyFC in your academic work or scientific publications, please cite the framework and link to the source repository:

Bustamante, M. (2026). *PyFC: A Python Framework for Feldman-Cousins Confidence Intervals*. GitHub Repository: https://github.com/mbustama/FeldmanCousins.

**Methodology References:**
* Feldman, G. J., & Cousins, R. D. (1998). Unified approach to the classical statistical analysis of small signals. *Physical Review D, 57*(7), 3873.
* Argüelles, C. A., Schneider, A., & Yuan, T. (2019). A binned likelihood for stochastic models. *Journal of High Energy Physics*, 2019(6), 1-18. [arXiv:1901.04645](https://arxiv.org/abs/1901.04645).