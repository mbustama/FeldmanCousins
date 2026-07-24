# PyFC: A Python Framework for Feldman-Cousins Confidence Intervals

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](#)

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
1. [Installation & Requirements](#installation--requirements)
2. [File Tree](#file-tree)
3. [Configuration & generate_config.py](#configuration--generate_configpy)
4. [Quick Start Guide](#quick-start-guide)
5. [Configuration Parameters (CLI / JSON)](#configuration-parameters-cli--json)
6. [Execution Strategies & Algorithmic Optimizations](#execution-strategies--algorithmic-optimizations)
7. [HPC & Parallelization Guidelines](#hpc--parallelization-guidelines)
8. [Statistical Methodology & Mathematics](#statistical-methodology--mathematics)
9. [Outputs, Plots, and Checkpointing](#outputs-plots-and-checkpointing)
10. [Common Recipes and Questions](#common-recipes-and-questions)
11. [Contributing](#contributing)
12. [License](#license)
13. [How to Cite](#how-to-cite)

---

## Installation & Requirements

PyFC requires **Python 3.8+**. Core dependencies include:
* `numpy >= 1.20`
* `scipy`
* `numba`
* `ultranest`
* `corner`
* `matplotlib`

Clone the repository and install via `pip` to automatically resolve and install all dependencies:

```bash
git clone [https://github.com/mbustama/FeldmanCousins.git](https://github.com/mbustama/FeldmanCousins.git)
cd FeldmanCousins
pip install -e .
```

**Testing:**
To verify that the installation was successful and the optimizers are functioning correctly on your hardware, run the test suite:
    
```bash
pip install pytest
pytest tests/
```

---

## File Tree

The project structure is organized modularly to separate analytical likelihood math from plotting and execution loops:

```text
FeldmanCousins/
├── pyproject.toml         # Build system and dependency specifications
├── README.md              # Project documentation
├── examples/              # Full end-to-end scripts for binned/unbinned workflows
├── tests/                 # Unit and integration test suite
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
```

---

## Configuration & generate_config.py

PyFC provides an interactive command-line tool to help users build their analysis configuration files securely and without typos. 

To generate a configuration file, simply run:
    
```bash
python -m pyfc.generate_config
```

The script will prompt you with questions regarding your likelihood type, number of toys, parallelization preferences, and smoothing options. It validates your inputs and writes a file (by default, `fc_config.json`) to your current working directory.

**Example `fc_config.json` with Default Values:**

```json
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
```

---

## Quick Start Guide

*Note: The code snippets below are examples of user-defined cases. You will need to substitute the models and input grids with your own specific physical models and parameter configurations. See the `examples/` directory in the repository for full, runnable end-to-end scripts demonstrating complex binned and unbinned atmospheric fits.*

### Defining Physics Models
The way you define your physics model depends entirely on whether your analysis is **binned** or **unbinned**.

**1. Binned Models**
For a phenomenological analysis, your models should be parameterized functions (callables) that map your physics parameters to 1D numpy arrays representing expected event counts per bin. 
*(Crucially: The output length and shape of your model functions must strictly match the shape of the `observed_counts` data array passed to the orchestrator to avoid broadcasting errors. Tip: Your custom functions must return strict NumPy arrays. Standard Python lists will not be automatically cast.)*

```python
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
```

**2. Unbinned Models**
For unbinned data, you must provide normalized probability density functions (PDFs) that can evaluate an array of kinematic coordinates (events) and return the probability density at each point. You must also provide a function that calculates the total expected event integral $N_{\text{expected}}$ across your parameter space.
*(Tip: If you wish to compile your unbinned continuous PDFs using Numba (`@njit`) to maximize evaluation speed, you must use pure NumPy mathematical expressions, as `scipy.stats` distributions are generally not Numba-compatible).*

```python
import numpy as np

def S_model_pdf(events):
    """
    Signal PDF for multi-dimensional unbinned events.
    'events' is a 2D array of shape (N_events, 2)
    """
    energy = events[:, 0]
    zenith = events[:, 1]
    
    # Evaluate density using pure NumPy (Numba-compatible)
    prob_e = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((energy - 5.0) / 1.0)**2)
    prob_z = (1.0 / (0.5 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((zenith - 0.0) / 0.5)**2)
    return prob_e * prob_z

def N_expected_func(flux_norm, bg_norm):
    """
    Returns the total expected number of events for the EUML formulation.
    """
    signal_integral = flux_norm * 500.0
    bg_integral = bg_norm * 1200.0
    return signal_integral + bg_integral
```

### Execution via JSON Config
Once you have your models defined and your config generated, you can pass them dynamically into the central orchestrator. 
*(Crucially: The JSON configuration file only dictates the hyperparameters of the algorithm. You must still pass your physical data, models, and spatial grids explicitly in your Python script. Note: If you choose not to use the CLI tool and omit passing a `**config` dictionary, the orchestrator will safely fall back to a set of hard-coded default hyperparameters.)*

```python
import json
import numpy as np
from pyfc.orchestrator import compute_fc_intervals

# 1. Define physical grids (the parameter space to scan)
# These grids correspond sequentially to the inputs of your model functions
grids = [
    np.linspace(1e-9, 1e-7, 20), # Param 1: flux_norm
    np.linspace(2.0, 3.0, 15),   # Param 2: spectral_index
    np.linspace(0.8, 1.2, 10)    # Param 3: bg_norm (nuisance parameter)
]

# 2. Mock Data
observed_counts = np.array([20, 7, 2, 0])

# 3. Load the generated JSON configuration hyperparameters
with open('fc_config.json', 'r') as f:
    config = json.load(f)

# 4. Execute Feldman-Cousins (unpacking the config dictionary via **)
# Note: If you only have a single unified physics model instead of separate 
# S/B models, simply pass S_model=unified_model_func and B_model=None.
# For unbinned runs, pass your N_expected_func via the N_expected kwarg.
results = compute_fc_intervals(
    data=observed_counts,
    S_model=S_model_func,
    B_model=B_model_func,
    grids=grids,
    poi_indices=[0, 1], # Explicitly scans Param 1 & 2, profiling Param 3
    **config 
)
```

---

## Configuration Parameters (CLI / JSON)

| Parameter | Description | Allowed Values | Default |
| :--- | :--- | :--- | :--- |
| `likelihood_type` | Evaluates models via Poisson bins or Extended Unbinned Maximum Likelihood. | `"binned"`, `"unbinned"` | `"binned"` |
| `cl` | Confidence Levels determining exact frequentist coverage integration targets. Dynamically sized; output keys in .npz will automatically match the provided levels (e.g., `mask_1d_95` for 0.95). | List of floats `(0.0, 1.0)` | `[0.68, 0.90]` |
| `n_toys` | Monte Carlo pseudo-experiments generated per parameter space point. | Integer `> 0` | `2000` |
| `strategy` | Optimizer used for finding global and conditional likelihood minima. | `"scipy"`, `"ultranest"`, `"hybrid"`, `"grid"` | `"scipy"` |
| `use_finite_mc_correction_binned` | Shifts Poisson likelihood to a Negative Binomial to account for finite simulation stats. | `True`, `False` | `True` |
| `compute_1D_intervals` | Toggles profiling nuisance parameters for 1D parameter limits. | `True`, `False` | `True` |
| `compute_2D_intervals` | Toggles joint 2D contour scanning and edge tracing. | `True`, `False` | `True` |
| `num_cores` | Thread/process count for parallel toy generation. `0` maps to max hardware threads. | Integer `>= 0` | `0` (Max) |
| `verbose` | Logging detail level. | `0` (Silent), `1`, `2` (Debug) | `1` |
| `warm_start` | Checkpoints interim state to `.npz` files to recover from preemptions. | `True`, `False` | `True` |
| `param_names` | Labels mapping the physical parameters for plotting outputs. Supports raw LaTeX (e.g., `[r"$\Phi$", r"$\gamma$"]`). | List of strings | `["param1", "param2", ...]` |
| `smooth_1d` | If True, applies default Gaussian kernel smoothing to 1D limit profiles in plots. (Support for custom scalar intensity is planned). | `True`, `False` | `False` |
| `smooth_2d` | If True, applies default interpolation smoothing to final 2D contour graphics. (Support for custom scalar intensity is planned). | `True`, `False` | `False` |
| `adaptive_toys` | Dynamically stops toy generation early if a grid point is definitively excluded, saving compute time. | `True`, `False` | `True` |
| `toy_batch_size` | Chunk size for batched array generation (optimizes memory/speed). | Integer `> 0` | `200` |
| `sparsify_grid` | Traces contour perimeters in 2D space to skip resolving deep interior/exterior nodes. | `True`, `False` | `True` |
| `save_log` | Pipes output directly to a persistent text log file. | `True`, `False` | `False` |
| `save_directory` | Directory path where final results, plots, and checkpoints reside. | String (path) | `"fc_output"` |
| `output_file` | Prefix for the serialized `.npz` and `.json` result data structures. | String | `"fc_results"` |

---

## Execution Strategies & Algorithmic Optimizations

PyFC provides several comprehensive levers to optimize computation time versus robustness based on the complexity of your likelihood surface.

```text
[Orchestrator] --> Defines Grids & POIs
     |
     v
[Optimizer] -----> Evaluates t_data at Data
     |
     v
[Toy Generator] -> Simulates N_toys at POI
     |
     v
[NLL Math] ------> Binned (Poisson/FiniteMC) or Unbinned (EUML)
     |
     v
[Results] -------> Yields t_critical & Acceptance Mask
```

### Handling Multi-Dimensional Parameter Spaces
PyFC is built to handle arbitrary $N$-dimensional physics models. You provide the full parameter space via the `grids` argument. By passing `poi_indices` to the orchestrator, you designate which dimensions serve as the primary Parameters of Interest (POIs) for the 1D/2D intervals. The algorithm seamlessly profiles (maximizes the likelihood over) all non-POI dimensions at every point during both the data fitting and the empirical toy generation phases.

### Optimizer Strategy (`strategy`)
*   **`"scipy"` (L-BFGS-B)**: Highly recommended for most physics applications. It utilizes bounding constraints and analytical approximations of the gradient to find likelihood minima incredibly quickly. It assumes a relatively smooth parameter space. *Bounds Logistics:* Bounds are automatically inferred from the minimum and maximum values of the corresponding parameter arrays you pass in the `grids` list. You do not need to pass a separate bounds argument.
*   **`"ultranest"`**: Utilizes Nested Sampling. Extremely robust against complex, multi-modal likelihood surfaces where standard gradient minimizers get trapped in local minima. It is much slower than SciPy but guarantees finding the global minimum.
*   **`"hybrid"`**: A balanced approach. Uses UltraNest to find the global unconditional best-fit (which happens only once), and uses SciPy for the thousands of conditional minimizations during the profile scanning and MC toy fitting.
*   **`"grid"`**: Brute-force scanning over the provided parameter nodes. Safest but computationally restrictive. Scales poorly with $N > 2$ dimensions.
*   **Error Handling:** If an optimizer fails to converge for a specific toy or grid point, PyFC logs a warning to the console (if `verbose > 0`), discards the failed toy, and automatically attempts to generate a replacement to preserve exact $N_{\text{toys}}$ statistics.

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

## HPC & Parallelization Guidelines

PyFC is explicitly engineered to scale across multi-core High-Performance Computing (HPC) nodes. Parallelization is handled contextually based on your analysis type:
*   **Binned Likelihoods**: Exploits GIL-bypassing via Numba-compiled C-extensions. Toy generation and fitting are parallelized at the NumPy array level.
*   **Unbinned Likelihoods**: Utilizes Python's `concurrent.futures.ProcessPoolExecutor` to spawn independent worker processes for continuous function evaluation.

**HPC Do's and Don'ts:**
*   **DO map `num_cores` to your Slurm/PBS allocations.** If you set `num_cores=0` (default), PyFC requests all available threads on the hardware. If running via a Slurm workload manager, it is highly recommended to explicitly pass the allocated CPUs to avoid overcommitting: 
    ```python
    import os
    config['num_cores'] = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    ```
*   **DO utilize whole nodes.** Because the unbinned optimizer relies on multi-processing, it scales near-linearly. Requesting exclusive nodes (e.g., 64 or 128 cores) will drastically reduce Feldman-Cousins runtime.
*   **DO monitor memory scaling for unbinned analyses.** Because `toy_batch_size` creates intermediate kinematic matrices inside the `ProcessPoolExecutor`, running $N_{\text{toys}} = 5000$ on 128 cores can cause memory exhaustion (OOM slurm kills) if your event arrays are massive. If this occurs, reduce `toy_batch_size` from 200 to 50.
*   **DON'T enable `save_log` for massive array jobs.** If you are submitting hundreds of job arrays to an HPC, writing individual `.txt` logs continuously can bottleneck shared network file systems (NFS). Rely on the binary checkpointing instead.

### Advanced Integrations
Note that PyFC's native parallelization logic maps threads and processes strictly within a single node's resource limits. Multi-node distribution (e.g., communicating via MPI) is not natively built into the `orchestrator`. Users requiring cluster-wide multi-node deployment should consider manually sharding their `grids` and wrapping `compute_fc_intervals` within an `mpi4py` communicator.

---

## Statistical Methodology & Mathematics

PyFC implements exact classical frequentist intervals. The code executes the methodology prescribed by Gary Feldman and Robert Cousins (1998) to solve the empty-set problem near physical boundaries.

### The Profile Likelihood Ratio
For a general parameter vector divided into parameters of interest $\boldsymbol{\theta}$ and nuisance parameters $\boldsymbol{\nu}$, we construct the Profile Likelihood Ratio (PLR) test statistic $t$:

$$t_{\text{data}}(\boldsymbol{\theta}) = -2 \ln \frac{\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})}{\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})}$$

Where:
* $\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})$ is the unconditional Maximum Likelihood Estimate (MLE) over the entire allowed parameter space.
* $\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})$ is the conditional MLE, evaluated at a fixed point $\boldsymbol{\theta}_{\text{test}}$, while profiling (maximizing) out the nuisance parameters $\boldsymbol{\nu}$.

By construction, $t \geq 0$. Lower values indicate excellent agreement between the data and the test hypothesis.

### Binned Likelihood
For binned configurations, the likelihood is the product of independent Poisson probabilities across $N$ bins. Dropping the data-dependent factorial constant, PyFC evaluates the Negative Log-Likelihood (NLL):

$$-\ln \mathcal{L}_{\text{Poisson}} = \sum_{i=1}^{N} \left( \mu_i(\boldsymbol{\theta}) - n_i \ln \mu_i(\boldsymbol{\theta}) \right)$$

**Finite Monte Carlo Correction:**
If `use_finite_mc_correction_binned` is True, the pure Poisson distribution is convoluted with a Gamma prior, shifting the likelihood to a Negative Binomial distribution:

$$-\ln \mathcal{L}_{\text{FiniteMC}} = \sum_{i=1}^{N} \left( \frac{\mu_i^2}{\sigma_i^2} \ln \left( 1 + \frac{\sigma_i^2}{\mu_i} \right) - n_i \ln \left( \frac{\mu_i}{1 + \sigma_i^2 / \mu_i} \right) \right)$$
where $\sigma_i^2$ is the variance (sum of squared MC weights) in bin $i$.

### Extended Unbinned Maximum Likelihood
When binning causes unacceptable information loss (e.g., highly complex kinematics with low event counts), PyFC evaluates the unbinned likelihood. Instead of bin counts, it uses the exact coordinates $x_j$ of the $M$ observed events.

$$-\ln \mathcal{L}_{\text{EUML}} = N_{\text{expected}}(\boldsymbol{\theta}) - \sum_{j=1}^{M} \ln \lambda(x_j | \boldsymbol{\theta})$$
where $N_{\text{expected}}$ is the integral of the total rate, and $\lambda(x_j)$ is the non-normalized rate evaluated strictly at the properties of event $j$.

---

## Outputs, Plots, and Checkpointing

### The Checkpoint Engine (`warm_start`)
Feldman-Cousins calculations are highly resource-intensive and often run on shared HPC clusters subject to preemption limits (e.g., Slurm time limits). By default, `warm_start` is set to `True`. PyFC dynamically writes its state to `fc_output/checkpoint_fc.npz` after processing each 1D slice of the parameters of interest. 

If your script is interrupted, simply run it again. PyFC will detect the `checkpoint_fc.npz` file, rigorously verify that your newly requested parameter grids match the saved geometry exactly, and seamlessly resume toy generation from the exact point of interruption.

### Stored Results & Custom Plotting
Upon successful completion, the pipeline outputs final structures directly to your `save_directory` (default: `fc_output/`):
* **`fc_results.npz`**: A highly compressed NumPy archive containing raw matrices and boolean masks.
* **`fc_results.json`**: A dictionary containing structural metadata, exact confidence levels, and the global unconditional best-fit coordinate.

**Available `.npz` Keys:**

| Key | Description | Shape |
| :--- | :--- | :--- |
| `grid_x`, `grid_y` | 2D parameter space meshgrids for plotting. | `(N, M)` |
| `t_data` | PLR test statistic evaluated on the real data. | `(N, M)` |
| `t_critical_68`, `t_critical_90` | Interpolated MC threshold surfaces. | `(N, M)` |
| `acceptance_68`, `acceptance_90` | 2D boolean masks (True = inside contour). | `(N, M)` |
| `mask_1d_68`, `mask_1d_90` | 1D boolean limits profiling other parameters. | `(N,)` or `(M,)` |

**Code Snippet: Generating Custom Contours**
While PyFC automatically saves default plots, you will likely want to format your own figures for publication. You can easily ingest the output files to do so. (Note: To access 1D limits instead of 2D contours, simply extract `mask_1d_90` and plot it against a 1D slice of your physical grid).

```python
import numpy as np
import matplotlib.pyplot as plt
import json

# 1. Load the numerical matrices
results = np.load('fc_output/fc_results.npz')
X = results['grid_x']
Y = results['grid_y']
mask_90 = results['acceptance_90']  # Boolean array where True = inside 90% CL limit

# 2. Load the metadata
with open('fc_output/fc_results.json', 'r') as f:
    meta = json.load(f)
best_fit = meta['best_fit']

# 3. Plot the allowed region
plt.figure(figsize=(8, 6))
# contourf uses the boolean mask to shade the allowed parameter space
plt.contourf(X, Y, mask_90, levels=[0.5, 1.5], colors=['#1f77b4'], alpha=0.3)

# Plot the global best fit point
plt.plot(best_fit[0], best_fit[1], 'r*', markersize=12, label='Global Best Fit')

plt.xlabel(meta['param_names'][0])
plt.ylabel(meta['param_names'][1])
plt.title('Custom 90% CL Feldman-Cousins Region')
plt.legend()
plt.show()
```

---

## Common Recipes and Questions

### Troubleshooting
*   **ValueError: operands could not be broadcast together**: This almost always means the output NumPy array shape of your model functions does not exactly match the shape of the `observed_counts` data array you passed to the orchestrator.
*   **UltraNest Convergence Warnings**: If you receive warnings that the nested sampler is struggling to converge, consider increasing your target parameter bounds (by expanding the ranges in your `grids`) or verifying that your likelihood surface does not contain unhandled `NaN` or `inf` values.

### Frequently Asked Questions
**Q: How many toys should I use?**
A: For a 68% CL interval (1-sigma), 500-1000 toys are often sufficient. For a 90% or 95% limit, 2000-5000 toys are required to smoothly resolve the tail of the test statistic distribution. Ensure `n_toys` is large enough that $N_{\text{toys}} \times (1 - \alpha) \gg 1$.

**Q: I have a parameter that represents a systematic uncertainty. How do I profile it?**
A: Simply pass a `np.linspace()` grid for that parameter into the `grids` list, and ensure its index is not included in the `poi_indices` when calling `compute_fc_intervals`. PyFC automatically profiles (maximizes) all parameters in the `grids` list that are not explicitly flagged as the parameters of interest.

---

## Contributing

We welcome contributions to PyFC, including bug reports, feature requests, and code modifications! 
1. Open an issue on the GitHub repository to discuss the proposed change.
2. Fork the repository and create a feature branch (`git checkout -b feature/new-optimizer`).
3. Ensure all tests pass (`pytest tests/`) and code is fully documented.
4. Submit a Pull Request.

---

## License

PyFC is distributed under the **GNU General Public License v3.0 (GPLv3)**. You are free to use, modify, and distribute this software, provided that any derivative works are also open-source and licensed under GPLv3. See the `LICENSE` file in the repository root for full details.

---

## How to Cite

If you utilize PyFC in your academic work or scientific publications, please cite the framework and link to the source repository:

Mauricio Bustamante (2026). *PyFC: A Python Framework for Feldman-Cousins Confidence Intervals*. GitHub Repository: https://github.com/mbustama/FeldmanCousins.

**Methodology References:**
* Gary J. Feldman & Robert D. Cousins (1998). Unified approach to the classical statistical analysis of small signals. *Physical Review D, 57*(7), 3873.
* Carlos A. Argüelles, Austin Schneider & Tianlu Yuan (2019). A binned likelihood for stochastic models. *Journal of High Energy Physics*, 2019(6), 1-18. [arXiv:1901.04645](https://arxiv.org/abs/1901.04645).