<p align="center">
  <img src="https://raw.githubusercontent.com/mbustama/FeldmanCousins/main/docs/source/_static/pyfc_logo.png" alt="PyFC Logo" width="180">
</p>

# PyFC: A Python Framework for Feldman-Cousins Confidence Intervals

[![CI Tests](https://github.com/mbustama/FeldmanCousins/actions/workflows/pytest.yml/badge.svg)](https://github.com/mbustama/FeldmanCousins/actions/workflows/pytest.yml)
[![Code Quality](https://github.com/mbustama/FeldmanCousins/actions/workflows/lint.yml/badge.svg)](https://github.com/mbustama/FeldmanCousins/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/mbustama/FeldmanCousins/branch/main/graph/badge.svg)](https://codecov.io/gh/mbustama/FeldmanCousins)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://mbustama.github.io/FeldmanCousins/)
[![PyPI](https://img.shields.io/pypi/v/PyFeldmanCousins.svg)](https://pypi.org/project/PyFeldmanCousins/)
[![License: GPL v3+](https://img.shields.io/badge/License-GPLv3+-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://pepy.tech/badge/PyFeldmanCousins)](https://pepy.tech/project/PyFeldmanCousins)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**PyFC** is a rigorous, high-performance frequentist statistical analysis framework for Python. It automates the construction of classical confidence intervals and regions using the unified Feldman-Cousins approach, seamlessly transitioning between one-sided upper limits and two-sided bounds while guaranteeing exact frequentist coverage. 

Designed for high-energy physics, astrophysics, and general parametric modeling, PyFC handles both binned (histogram) and unbinned (event-by-event) data, integrates multiple optimization strategies (SciPy, UltraNest, Grid, and a Hybrid of the two), and utilizes highly parallelized Monte Carlo pseudo-experiment generation.

## Salient Features
* **Unified binned and unbinned analysis**: Natively supports Poisson binned data (including arbitrary N-dimensional, even non-contiguous, histograms) and Extended Unbinned Maximum Likelihood (EUML) formulations.
* **Exact coverage**: Empirically derives the Profile Likelihood Ratio (PLR) test statistic distribution via dynamically generated Monte Carlo pseudo-experiments (toys), correctly reporting disconnected accepted regions as separate intervals rather than silently merging them.
* **Advanced optimizers**: Supports gradient-based L-BFGS-B (SciPy), nested sampling (UltraNest), brute-force grid scanning, and a Hybrid strategy that uses UltraNest's robustness for the one-time global fit and SciPy's speed for the many conditional fits.
* **Joint/simplex-constrained parameters**: `bounds_func` and `constraints` mechanisms express physical constraints between parameters (e.g. a simplex `a + b <= 1`) that an independent per-parameter box can't -- no fragile hand-rolled penalty function required.
* **Massive parallelization**: GIL-bypassing via NumPy/Numba C-extensions for binned thread pooling, and `ProcessPoolExecutor` for unbinned continuous functions, with statistically-validated adaptive early-stopping (`adaptive_toys`) and batched dispatch (`toy_batch_size`) to cut wasted compute without introducing bias.
* **Finite Monte Carlo corrections**: Implements Argüelles, Schneider & Yuan's marginalized Poisson-Gamma mixture likelihood (`L_Eff`, arXiv:1901.04645) to account for finite simulation statistics -- not the (different, profiled-nuisance-parameter) Barlow-Beeston likelihood, despite the similar "Poisson-Gamma" framing.
* **Optional 2D grid sparsification**: `sparsify_grid` (opt-in, off by default) uses Bivariate Spline interpolation and edge-tracing to skip toy generation deep inside/outside 2D contours -- a real speedup where it applies cleanly, though it doesn't yet scale reliably much past ~20x20 nodes per axis (see the caveat in [Execution Strategies & Algorithmic Optimizations](#execution-strategies--algorithmic-optimizations) below).
* **State checkpointing**: "Warm start" capability saves binary state matrices after each parameter/pair scan completes, so an interrupted run (e.g. a Slurm walltime kill) resumes instead of restarting from scratch.
* **Interactive configuration**: Built-in CLI for generating serialized JSON experiment configurations.

---

## Table of Contents
1. [Installation & Requirements](#installation--requirements)
2. [File Tree](#file-tree)
3. [Configuration & generate_config.py](#configuration--generate_configpy)
4. [Quick Start Guide](#quick-start-guide)
5. [Tutorial Notebooks](#tutorial-notebooks)
6. [Documentation](#documentation)
7. [Configuration Parameters (CLI / JSON)](#configuration-parameters-cli--json)
8. [Execution Strategies & Algorithmic Optimizations](#execution-strategies--algorithmic-optimizations)
9. [HPC & Parallelization Guidelines](#hpc--parallelization-guidelines)
10. [Statistical Methodology & Mathematics](#statistical-methodology--mathematics)
11. [Outputs, Plots, and Checkpointing](#outputs-plots-and-checkpointing)
12. [Common Recipes and Questions](#common-recipes-and-questions)
13. [Contributing](#contributing)
14. [License](#license)
15. [How to Cite](#how-to-cite)

---

## Installation & Requirements

### Requirements

PyFC requires **Python 3.9+**. Core dependencies include:
* `numpy >= 1.20`
* `scipy`
* `numba`
* `matplotlib`

**Optional Dependencies:**
* `ultranest` (Required for utilizing the nested sampling optimizer)
* `tqdm` (Provides progress bars during execution)

### Installation

#### Option 1: Direct PyPI Installation (Recommended)
To install the latest stable release directly from the Python Package Index (PyPI), run:

```bash
pip install PyFeldmanCousins
```

`pip` is not case-sensitive, so using `pip install pyfeldmancousins` will have the same result.

#### Option 2: Local Installation from Source
If you want to run the latest development version or modify the codebase, clone the repository and install via `pip` to automatically resolve and install all dependencies:

```bash
git clone https://github.com/mbustama/FeldmanCousins.git
cd FeldmanCousins
pip install -e .
```

### Verifying the Installation

After installing the package, you can run the local test suite to ensure everything is configured correctly for your system architecture. We utilize `pytest` to validate statistical correctness, optimizer behavior, and parallelization scaling. 

```bash
# Install the testing framework
pip install pytest

# Run the test suite from the repository root
pytest tests/ -v
```

#### Measuring Test Coverage

The `test` extra also installs `pytest-cov`, so the same suite can report which lines and branches of `pyfc/` it actually exercises. What is measured — the source tree, branch coverage, and what is excluded — is configured once in `[tool.coverage.run]` in `pyproject.toml`, so a local run measures exactly what CI measures:

```bash
bash scripts/run_coverage.sh
```

Add `--html` for a browsable `htmlcov/` tree, which is the more useful form when the question is *which* branch of a particular function is cold, or `--xml` for the Cobertura `coverage.xml` that CI uploads.

That wrapper exists because of Numba. PyFC's mathematical core (`pyfc/binned.py`) is six `@njit`-compiled functions, `calc_nll` among them, and `coverage.py` traces Python bytecode — which Numba never executes, since it compiles those functions to machine code on first call. Every line inside them is therefore reported as missed no matter how hard the suite hits them. Measured over the three test files that target `calc_nll` directly:

| Run | `pyfc/binned.py` reports |
|---|---|
| default (JIT on) | **10%** |
| `NUMBA_DISABLE_JIT=1` | **90%** |

10% is not a real number, and publishing it would be actively misleading about the part of PyFC that most needs to be trustworthy. Setting `NUMBA_DISABLE_JIT=1` makes Numba run the decorated functions as ordinary Python, which coverage can trace. That is not uniformly free, though: most files are *faster* without JIT (they stop paying compilation overhead), but `strategy="grid"`'s `@njit(parallel=True)` toy generators degrade into plain Python loops, and a whole-suite run with JIT disabled does not finish in any reasonable time. So `run_coverage.sh` splits the suite in two — the pathological files run with JIT on, everything else with JIT off — and lets `coverage` combine the halves. Running a bare `pytest tests/ --cov` still works and is measured identically; it just reports the misleading ~10% for `binned.py`.

Branch coverage is on deliberately. A plain line-coverage figure flatters this codebase: the orchestrator is a dispatch machine, and most of what can go wrong in it is a branch only ever taken one way — `strategy="grid"` vs `"adaptive"`, a resumed checkpoint vs a fresh run, the disconnected-interval reporting path, the `if ultranest is None` guards. Line coverage counts those as covered the moment the function runs at all.

### Continuous Integration (CI)

PyFC is protected by a Continuous Integration (CI) pipeline powered by GitHub Actions. Every time code is pushed or a Pull Request is opened, the CI automatically provisions pristine Ubuntu runners across a matrix of Python versions (e.g., 3.9, 3.10, 3.11). It installs PyFC entirely from scratch and executes the full test suite. This strict isolation eliminates "it works on my machine" biases and guarantees that new code contributions do not introduce regressions.

The matrix job deliberately installs only `.[test]`, *without* the optional `optimizers` extra: that `ultranest` is optional is a promise PyFC makes to its users, and this is where that promise is checked.

The test suite currently comprises 181 tests across 27 files under `tests/` (164 always run, plus 17 UltraNest-specific tests that skip automatically if the optional `ultranest` package isn't installed), grouped into four categories:
* **Core statistical correctness** (48 tests, 7 files): the smoothed unphysical-rate NLL penalty and its boundary-discontinuity/NaN handling, the finite-MC likelihood formula checked against hand-computed references (including its numerically-stable high-`alpha` reformulation), N-dimensional and non-contiguous binned data, disconnected 1D accepted-interval reporting, unbinned `pdf_components` correctness (2- and 3+-component models), the unbinned toy-generation/fitting pipeline end to end, and the unbinned grid-search path checked against brute-force oracles and the profile-likelihood invariants (a conditional fit never undercuts the unconditional minimum, and the two coincide exactly at the best-fit point).
* **Optimizer robustness** (41 tests, 6 files): optimizer restarts and neighbor warm-starting, `bounds_func`/`constraints` support (SciPy and UltraNest `LinearConstraint`/`NonlinearConstraint`, including the zero-free-parameters edge case), SciPy boundary clamping, Asimov-dataset convergence (the minimizer must recover the known true parameters exactly), and the three UltraNest fits run for real against an Asimov dataset -- checked for MLE recovery, the profile-likelihood invariants, `bounds_func` handling, the unbinned likelihood path, and agreement with SciPy on the same optimum.
* **Algorithmic features** (13 tests, 2 files): `sparsify_grid`'s boundary-refinement guard, and the validated `adaptive_toys`/`toy_batch_size` early-stopping behavior.
* **Infrastructure** (79 tests, 12 files): core imports and optional-dependency detection, I/O integrity, the real `warm_start`/checkpoint-resume machinery (not just an `.npz` round-trip), including resuming a run interrupted mid-2D-scan and `sparsify_grid`'s nearest-neighbour interpolation fallback for installs without SciPy, multiprocessing serialization via `ProcessPoolExecutor` and the toy pool's documented recovery when a user's callable turns out to be unpicklable, Numba JIT compilation hooks, UltraNest's internal-bug retry wrapper, corner-plot generation, the JSON results export (`NumpyEncoder`'s type conversions and the 2D payload, round-tripped through a stock `json.load`), and the CLI/JSON configuration layer — its `Defaults -> JSON -> CLI` precedence chain, its refusal to spin or invent answers when stdin runs out, and a set of self-discovering structural sweeps that check every analysis parameter lines up across all four places it is written down (the hardcoded defaults, the `argparse` flags, the interactive wizard, and `compute_fc_intervals`' own signature).

(Counts and groupings reflect the test suite at the time of writing; run `pytest tests/ --collect-only -q` for the current, authoritative total.)

The same workflow also has a separate **Coverage** job. It runs the suite once more under `scripts/run_coverage.sh`, prints the per-module statement and branch coverage on the run's summary page, and uploads `coverage.xml` as a build artifact. It reports rather than gates — there is no `--cov-fail-under` — and it is its own job rather than a fourth entry in the matrix because coverage does not depend on the interpreter, so measuring it three times would only pay the instrumentation cost three times. Unlike the matrix job it *does* install `.[test,optimizers]`: without `ultranest`, the three `*_ultranest` fit functions count as missed and the reported figure for `optimizers.py` understates its real coverage, which invites someone to go and "fix" coverage that is not actually missing.

It also carries a Codecov upload step, gated on a `CODECOV_TOKEN` repository secret: the step is skipped entirely when that secret is absent, so the workflow is safe to run in a fork or before coverage reporting is wanted. That secret **does** now exist for this repository, so uploads are live, the coverage badge above resolves, and Codecov comments per-PR diff coverage — the part that actually changes day-to-day behaviour, since it shows the coverage of the lines in front of you. `fail_ci_if_error` is on: an upload that quietly failed would leave the badge frozen at a stale number under a green checkmark, which is worse than a visible failure. It cannot affect the tests' own pass/fail signal, which comes from the matrix job.

### Developer Installation
If you plan to modify the codebase or contribute to the project, you should install the package with its optional testing and development dependencies included. This ensures you have tools like `pytest` and `pytest-cov` ready to go without cluttering the requirements for standard users:

```bash
# The quote marks are important for some shells (like zsh)
pip install -e ".[test]"

# To reproduce the CI coverage job exactly, add the optimizers extra so the
# UltraNest code paths are measured rather than counted as missed
pip install -e ".[test,optimizers]"
```

---

## File Tree

The project structure is organized modularly to separate analytical likelihood math from plotting, execution loops, and documentation (file tree generated by running `git ls-files`):

```text
FeldmanCousins/
├── .github/
│   └── workflows/
│       ├── lint.yml                 # CI linting and formatting pipeline
│       ├── pages.yml                # GitHub Pages deployment for documentation
│       ├── publish.yml              # PyPI (OIDC) automated publishing workflow
│       └── pytest.yml               # GitHub Actions CI testing and coverage pipeline
├── config/
│   └── example_fc_config.json       # Template configuration file for CLI usage
├── docs/                            # Sphinx documentation configuration and source
│   ├── dev/                         # Handoff/planning notes from past refactors (not part of the built docs)
│   │   ├── AUDIT_BRIEF_dev-no-templates.md
│   │   ├── FIXES_BRIEF_dev-no-templates.md
│   │   ├── FOLLOWUP_BRIEF_dev-no-templates.md
│   │   └── REFACTOR_BRIEF_dev-no-templates.md
│   ├── source/
│   │   ├── _static/
│   │   │   └── pyfc_logo.png        # Project branding asset
│   │   ├── changelog.rst            # Renders the root CHANGELOG.md via myst-parser
│   │   ├── conf.py                  # Sphinx build configuration
│   │   ├── configuration.rst        # Interactive CLI tool and parameter definitions
│   │   ├── index.rst                # Master documentation page with high-level overview and table of contents
│   │   ├── installation.rst         # System requirements, dependencies, and setup instructions
│   │   ├── methodology.rst          # Statistical math formulations and optimizer execution strategies
│   │   ├── modules.rst              # Autodoc stubs for the PyFC submodules
│   │   ├── outputs.rst              # Explanation of checkpointing files and returned array structures
│   │   ├── pyfc.rst                 # Package-level API structure
│   │   ├── quickstart.rst           # Code tutorials for setting up binned and unbinned models
│   │   ├── references.rst           # Bibliography page rendering
│   │   ├── refs.bib                 # BibTeX citations for physics and statistics literature
│   │   └── tutorials.rst            # Guide to the numbered example notebooks in examples/
│   ├── Makefile                     # Build commands for Unix
│   └── make.bat                     # Build commands for Windows
├── examples/                         # Numbered in suggested reading order -- see "Tutorial Notebooks" below
│   ├── 01_pyfc_quickstart_tutorial.ipynb            # Runnable counterpart to the Quick Start Guide (binned + unbinned)
│   ├── 02_pyfc_high_dimensional_tutorial.ipynb      # Scaling to many parameters (5- and 10-parameter models)
│   ├── 03_pyfc_non_contiguous_data_tutorial.ipynb   # Passing non-contiguous (transposed) N-D data arrays directly
│   ├── 04_pyfc_algorithmic_features_tutorial.ipynb  # sparsify_grid / smooth_1d,2d / finite-MC correction, on vs. off
│   ├── 05_pyfc_strategy_comparison_tutorial.ipynb   # grid/scipy/hybrid strategy comparison + custom plotting from disk
│   ├── 06_pyfc_joint_constraints_tutorial.ipynb     # Worked example: bounds_func/constraints for joint & simplex-constrained parameters
│   └── 07_pyfc_checkpointing_tutorial.ipynb         # Interrupting (SIGTERM) and resuming a real run via warm_start
├── pyfc/                            # Main Python package
│   ├── __init__.py                  # Package initialization and metadata
│   ├── binned.py                    # Binned NLL math and Numba-accelerated optimizers
│   ├── config.py                    # CLI argument definitions and JSON parsing
│   ├── generate_config.py           # Interactive CLI wizard for creating configuration files
│   ├── optimizers.py                # Wrapper functions mapping objective functions to SciPy/UltraNest
│   ├── orchestrator.py              # The main pipeline executing the Feldman-Cousins algorithm
│   ├── plotting.py                  # Visualization suite for 1D profiles and 2D contours
│   ├── toys.py                      # Multiprocessing engines for MC pseudo-experiment generation
│   └── unbinned.py                  # Extended Unbinned Maximum Likelihood (EUML) formulations
├── scripts/
│   └── run_coverage.sh              # Coverage measurement, splitting the suite around Numba's JIT
├── tests/                           # Unit and integration test suite
│   ├── test_2d_resume_and_sparsify.py             # Tests for 2D checkpoint resume and sparsify_grid's no-SciPy fallback
│   ├── test_adaptive_toys.py                        # Tests for the validated adaptive_toys/toy_batch_size early-stopping behavior
│   ├── test_bounds_func.py                          # Tests for the bounds_func parameter-dependent bounds mechanism
│   ├── test_checkpoint_resume.py                    # Tests for the real warm_start/checkpoint-resume machinery, simulating a genuine mid-run interruption
│   ├── test_config_layer.py                         # Structural + behavioural tests for the CLI/JSON config layer and the interactive wizard
│   ├── test_constraints.py                          # Tests for scipy/UltraNest LinearConstraint/NonlinearConstraint support
│   ├── test_core.py                                 # Core installation and import tests, including checks for optional dependencies
│   ├── test_disconnected_intervals.py               # Tests for the contiguous-run helper and disconnected 1D interval reporting
│   ├── test_finite_mc_likelihood.py                 # Hand-computed-reference tests for the finite-MC likelihood formula
│   ├── test_json_export.py                          # Tests for the JSON results export: NumpyEncoder conversions and the 2D payload
│   ├── test_io.py                                   # Tests for File I/O and intermediate .npz checkpoint recovery mechanisms
│   ├── test_multiprocessing.py                      # Tests for unbinned execution and multi-processing concurrency using ProcessPoolExecutor
│   ├── test_nd_binned.py                            # Tests for N-dimensional binned data (flatten-invariance, 2D histogram smoke test)
│   ├── test_numba_compilation.py                    # Tests to ensure Numba JIT compilation executes correctly on the host architecture
│   ├── test_optimizers.py                           # Tests for continuous optimization routines, including SciPy boundary clamping
│   ├── test_pdf_components.py                       # Tests for the pdf_components mechanism (2- and 3+-component correctness)
│   ├── test_plotting.py                             # Tests for generate_corner_plot's figure/axes shape, smoothing toggles, and output file
│   ├── test_restarts.py                             # Tests for optimizer restarts, neighbor warm-starting, and res.success handling
│   ├── test_smoothing.py                            # Tests for the smoothed unphysical-rate NLL penalty, including NaN handling
│   ├── test_sparsify_grid.py                        # Regression tests for the sparsify_grid boundary-refinement guard fix
│   ├── test_statistics.py                           # Tests for statistical correctness, likelihood behavior, and Asimov treatment convergence
│   ├── test_toy_pool_fallback.py                    # Tests for the toy pool's ProcessPoolExecutor -> ThreadPoolExecutor recovery on unpicklable callables
│   ├── test_ultranest_fits.py                       # Tests that actually run the three UltraNest fits against an Asimov dataset
│   ├── test_ultranest_retry.py                      # Tests for the UltraNest internal-bug retry wrapper
│   ├── test_unbinned_grid.py                        # Tests for the unbinned grid-search path, against brute-force oracles and the PLR invariants
│   ├── test_unbinned_toys.py                        # Tests for the unbinned toy-generation/fitting pipeline via a real ProcessPoolExecutor
│   └── test_version.py                              # Tests that the version is written down once and reaches package and docs from there
├── xbranch_compare/                 # Cross-branch (dev vs dev-no-templates) regression harness
│   ├── comparator.py                # Recursive .npz diff (shape + NaN-mask + tolerant allclose)
│   ├── compare_results.py           # Diffs the cross-branch scenario .npz outputs
│   ├── run_comparison.sh            # Driver: git worktree add/remove + both scenario scripts + diff
│   ├── shared_constants.py          # Physical-model constants shared by both scenario scripts (no pyfc import)
│   ├── xbranch_new_api.py           # Runs all scenarios against the new (pdf_components) API
│   └── xbranch_old_api.py           # Runs the old-API-compatible scenarios against a `dev` worktree
├── .gitignore                       # Git untracked files exclusions
├── CHANGELOG.md                     # Version history and notable changes
├── LICENSE                          # Open-source license terms
├── pyproject.toml                   # Build system and dependency specifications
└── README.md                        # Project documentation
```

---

## Configuration & generate_config.py

PyFC provides an interactive command-line tool to help users build their analysis configuration files securely and without typos. 

Once PyFC is installed via `pip`, the package modules become available globally in your Python environment. To interactively generate a configuration file from anywhere, simply run:
    
```bash
python -m pyfc.generate_config
```

The script will prompt you with questions regarding your likelihood type, number of toys, parallelization preferences, and smoothing options. It validates your inputs and writes a file (by default, `fc_config.json`) to your current working directory. Its default choices for each question match `compute_fc_intervals`'s own documented defaults (below) -- press Enter on every question to reproduce them, with three deliberate exceptions: `num_cores`, `output_file`, and `param_names` all *actually* default to `null` (auto-detect every hardware thread, skip writing a results file, and auto-generate `param{i}` names matching your model's real parameter count, respectively) rather than the illustrative literal values (`8`, `"fc_results"`, `["param1", "param2", "param3"]`) shown for them in the JSON block below -- see each parameter's own docstring on `compute_fc_intervals` for why a concrete example wouldn't generalize.

**Example `fc_config.json` Default Values:**
*(Note: This is a fully-populated example, not a literal dump of `compute_fc_intervals`'s embedded fallback defaults -- `num_cores`/`output_file`/`param_names` are shown here with concrete illustrative values, but actually default to `null`/auto-detecting behavior when omitted; see the exception noted above.)*

```json
{
    "likelihood_type": "binned",
    "cl": [0.68, 0.90],
    "n_toys": 500,
    "strategy": "scipy",
    "num_cores": 8,
    "verbose": 1,
    "adaptive_toys": true,
    "toy_batch_size": 200,
    "sparsify_grid": false,
    "n_restarts": 1,
    "neighbor_seeding": true,
    "warm_start": true,
    "output_file": "fc_results",
    "save_log": true,
    "save_directory": "output/example_fc_output",
    "use_finite_mc_correction_binned": true,
    "compute_1D_intervals": true,
    "compute_2D_intervals": true,
    "param_names": ["param1", "param2", "param3"],
    "smooth_1d": true,
    "smooth_2d": true
}
```

---

## Quick Start Guide

Because PyFC evaluates abstract $N$-dimensional grids, you **must** supply Python functions instructing the framework how to mathematically map a coordinate in parameter space to your physical expectations. 

*(See the `examples/01_pyfc_quickstart_tutorial.ipynb` notebook in the repository for a full, runnable end-to-end walkthrough of both models below -- or the "Tutorial Notebooks" section below for the full set.)*

### 1. Binned Models
For a binned analysis, fixed templates are ordinary NumPy arrays referenced via closure (module-global or nested-function capture) from your `compute_rates_func` -- they are no longer passed in as function arguments. You must write a `compute_rates_func` that combines them with your varied parameters to yield the total expected bin counts ($\mu$) and variances ($\sigma^2$). `data`/`mu`/`sigma2` may be **any shape**, not just 1D -- a genuine 2D `(E, cos_theta)` histogram is fully supported and evaluated directly.

*(Crucially: If you rely on Numba's maximum parallelization via the `"grid"` strategy or massive thread pools, your `compute_rates_func` **must** be decorated with `@njit`.)*

```python
import numpy as np
import json
from numba import njit
from pyfc import compute_fc_intervals

# A) Fixed templates as module-level constants, referenced via closure
S_template = np.array([0.1, 0.5, 2.0, 5.0])
B_template = np.array([15.0, 5.0, 1.0, 0.1])

# B) Define the physics mapper function
@njit(fastmath=True, nogil=True)
def my_compute_rates_binned(params, S_sumw2, B_sumw2):
    """ Maps parameters to expected counts (mu) and simulated variances (sigma2). """
    # Example: params[0] = flux_norm, params[1] = spectral_index, params[2] = bg_norm
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
```

### 2. Unbinned Models
For unbinned data, `pdf_components` is a **list** of Python functions that each evaluate a PDF over exact kinematic coordinates (signal, background, or any number of further components -- no longer limited to exactly two). Each is evaluated exactly once per fit and the resulting arrays are reused across every subsequent NLL evaluation. You must provide a `compute_rates_func` that consumes the pre-evaluated `probs` list to yield the overall expected integral and localized probability density, alongside a `generate_toy_func` that handles parametric bootstrapping.

*(Note: Unbinned operations utilize `ProcessPoolExecutor` to bypass the GIL. Ensure your custom functions are defined at the top-level of your script so Python can serialize them across processes.)*

```python
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
    # Edge case protection for zero events
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
    data=observed_events, # Shape: (N_events,) or (N_events, D_features)
    grids=grids,
    compute_rates_func=my_compute_rates_unbinned,
    generate_toy_func=my_generate_unbinned_toy,
    pdf_components=[s_pdf, b_pdf],
    **config 
)
```

---

## Tutorial Notebooks

The `examples/` directory contains seven runnable Jupyter notebooks, numbered `01`-`07` in the
order we'd suggest reading them. Each one is self-contained (states its own imports and mock
data) and ends with a "Next steps" pointer to the notebooks that naturally follow it, so you
can also jump straight to whichever topic matches your own project.

| # | Notebook | What it covers | Read this if... |
|---|---|---|---|
| 01 | [`pyfc_quickstart_tutorial.ipynb`](https://github.com/mbustama/FeldmanCousins/blob/main/examples/01_pyfc_quickstart_tutorial.ipynb) | A full binned model and a full unbinned model, end to end, with real output. The runnable counterpart to the [Quick Start Guide](#quick-start-guide) below. | **Start here.** This is the on-ramp for a brand-new project -- copy whichever of the two models matches your data. |
| 02 | [`pyfc_high_dimensional_tutorial.ipynb`](https://github.com/mbustama/FeldmanCousins/blob/main/examples/02_pyfc_high_dimensional_tutorial.ipynb) | Scaling `compute_fc_intervals` to 5 and then 10 free parameters; why 1D scans stay cheap while 2D contours grow combinatorially ($\binom{n}{2}$ pairs). | Your model has more than 2-3 free parameters. |
| 03 | [`pyfc_non_contiguous_data_tutorial.ipynb`](https://github.com/mbustama/FeldmanCousins/blob/main/examples/03_pyfc_non_contiguous_data_tutorial.ipynb) | Passing arbitrarily-shaped, even non-contiguous (e.g. transposed), N-D binned `data` arrays straight into PyFC, with a proof of correctness. | Your histogram isn't a plain 1D array, or you build it with a different axis order than PyFC expects. |
| 04 | [`pyfc_algorithmic_features_tutorial.ipynb`](https://github.com/mbustama/FeldmanCousins/blob/main/examples/04_pyfc_algorithmic_features_tutorial.ipynb) | The `sparsify_grid`, `smooth_1d`/`smooth_2d`, and `use_finite_mc_correction_binned` knobs, each demonstrated on vs. off -- plus how disconnected accepted intervals get reported. | You need to speed up a 2D scan, polish a plot, or your MC templates have limited statistics. |
| 05 | [`pyfc_strategy_comparison_tutorial.ipynb`](https://github.com/mbustama/FeldmanCousins/blob/main/examples/05_pyfc_strategy_comparison_tutorial.ipynb) | Benchmarking `"grid"`/`"scipy"`/`"hybrid"`/`"ultranest"` on the same model, then building a fully custom Matplotlib figure directly from PyFC's saved `.json`/`.npz` output (no dependency on `generate_corner_plot`). | You're choosing an optimizer strategy, or you want a publication-quality figure beyond PyFC's built-in plot. |
| 06 | [`pyfc_joint_constraints_tutorial.ipynb`](https://github.com/mbustama/FeldmanCousins/blob/main/examples/06_pyfc_joint_constraints_tutorial.ipynb) | Reproducing and fixing the flat-gradient optimizer trap that comes from a joint (non-box) constraint like a simplex ($f_e + f_\mu \leq 1$), using `bounds_func` and `constraints`. | Two or more of your parameters are linked by an inequality that a per-parameter `[lo, hi]` box can't express. |
| 07 | [`pyfc_checkpointing_tutorial.ipynb`](https://github.com/mbustama/FeldmanCousins/blob/main/examples/07_pyfc_checkpointing_tutorial.ipynb) | A genuinely-running `compute_fc_intervals` call, killed mid-analysis with `SIGTERM` (mimicking a Slurm walltime kill), then correctly resumed with `warm_start=True` -- with a checkpoint inspection and a determinism check proving nothing was silently recomputed or corrupted. | You're running on a preemptible/walltime-limited cluster, or just want to see the checkpointing "Salient Feature" actually happen. |

`01`-`03` build on each other and are worth reading in order for a first project; `04`-`07`
are independent, narrower deep-dives you can read in any order once you need that specific
capability. See also `docs/source/tutorials.rst` (rendered as part of the [hosted
documentation](https://mbustama.github.io/FeldmanCousins/)) for the same guide.

---

## Documentation

Comprehensive documentation for PyFC is hosted on GitHub Pages. It includes a quickstart guide, a breakdown of the statistical methodology, and the full API reference.

📚 **[Read the PyFC Documentation & API](https://mbustama.github.io/FeldmanCousins/)**

---

## Configuration Parameters (CLI / JSON)

| Parameter | Description | Allowed Values | Default |
| :--- | :--- | :--- | :--- |
| `likelihood_type` | Evaluates models via Poisson bins or Extended Unbinned Maximum Likelihood. | `"binned"`, `"unbinned"` | `"binned"` |
| `cl` | Confidence Levels determining exact frequentist coverage integration targets. Dynamically sized; output keys in .npz will automatically match the provided levels (e.g., `1d_accepted_p1_0.9` for 0.90). | List of floats `(0.0, 1.0)` | `[0.68, 0.90]` |
| `n_toys` | Monte Carlo pseudo-experiments generated per parameter space point. | Integer `> 0` | `500` |
| `strategy` | Optimizer used for finding global and conditional likelihood minima. | `"scipy"`, `"ultranest"`, `"hybrid"`, `"grid"` | `"scipy"` |
| `scipy_method` | Overrides the `scipy.optimize.minimize` method. Only meaningful for `strategy="scipy"`/`"hybrid"`. `null` lets PyFC pick automatically: `"L-BFGS-B"` by default, or `"SLSQP"` if `constraints` are supplied programmatically (see [Handling joint/simplex-constrained parameters](#handling-jointsimplex-constrained-parameters)). | `null`, `"L-BFGS-B"`, `"SLSQP"`, `"trust-constr"` | `null` |
| `use_finite_mc_correction_binned` | Shifts Poisson likelihood to a Negative Binomial to account for finite simulation stats. | `True`, `False` | `True` |
| `compute_1D_intervals` | Toggles 1D limits mapping. | `True`, `False` | `True` |
| `compute_2D_intervals` | Toggles joint 2D contour scanning and edge tracing. | `True`, `False` | `True` |
| `num_cores` | Thread/process count for parallel toy generation. `null` (None) maps to max hardware threads. | Integer `>= 0` | `8` |
| `verbose` | Logging detail level. | `0` (Silent), `1`, `2` (Debug) | `1` |
| `warm_start` | Checkpoints interim state to `.npz` files to recover from preemptions. | `True`, `False` | `True` |
| `param_names` | Labels mapping the physical parameters for plotting outputs. Supports raw LaTeX (e.g., `[r"$\Phi$", r"$\gamma$"]`). | List of strings | `["param1", "param2", ...]` |
| `smooth_1d` | If True, applies default Gaussian kernel smoothing to 1D limit profiles in plots. | `True`, `False` | `True` |
| `smooth_2d` | If True, applies default interpolation smoothing to final 2D contour graphics. | `True`, `False` | `True` |
| `adaptive_toys` | Dynamically stops toy generation early once a grid point's accept/reject verdict is statistically settled (strict 99.9% confidence, never before 100 toys), saving compute time. Only for `strategy` in `"scipy"`/`"ultranest"`/`"hybrid"`; no effect for `"grid"`. | `True`, `False` | `True` |
| `toy_batch_size` | Chunk size for submitting/collecting toys from the executor (bounds peak memory for `likelihood_type="unbinned"`; also the granularity of `adaptive_toys`' stopping checks). Same total `n_toys` either way. Only for `strategy` in `"scipy"`/`"ultranest"`/`"hybrid"`; no effect for `"grid"`. | Integer `> 0` | `200` |
| `sparsify_grid` | Traces contour perimeters in 2D space to skip resolving deep interior/exterior nodes. | `True`, `False` | `False` |
| `n_restarts` | Number of distinct starting points tried per DATA fit (the unconditional fit and every 1D/2D conditional fit), keeping whichever converges to the lowest NLL. Only affects `strategy="scipy"`'s DATA fit -- not the MC toy fits (already well-seeded from the conditional MLE), not other strategies. `res.success` is always checked and non-convergence retried once regardless of this setting; `n_restarts` controls additional restarts beyond that. | Integer `> 0` | `1` |
| `neighbor_seeding` | When True, seeds each 1D/2D scan grid point's DATA fit from an adjacent, already-evaluated grid point's profiled parameters instead of always starting from the bounds midpoint (whenever used, the effective restart count for that point is also raised to at least 2, so a bad neighbor optimum can't silently cascade forward). Only affects `strategy="scipy"`'s DATA fit -- not the MC toy fits, not other strategies. | `True`, `False` | `True` |
| `save_log` | Pipes output directly to a persistent text log file. | `True`, `False` | `True` |
| `save_directory` | Directory path where final results, plots, and checkpoints reside. | String (path) | `"output/example_fc_output"` |
| `output_file` | Prefix for the serialized `.npz` and `.json` result data structures. | String | `"fc_results"` |

---

## Execution Strategies & Algorithmic Optimizations

PyFC provides several comprehensive levers to optimize computation time versus robustness based on the complexity of your likelihood surface.

```text
[Orchestrator] --> Iterates over Grids
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
PyFC is built to handle arbitrary $N$-dimensional physics models. You provide the full parameter space via the `grids` argument. The orchestrator automatically evaluates your parameter space systematically:
* **1D Intervals:** The framework iterates through every single parameter provided in the `grids` list. For each parameter, it treats it as the primary dimension while automatically profiling (maximizing the likelihood over) all other parameters.
* **2D Intervals:** The framework automatically generates and scans every unique pair of parameters from your `grids`, profiling out the remaining parameters not included in that specific 2D combination.

### Optimizer Strategy (`strategy`)
*   **`"scipy"` (L-BFGS-B)**: Highly recommended for most physics applications. It utilizes bounding constraints and analytical approximations of the gradient to find likelihood minima incredibly quickly. It assumes a relatively smooth parameter space. *Bounds Logistics:* Bounds are automatically inferred from the minimum and maximum values of the corresponding parameter arrays you pass in the `grids` list. You do not need to pass a separate bounds argument.
*   **`"ultranest"`**: Utilizes Nested Sampling. Extremely robust against complex, multi-modal likelihood surfaces where standard gradient minimizers get trapped in local minima. It is much slower than SciPy but guarantees finding the global minimum.
*   **`"hybrid"`**: A balanced approach. Uses UltraNest to find the global unconditional best-fit (which happens only once), and uses SciPy for the thousands of conditional minimizations during the profile scanning and MC toy fitting.
*   **`"grid"`**: Brute-force scanning over the provided parameter nodes. Safest but computationally restrictive. Scales poorly with $N > 2$ dimensions.
*   **Error Handling:** For `strategy="scipy"`/`"hybrid"` fits (both DATA fits and MC toy fits), PyFC always checks `res.success`. If a fit does not converge, it is retried once from a randomly perturbed starting point; if that retry also fails to converge, PyFC emits a `warnings.warn` message identifying which fit failed. **No toy or grid point is discarded or regenerated** -- whatever result comes out of a fit (converged or not) is used exactly like any other, so $N_{\text{toys}}$ is always preserved by construction, never by replacement.

### Finite MC Corrections (`use_finite_mc_correction_binned`)
When Monte Carlo templates are generated from limited statistics, treating the expected counts $\mu_i$ as absolute fixed truths leads to overconfidence. Enabling this flag triggers the continuous Poisson-Gamma mixture model formulation (see Mathematics section below) to strictly penalize the likelihood based on the simulated variance ($\sigma_i^2$) in each bin.

### Contour Edge Tracing (`sparsify_grid`)
Calculating $N_{\text{toys}}$ for every node in a $100 \times 100$ 2D grid is computationally wasteful. Enabling `sparsify_grid` activates a heuristic algorithm:
1. Selects a sparse, widely spaced sub-grid of coordinate nodes.
2. Evaluates the data test statistic ($t_{\text{data}}$) and generates complete toy distributions to find exact thresholds ($t_{\text{critical}}$) *only* at those sparse nodes.
3. Fits a Scipy `RectBivariateSpline` to interpolate the critical threshold surface across the rest of the grid.
4. Locates the decision boundary (the "edge" of the contour where $t_{\text{data}} \approx t_{\text{critical}}$).
5. Evaluates exact data fits and expensive MC toys on the specific high-resolution cells lying strictly on this perimeter to perfect the contour edge, drastically cutting runtime.

**Known limitation:** this defaults to `False` because step 4/5's boundary refinement is currently a single, non-iterative pass with a 1-cell-wide halo around the sparse coarse nodes. For grids much larger than roughly 20x20 per axis (i.e. the exact regime this feature targets), that halo does not reach deep interior/exterior cells far from any coarse node -- those cells are never evaluated and are force-excluded from the accepted region regardless of their true status. Opt in with care and verify against a `sparsify_grid=False` reference run on a representative grid size before trusting the result.

### Adaptive Toy Generation (`adaptive_toys`, `toy_batch_size`)
For `strategy` in `"scipy"`, `"ultranest"`, or `"hybrid"` (no effect for `"grid"`), toys for a given grid point are generated in batches of `toy_batch_size` rather than all $N_{\text{toys}}$ at once. This bounds peak memory (mainly relevant for `likelihood_type="unbinned"`, where each toy is a variable-size event array held in a `ProcessPoolExecutor`) and gives `adaptive_toys` a checkpoint at which to decide whether more toys are needed.

When `adaptive_toys=True`, after each batch PyFC computes the running fraction of toys with $t_{\text{toy}} \geq t_{\text{data}}$ -- an estimate of the p-value for that point's accept/reject decision -- and a strict (99.9% confidence) Wilson score interval around it. If that interval lies entirely on one side of the target significance $\alpha = 1 - \text{CL}$ (using the *largest* requested `cl`, the hardest verdict to settle, if several are given), the accept/reject verdict cannot plausibly flip with more toys, and generation stops early for that point. Early-stopping is never considered before 100 toys, regardless of how extreme the running estimate looks -- small-sample binomial confidence intervals are unreliable.

This only meaningfully saves time for points *far* from the accept/reject boundary (deep inside or deep outside the confidence region); points near the boundary will and should run the full `n_toys`. Validated (see `tests/test_adaptive_toys.py`) to reach the same accept/reject verdicts as `adaptive_toys=False` on both isolated toy-generation calls and full `compute_fc_intervals` runs.

### Handling Joint/Simplex-Constrained Parameters

`bounds_list` (built automatically from your `grids`) only expresses **independent per-parameter box constraints** -- `param_i` must lie in `[lo_i, hi_i]`, with no notion of a relationship between two different parameters. Neither L-BFGS-B/SLSQP (SciPy) nor the nested-sampling prior transform (UltraNest) have any native concept of a *joint* constraint like a simplex (`a + b <= 1`) or a sphere (`a^2 + b^2 <= 1`).

If your physical model has such a constraint and you don't tell PyFC about it, the optimizer's default starting guess (the bounds midpoint) can land in the unphysical region, and a naive `compute_rates_func` that just returns a degenerate/zero rate there gives gradient-based optimizers nothing to climb out with (see the NLL smoothing discussion above -- this was the original motivating failure case for that fix). **Do not work around this with a hand-rolled smooth penalty function multiplying your rate function** -- it's fragile to tune (too narrow a penalty width reproduces the same flat-gradient trap) and PyFC has two purpose-built mechanisms instead.

**Worked example:** a 6-parameter neutrino flavor-fraction fit where `f_e` (index 0) and `f_mu` (index 1) are subject to `f_e + f_mu <= 1`, since the derived third fraction `f_tau = 1 - f_e - f_mu` must stay non-negative.

**Mechanism 1 -- `bounds_func`: use when the constraint only involves the scan's currently-FIXED test parameter.** During a 1D/2D profile scan, one (or two) parameters are fixed to a grid test value while the rest are profiled (free). If your constraint couples a free nuisance parameter to whichever parameter(s) happen to be fixed right now, `bounds_func` lets you tighten that free parameter's box bound itself -- so the box is always physical, and the optimizer's bounds-midpoint starting guess is automatically valid too:

```python
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
```

`bounds_func` is called as `bounds_func(fixed_values, free_indices, default_bounds_list)`, where `fixed_values` is `{param_index: test_value}` for whichever parameter(s) the current scan step has fixed (an empty dict `{}` during the unconditional fit, since nothing is fixed there), `free_indices` are the indices being optimized (in the order your returned list must match), and `default_bounds_list` is the full, untouched `bounds_list` (indexed by the *original* parameter index, not by position in `free_indices`). Return one `(lo, hi)` tuple per entry of `free_indices`.

**Limitation:** `bounds_func` cannot express a constraint between two parameters that are *both* free at the same time (e.g. if `f_e` and `f_mu` are never the scan's fixed test parameter in a particular grid, such as while profiling a 2D scan over two unrelated parameters). That case needs mechanism 2.

**Mechanism 2 -- `constraints`: use for constraints among multiple simultaneously-free nuisance parameters.** Pass a list of `scipy.optimize.LinearConstraint`/`NonlinearConstraint` objects, expressed in the **full** parameter-vector space (not just the free subspace -- PyFC projects them down internally, substituting whichever value(s) the scan currently has fixed):

```python
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
```

For the SciPy path, supplying `constraints` automatically switches the optimizer from `L-BFGS-B` to `SLSQP` (the only two of PyFC's supported methods that accept `constraints`; `trust-constr` is also available via `scipy_method="trust-constr"` for better-conditioned but slower fits). For the UltraNest path, since nested sampling doesn't use gradients, a violated constraint simply makes that point's log-likelihood a very large negative number -- no special method switch needed.

**Which one should I use?** If the constraint only ever couples the scan's currently-fixed test parameter(s) to a free nuisance parameter, `bounds_func` is cheaper (it tightens the box itself, so the optimizer never has to work near a boundary at all) and is the mechanism used in the example above for a 1D/2D scan *over* `f_e` or `f_mu`. If your scan fixes some *other* parameter and profiles `f_e` and `f_mu` together as simultaneously-free nuisance parameters, only `constraints` can express that. The two mechanisms compose: pass both if you have constraints of both shapes, and they can be used together on the very same fit call.

---

## HPC & Parallelization Guidelines

PyFC is explicitly engineered to scale across multi-core High-Performance Computing (HPC) nodes. Parallelization is handled contextually based on your analysis type:
*   **Binned Likelihoods**: Exploits GIL-bypassing via Numba-compiled C-extensions. Toy generation and fitting are parallelized at the NumPy array level.
*   **Unbinned Likelihoods**: Utilizes Python's `concurrent.futures.ProcessPoolExecutor` to spawn independent worker processes for continuous function evaluation.

**HPC Do's and Don'ts:**
*   **DO map `num_cores` to your Slurm/PBS allocations.** If you set `num_cores=0` (or leave it `null` / `None`), PyFC requests all available threads on the hardware. If running via a Slurm workload manager, it is highly recommended to explicitly pass the allocated CPUs to avoid overcommitting: 
    ```python
    import os
    config['num_cores'] = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    ```
*   **DO utilize whole nodes.** Because the unbinned optimizer relies on multi-processing, it scales near-linearly. Requesting exclusive nodes (e.g., 64 or 128 cores) will drastically reduce Feldman-Cousins runtime.
*   **DO monitor memory scaling for unbinned analyses.** Toys are submitted to the `ProcessPoolExecutor` in batches of `toy_batch_size` rather than all $N_{\text{toys}}$ at once, bounding how many toy event arrays / in-flight result objects are alive at any moment. Running $N_{\text{toys}} = 5000$ on 128 cores can still cause memory exhaustion (OOM slurm kills) if your event arrays are massive. If this occurs, reduce `toy_batch_size` from 200 to 50. (Only applies to `strategy` in `"scipy"`/`"ultranest"`/`"hybrid"` -- `strategy="grid"` doesn't batch, since its toy generators are a single vectorized `numba` call rather than a pool of per-toy tasks.)
*   **DON'T enable `save_log` for massive array jobs.** If you are submitting hundreds of job arrays to an HPC, writing individual `.txt` logs continuously can bottleneck shared network file systems (NFS). Rely on the binary checkpointing instead.

### Advanced Integrations
Note that PyFC's native parallelization logic maps threads and processes strictly within a single node's resource limits. Multi-node distribution (e.g., communicating via MPI) is not natively built into the `orchestrator`. Users requiring cluster-wide multi-node deployment should consider manually sharding their `grids` and wrapping `compute_fc_intervals` within an `mpi4py` communicator.

**Note on UltraNest:**
If utilizing UltraNest for unbinned likelihoods via `strategy="ultranest"`, be aware that the `ProcessPoolExecutor` explicitly overrides the GIL, meaning each core evaluates its own entirely isolated UltraNest sampler sequence. PyFC does not use MPI-based sampler sharing under the hood; the nested sampling operates completely localized to the spawned sub-process during the toy fits. 

---

## Statistical Methodology & Mathematics

PyFC implements exact classical frequentist intervals. The code executes the methodology prescribed by Gary Feldman and Robert Cousins (1998) to solve the empty-set problem near physical boundaries.

### The Profile Likelihood Ratio
For a general parameter vector divided into parameters of interest $\boldsymbol{\theta}$ and nuisance parameters $\boldsymbol{\nu}$, we construct the Profile Likelihood Ratio (PLR) test statistic $t$:

$$t_{\text{data}}(\boldsymbol{\theta}) = -2 \ln \frac{\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})}{\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})}~,$$

where:
* $\mathcal{L}(\hat{\boldsymbol{\theta}}, \hat{\boldsymbol{\nu}} | \text{data})$ is the unconditional Maximum Likelihood Estimate (MLE) over the entire allowed parameter space.
* $\mathcal{L}(\boldsymbol{\theta}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})$ is the conditional MLE, evaluated at a fixed point $\boldsymbol{\theta}_{\text{test}}$, while profiling (maximizing) out the nuisance parameters $\boldsymbol{\nu}$.

By construction, $t \geq 0$. Lower values indicate excellent agreement between the data and the test hypothesis.

### Binned Likelihood
For binned configurations, the likelihood is the product of independent Poisson probabilities across $N$ bins. PyFC evaluates the saturated (Baker-Cousins) form of the Negative Log-Likelihood -- the likelihood ratio relative to the saturated model ($\mu_i = n_i$) -- dropping the data-only constant $\ln(n_i!)$ term:

$$-\ln \mathcal{L}_{\text{Poisson}} = \sum_{i=1}^{N} \left( \mu_i(\boldsymbol{\theta}) - n_i + n_i \ln\frac{n_i}{\mu_i(\boldsymbol{\theta})} \right)~,$$

PyFC's implementation carries an extra overall factor of 2 relative to this expression -- it computes $-2\ln \mathcal{L}_{\text{Poisson}}$ directly, matching the Baker-Cousins/Wilks convention used for the PLR test statistic $t$ itself (see above), so no separate doubling is needed when forming $t = \text{NLL}_{\text{cond}} - \text{NLL}_{\text{uncond}}$.

**Finite Monte Carlo Correction:**
If `use_finite_mc_correction_binned` is True, the pure Poisson distribution is convoluted with a Gamma prior, shifting the likelihood to a Negative Binomial distribution. PyFC implements the marginalized effective likelihood $\mathcal{L}_{\text{Eff}}$ derived by Argüelles, Schneider & Yuan (2019) [arXiv:1901.04645] -- see "Methodology References" below -- with

$$\alpha_i = \frac{\mu_i^2}{\sigma_i^2} + 1~, \qquad \beta_i = \frac{\mu_i}{\sigma_i^2}~,$$

$$\ln \mathcal{L}_{\text{FiniteMC}} = \sum_{i=1}^{N} \left[ \alpha_i \ln \beta_i + \ln \Gamma(n_i + \alpha_i) - (n_i + \alpha_i) \ln(1 + \beta_i) - \ln \Gamma(\alpha_i) \right]~, \qquad -2\ln \mathcal{L}_{\text{FiniteMC}} = \text{NLL}_{\text{FiniteMC}}~,$$

where $\sigma_i^2$ is the variance (sum of squared MC weights) in bin $i$. The "+1" in $\alpha_i$ is not a typo -- it is what distinguishes this (recommended, best-coverage) parameterization from the paper's alternative moment-matching-only choice ($\alpha_i = \mu_i^2/\sigma_i^2$, no "+1"), which the paper's own coverage tests show performs worse. The data-only constant $\ln(n_i!)$ term is dropped, matching the same convention used for the EUML formula below.

### Extended Unbinned Maximum Likelihood
When binning causes unacceptable information loss (e.g., highly complex kinematics with low event counts), PyFC evaluates the unbinned likelihood. Instead of bin counts, it uses the exact coordinates $x_j$ of the $M$ observed events.

$$-\ln \mathcal{L}_{\text{EUML}} = N_{\text{expected}}(\boldsymbol{\theta}) - \sum_{j=1}^{M} \ln \lambda(x_j | \boldsymbol{\theta})~,$$

where $N_{\text{expected}}$ is the integral of the total rate, and $\lambda(x_j)$ is the non-normalized rate evaluated strictly at the properties of event $j$.

---

## Outputs, Plots, and Checkpointing

### The Checkpoint Engine (`warm_start`)
Feldman-Cousins calculations are highly resource-intensive and often run on shared HPC clusters subject to preemption limits (e.g., Slurm time limits). By default, `warm_start` is set to `True`. PyFC dynamically writes its state to `checkpoint_fc.npz` inside your `save_directory` (default: `output/example_fc_output/`) after processing each 1D slice of the parameters of interest. 

If your script is interrupted, simply run it again. PyFC will detect the `checkpoint_fc.npz` file, rigorously verify that your newly requested parameter grids match the saved geometry exactly, and resume from the last completed checkpoint -- **not** from the exact grid point of interruption: checkpoints are written after each full 1D parameter scan and each full 2D pair scan completes (see `_save_fc_archive`'s call sites in `orchestrator.py`), so an interruption mid-scan re-does that one in-progress parameter/pair, not the whole run.

**Critical Restart Warning:** While PyFC verifies your grid dimensions on restart, it does *not* rigorously verify hyperparameter modifications. If you abort a run and alter settings like `strategy`, `likelihood_type`, or `n_toys`, you **must** manually delete the `checkpoint_fc.npz` file before running again, otherwise, the system will permanently merge structurally corrupted pseudo-experiment p-values into your finalized thresholds. *(Note: Manual deletion is only necessary for aborted or preempted runs; successfully completed runs automatically clean up their checkpoint files).*

### Stored Results & Custom Plotting
Upon successful completion, the pipeline outputs final structures directly to your `save_directory` (default: `output/example_fc_output/`):
* **`fc_results.json`**: A dictionary containing structural metadata, evaluated interval bounds, exact confidence levels, the global unconditional best-fit coordinate, and the explicit `data_uncond_nll` required for cross-model ratio comparisons. **Important:** The internal keys nested under `"1d_intervals"` and `"2d_intervals"` are strictly mapped by integer index (e.g., `"param1"`, `"param2"`), entirely ignoring any custom string values you passed to the `param_names` configuration list.

  **`interval_bounds` schema:** `1d_intervals.{param}.thresholds.{cl}.interval_bounds` is always a **list of `[lo, hi]` pairs**, one per maximal *contiguous* run of accepted grid points -- never a single flat `[lo, hi]` pair, even when there's only one interval. Under the Feldman-Cousins unified construction, the accepted region for a parameter can in principle be genuinely disconnected near certain physical boundaries (e.g. a non-monotonic rate function crossing the data at more than one point); reporting a single min/max span would silently merge those separate intervals together, including whatever rejected gap sits between them. A two-interval example:
  ```json
  "interval_bounds": [[0.5, 0.9], [1.3, 1.6]]
  ```
  and the common single-interval case is still a one-element list, `[[0.5, 0.9]]`, for a consistent schema regardless of how many disjoint pieces the accepted region has. See `04_pyfc_algorithmic_features_tutorial.ipynb`'s "Disconnected accepted intervals" section for a concrete worked example. (2D `interval_bounds` are not precomputed -- `2d_intervals` stores the full accepted boolean grid instead, from which any 2D region shape, connected or not, can already be reconstructed directly.)
* **`fc_results.npz`**: A highly compressed NumPy archive containing exact parameter matrices and boolean masks. The keys are built dynamically based on the integer index of the parameter arrays:

**Available `.npz` Keys (Where `{i}` and `{j}` correspond to 1-based grid array indices like 1, 2, 3):**

| Key | Description | Shape |
| :--- | :--- | :--- |
| `best_fit` | Global unconditional MLE parameter vector -- the same point used as the numerator of every PLR test statistic. Also the two scalars called out above as "required for cross-model ratio comparisons". | `(n_params,)` |
| `data_uncond_nll` | Global unconditional NLL at `best_fit`, evaluated on the real data. | scalar |
| `grid_p{i}` | 1D parameter space meshgrid evaluating parameter `i`. | `(N,)` |
| `1d_test_p{i}` | Duplicate of `grid_p{i}` (identical array, stored under both keys). | `(N,)` |
| `1d_t_data_p{i}` | 1D PLR test statistic evaluated on the real data. | `(N,)` |
| `1d_prof_params_p{i}` | Full profiled parameter vector (all `n_params` coordinates, not just `p{i}`) at each 1D scan point. | `(N, n_params)` |
| `1d_t_critical_p{i}_{cl}` | 1D Interpolated MC threshold surface limits. | `(N,)` |
| `1d_accepted_p{i}_{cl}` | 1D boolean limits profiling other parameters. | `(N,)` |
| `2d_test_p{i}_p{i}p{j}` | 1D grid for parameter `i` as scanned in the `(i, j)` pair (duplicate of `grid_p{i}`). | `(N,)` |
| `2d_test_p{j}_p{i}p{j}` | 1D grid for parameter `j` as scanned in the `(i, j)` pair (duplicate of `grid_p{j}`). | `(M,)` |
| `2d_t_data_p{i}p{j}` | 2D PLR test statistic evaluating a joint region. | `(N, M)` |
| `2d_t_critical_p{i}p{j}_{cl}` | 2D threshold surfaces for the combination. | `(N, M)` |
| `2d_accepted_p{i}p{j}_{cl}` | 2D boolean masks (True = inside contour). | `(N, M)` |

**Code Snippet: Generating Custom Contours**
While PyFC automatically saves default matrix plots via Matplotlib, you will likely want to format your own figures for publication. You can easily ingest the output files to do so. 

```python
import numpy as np
import matplotlib.pyplot as plt
import json

# 1. Load the numerical matrices
results = np.load('output/example_fc_output/fc_results.npz')
X = results['grid_p1']
Y = results['grid_p2']

# Ensure correct meshgrid orientation for 2D plotting
X_mesh, Y_mesh = np.meshgrid(X, Y, indexing='ij')

# True indicates the parameter coordinate is inside the 90% CL limit
mask_90 = results['2d_accepted_p1p2_0.9'] 

# 2. Load the metadata
with open('output/example_fc_output/fc_results.json', 'r') as f:
    meta = json.load(f)
best_fit = meta['best_fit']

# 3. Plot the allowed region
plt.figure(figsize=(8, 6))
# contourf uses the boolean mask to shade the allowed parameter space
plt.contourf(X_mesh, Y_mesh, mask_90, levels=[0.5, 1.5], colors=['#1f77b4'], alpha=0.3)

# Plot the global best fit point
plt.plot(best_fit[0], best_fit[1], 'r*', markersize=12, label='Global Best Fit')

plt.xlabel("Parameter 1")
plt.ylabel("Parameter 2")
plt.title('Custom 90% CL Feldman-Cousins Region')
plt.legend()
plt.show()
```
*(Warning: The framework natively plots all dimensional combinations. Requesting a grid space with $N > 4$ parameters causes combinatorial explosions in both processing time and the size of the final plotted matrix axes).*

---

## Common Recipes and Questions

### Reproducibility and Random Seeds
Monte Carlo pseudo-experiment generation relies heavily on random number generators. To ensure exact reproducibility across multiple runs—especially when utilizing multi-processing via `ProcessPoolExecutor`—you must initialize the random seed in your main execution script before calling the orchestrator (e.g., `np.random.seed(42)`).

### Troubleshooting
*   **ValueError: operands could not be broadcast together**: This almost always means the output NumPy array shape generated by your `compute_rates_func` does not exactly match the shape of the `observed_counts` data array you passed to the orchestrator.
*   **UltraNest Convergence Warnings**: If you receive warnings that the nested sampler is struggling to converge, consider increasing your target parameter bounds (by expanding the ranges in your `grids`) or verifying that your likelihood surface does not contain unhandled `NaN` or `inf` values.

### Frequently Asked Questions
**Q: How many toys should I use?**
A: For a 68% CL interval (1-sigma), 200-1000 toys are often sufficient. For a 90% or 95% limit, 2000-5000 toys are required to smoothly resolve the tail of the test statistic distribution. Ensure `n_toys` is large enough that $N_{\text{toys}} \times (1 - \alpha) \gg 1$.

**Q: I have a parameter that represents a systematic uncertainty. How do I profile it?**
A: Simply pass a `np.linspace()` grid for that parameter into the `grids` list. The framework automatically profiles (maximizes) any parameter in the `grids` list that is not the direct target of the current 1D or 2D interval scan.

**Q: Two of my parameters have a joint constraint (e.g. `a + b <= 1`, or they lie on a simplex/sphere) -- how do I express that?**
A: `bounds_list` only supports independent per-parameter box bounds, with no native notion of a relationship between two parameters. Do not work around this with a hand-rolled penalty function multiplying your rate -- see [Handling Joint/Simplex-Constrained Parameters](#handling-jointsimplex-constrained-parameters) for the two purpose-built mechanisms (`bounds_func` and `constraints`) and a full worked example.

---

## Contributing

We welcome contributions to PyFC, including bug reports, feature requests, and code modifications! 
1. Open an issue on the GitHub repository to discuss the proposed change.
2. Fork the repository and create a feature branch (`git checkout -b feature/new-optimizer`).
3. Ensure all tests pass (`pytest tests/`) and code is fully documented. Note that all Pull Requests must also pass the automated Continuous Integration (CI) pipeline via GitHub Actions before they can be merged.
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