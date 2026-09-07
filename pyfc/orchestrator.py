"""
Feldman-Cousins Orchestrator Module

This file acts as the central hub and main driver for the PyFC package. 
It coordinates the execution of the Feldman-Cousins confidence interval construction, 
managing the interplay between the data, physical models, likelihood optimizers, 
and Monte Carlo toy generators. It handles checkpointing, parameter space scanning 
(including an optimized 2D grid sparsification algorithm), and data archiving.

Usage (Command Line Interface):
-------------------------------
This script can be executed directly from the terminal to run a demonstration 
of the PyFC pipeline using automatically generated mock data (binned or unbinned).

To run the demonstration using the current configuration (or `fc_config.json`):
    $ python orchestrator.py
    
Alternatively, if the PyFC package is installed in your Python environment:
    $ python -m pyfc.orchestrator
    
You can override configuration parameters directly via command-line arguments. 
For example, to run an unbinned analysis with 500 toys using the SciPy optimizer:
    $ python orchestrator.py --likelihood_type unbinned --n_toys 500 --strategy scipy

Created: v0.1.0 (July 24, 2026)
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import itertools
import json
import logging
import os
import warnings

import numpy as np

from .binned import (
    NUMBA_AVAILABLE,
    conditional_fit_grid_1d,
    conditional_fit_grid_2d,
    generate_and_fit_toys_grid_1d,
    generate_and_fit_toys_grid_2d,
    set_num_threads,
    unconditional_fit_grid,
)

# --- Module Imports ---
from .config import parse_arguments
from .optimizers import (
    SCIPY_AVAILABLE,
    ULTRANEST_AVAILABLE,
    conditional_fit_1d_scipy,
    conditional_fit_1d_ultranest,
    conditional_fit_2d_scipy,
    conditional_fit_2d_ultranest,
    unconditional_fit_scipy,
    unconditional_fit_ultranest,
)
from .plotting import generate_corner_plot
from .toys import generate_and_fit_toys_python
from .unbinned import (
    conditional_fit_grid_unbinned_1d,
    conditional_fit_grid_unbinned_2d,
    generate_and_fit_toys_grid_unbinned_1d,
    generate_and_fit_toys_grid_unbinned_2d,
    unconditional_fit_grid_unbinned,
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to safely serialize NumPy data types.
    
    Standard `json` libraries cannot handle `np.ndarray` or specific NumPy 
    numerical types (like `np.float64`). This intercepts those objects during 
    the `json.dump` process and converts them to standard Python equivalents.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def _save_fc_archive(results, grids, output_path, cl, compute_1D_intervals, compute_2D_intervals, n_params):
    """
    Serializes the computational state of the Feldman-Cousins environment to a binary `.npz` archive.
    
    This function is critical for the "warm start" (checkpointing) feature. It saves 
    all currently evaluated test statistics, parameter grids, and boolean acceptance maps 
    so that execution can be halted and resumed without losing expensive MC toy computations.

    Parameters:
    -----------
    results : dict
        The central dictionary holding all computed arrays and limits.
    grids : list of np.ndarray
        The parameter scan grids used for the evaluation.
    output_path : str
        The file path for the `.npz` archive.
    cl : list of float
        Confidence levels being evaluated.
    compute_1D_intervals, compute_2D_intervals : bool
        Flags indicating which dimensional modes are active.
    n_params : int
        Total number of parameters in the model.
        
    Returns:
    --------
    None (Writes file to disk)
    """
    save_dict = {
        "best_fit": results["best_fit"],
        "data_uncond_nll": results.get("data_uncond_nll", np.nan)
    }
    
    # Store parameter grids directly to verify states on resume
    for idx, g in enumerate(grids):
        save_dict[f"grid_p{idx+1}"] = g
        
    if compute_1D_intervals:
        for p_idx in range(n_params):
            save_dict[f"1d_test_p{p_idx+1}"] = results[f"1d_test_p{p_idx+1}"]
            save_dict[f"1d_t_data_p{p_idx+1}"] = results[f"1d_t_data_p{p_idx+1}"]
            save_dict[f"1d_prof_params_p{p_idx+1}"] = results[f"1d_prof_params_p{p_idx+1}"]
            for c in cl:
                save_dict[f"1d_t_critical_p{p_idx+1}_{c}"] = results[f"1d_t_critical_p{p_idx+1}"][c]
                save_dict[f"1d_accepted_p{p_idx+1}_{c}"] = results[f"1d_accepted_p{p_idx+1}"][c]

    if compute_2D_intervals and n_params > 1:
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            save_dict[f"2d_test_p{fix_A+1}_{pair_name}"] = results[f"2d_test_p{fix_A+1}_{pair_name}"]
            save_dict[f"2d_test_p{fix_B+1}_{pair_name}"] = results[f"2d_test_p{fix_B+1}_{pair_name}"]
            save_dict[f"2d_t_data_{pair_name}"] = results[f"2d_t_data_{pair_name}"]
            for c in cl:
                save_dict[f"2d_t_critical_{pair_name}_{c}"] = results[f"2d_t_critical_{pair_name}"][c]
                save_dict[f"2d_accepted_{pair_name}_{c}"] = results[f"2d_accepted_{pair_name}"][c]
                
    np.savez(output_path, **save_dict)


def _find_contiguous_intervals(test_points, accepted):
    """
    Finds every maximal contiguous run of `accepted=True` in `test_points`
    (assumed ascending, e.g. built from `np.linspace`) and returns one
    `[lo, hi]` pair per run, in scan order.

    Under the Feldman-Cousins unified construction, the accepted region
    for a parameter can in principle be genuinely disconnected near
    certain physical boundaries -- taking a single min/max across all
    accepted points would silently merge separate intervals into one,
    including whatever rejected gap sits between them.

    Parameters:
    -----------
    test_points : array_like, 1D
        The scanned parameter values, in ascending order.
    accepted : array_like of bool, 1D
        Same length as `test_points`; True where that point is inside the
        confidence region.

    Returns:
    --------
    list of [float, float]
        One `[lo, hi]` pair per maximal contiguous accepted run, in scan
        order. Empty list if nothing is accepted.
    """
    intervals = []
    run_start = None
    prev_val = None
    for val, is_acc in zip(test_points, accepted):
        if is_acc and run_start is None:
            run_start = val
        elif not is_acc and run_start is not None:
            intervals.append([run_start, prev_val])
            run_start = None
        prev_val = val
    if run_start is not None:
        intervals.append([run_start, prev_val])
    return intervals


def _save_fc_json(results, output_path, cl, compute_1D_intervals, compute_2D_intervals, n_params):
    """
    Exports computed 1D and 2D intervals to an external, human-readable JSON format.
    
    This provides an alternative to the binary `.npz` archive, facilitating 
    interoperability with web frameworks, external plotting tools, or other languages.

    Parameters:
    -----------
    results : dict
        The populated results dictionary.
    output_path : str
        The destination JSON file path.
    cl : list of float
        The confidence levels evaluated.
    compute_1D_intervals, compute_2D_intervals : bool
        Flags indicating which intervals were processed.
    n_params : int
        Number of parameters in the model.
        
    Returns:
    --------
    None (Writes file to disk)
    """
    json_dict = {
        "best_fit": results["best_fit"],
        "data_uncond_nll": results.get("data_uncond_nll", None)
    }
    
    if compute_1D_intervals:
        json_dict["1d_intervals"] = {}
        for p_idx in range(n_params):
            p_key = f"param{p_idx+1}"
            test_points = results.get(f"1d_test_p{p_idx+1}")
            
            json_dict["1d_intervals"][p_key] = {
                "test_points": test_points,
                "t_data": results.get(f"1d_t_data_p{p_idx+1}"),
                "prof_params": results.get(f"1d_prof_params_p{p_idx+1}"),
                "thresholds": {}
            }
            
            for c in cl:
                accepted = results[f"1d_accepted_p{p_idx+1}"][c]
                interval_bounds = []

                # One [lo, hi] pair per maximal contiguous accepted run, not
                # a single min/max across all accepted points -- the latter
                # would silently merge disconnected accepted regions (a real
                # possibility under the Feldman-Cousins unified construction)
                # into one interval spanning whatever rejected gap sits
                # between them.
                if test_points is not None and accepted is not None:
                    interval_bounds = _find_contiguous_intervals(test_points, accepted)
                
                json_dict["1d_intervals"][p_key]["thresholds"][str(c)] = {
                    "t_critical": results[f"1d_t_critical_p{p_idx+1}"][c],
                    "accepted": accepted,
                    "interval_bounds": interval_bounds
                }

    if compute_2D_intervals and n_params > 1:
        json_dict["2d_intervals"] = {}
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            json_dict["2d_intervals"][pair_name] = {
                f"test_p{fix_A+1}": results.get(f"2d_test_p{fix_A+1}_{pair_name}"),
                f"test_p{fix_B+1}": results.get(f"2d_test_p{fix_B+1}_{pair_name}"),
                "t_data": results.get(f"2d_t_data_{pair_name}"),
                "thresholds": {}
            }
            for c in cl:
                json_dict["2d_intervals"][pair_name]["thresholds"][str(c)] = {
                    "t_critical": results[f"2d_t_critical_{pair_name}"][c],
                    "accepted": results[f"2d_accepted_{pair_name}"][c]
                }

    with open(output_path, 'w') as f:
        json.dump(json_dict, f, cls=NumpyEncoder, indent=4)


def compute_fc_intervals(data, grids, compute_rates_func=None, generate_toy_func=None,
                         pdf_components=None,
                         cl=None, n_toys=500, strategy="scipy", num_cores=None, verbose=1,
                         adaptive_toys=True, toy_batch_size=200,
                         sparsify_grid=False, warm_start=True,
                         likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                         output_file=None, save_log=True, save_directory="output/example_fc_output",
                         use_finite_mc_correction_binned=True, S_sumw2=None, B_sumw2=None,
                         compute_1D_intervals=True, compute_2D_intervals=True, param_names=None,
                         smooth_1d=True, smooth_2d=True, bounds_func=None,
                         constraints=None, scipy_method=None,
                         n_restarts=1, neighbor_seeding=True, extra_nll=None):
    """
    Main execution pipeline for the Feldman-Cousins unified approach.
    
    Statistical Theory:
    The Feldman-Cousins method constructs confidence intervals (or regions) that transition 
    smoothly between upper limits and two-sided bounds. It relies on the Profile Likelihood 
    Ratio as the test statistic to determine ordering.
    
    For a given parameter point $\theta_{\text{test}}$, the data test statistic is:
    $$ t_{\text{data}} = -2 \\ln \frac{\\mathcal{L}(\theta_{\text{test}}, \\hat{\\hat{\boldsymbol{\nu}}} | \text{data})}{\\mathcal{L}(\\hat{\boldsymbol{\theta}} | \text{data})} $$
    where $\\hat{\boldsymbol{\theta}}$ are the unconditional Maximum Likelihood Estimate (MLE) 
    parameters, and $\\hat{\\hat{\boldsymbol{\nu}}}$ are the nuisance parameters profiled (maximized) 
    while fixing $\theta = \theta_{\text{test}}$. (Note: Since `calc_nll` returns $-2\\ln\\mathcal{L}$, 
    this simplifies in code to `cond_nll - data_uncond_nll`).
    
    Because the asymptotic distribution of $t$ (Wilks' theorem) may fail near physical boundaries, 
    we evaluate the exact critical threshold $t_{\text{critical}}$ at a required confidence level $\alpha$ 
    by generating and fitting Monte Carlo pseudo-experiments (toys) generated under the null 
    hypothesis $(\theta_{\text{test}}, \\hat{\\hat{\boldsymbol{\nu}}})$.
    
    The point $\theta_{\text{test}}$ is included in the final confidence set if:
    $$ t_{\text{data}} \\leq t_{\text{critical}}(\alpha) $$

    Parameters:
    -----------
    data : array_like
        The observed data: binned counts (any N-dimensional shape, e.g. a
        genuine 2D (E, cos_theta) histogram) or unbinned events.
    grids : list of np.ndarray
        A list of arrays, each defining the evaluation points for a parameter.
    compute_rates_func : callable
        User-provided mapping function matching parameters to physical
        expectations. For `likelihood_type="binned"`:
        `(params, S_sumw2, B_sumw2) -> (mu, sigma2)`, where `mu`/`sigma2`
        must have the same shape as `data`; any fixed template arrays should
        be referenced via closure/module-global rather than passed in. For
        `likelihood_type="unbinned"`: `(params, probs) -> (expected_total,
        p_events)`, where `probs` is the list of pre-evaluated
        `pdf_components` densities.
    generate_toy_func : callable, optional
        User-provided unbinned parametric bootstrap function.
    pdf_components : list of callable, optional
        Probability density functions, one per model component (e.g.
        signal, background, ...no longer limited to exactly two). Required
        for `likelihood_type="unbinned"`; ignored (with a warning) for
        `"binned"`. Each is evaluated exactly once per fit as `pdf(data)`
        and the resulting arrays are reused across every subsequent NLL
        evaluation in that fit.
    cl : list of float, optional
        Confidence levels to compute (e.g., [0.68, 0.90]).
    n_toys : int, optional
        Number of pseudo-experiments to generate per point.
    strategy : str, optional
        Optimizer strategy: "grid", "scipy", "ultranest", or "hybrid".
    num_cores : int, optional
        Number of parallel threads for toy generation. Defaults to None,
        which is forwarded as-is to the underlying
        `ThreadPoolExecutor`/`ProcessPoolExecutor`/`numba.set_num_threads`
        calls; all three treat None as "use every available hardware
        thread" (equivalent to `os.cpu_count()`). Pass an explicit integer
        to cap this, e.g. to match a Slurm/PBS core allocation.
    verbose : int, optional
        0 = Silent, 1 = Normal, 2 = Debug.
    adaptive_toys : bool, optional
        When True and strategy in ("scipy", "ultranest", "hybrid"), stops
        generating toys for a grid point early once its accept/reject
        verdict is statistically settled (a strict 99.9% Wilson confidence
        interval on the running p-value estimate lies entirely on one side
        of the target significance, and at least `toys.ADAPTIVE_MIN_TOYS`
        toys have been generated) -- see `toys._adaptive_stop_decision`.
        This only meaningfully saves time for points far from the accept/
        reject boundary; points near it will and should run the full
        `n_toys`. No effect for strategy="grid" (its toy generators are a
        `numba` `prange`-parallelized batch operation, not a natural fit
        for a sequential early-exit check).
    toy_batch_size : int, optional
        For strategy in ("scipy", "ultranest", "hybrid"), toys are
        submitted/collected from the executor in batches of this size
        instead of all `n_toys` at once, bounding how many toy datasets /
        in-flight results are alive at once (matters most for
        `likelihood_type="unbinned"`; binned toys are small fixed-size
        per-bin arrays with no comparable memory concern). Also the
        granularity at which `adaptive_toys` checks its stopping
        condition. Same total `n_toys` are generated either way when
        `adaptive_toys=False` -- this only changes memory/dispatch
        pattern, never the statistics. No effect for strategy="grid".
    warm_start : bool
        Algorithmic enhancement to reduce execution time via checkpointing.
    sparsify_grid : bool, optional
        For 2D scans, coarsely samples the grid and interpolates the
        t_critical surface, then refines only cells adjacent to the
        interpolated accept/reject boundary, instead of evaluating every
        cell. Defaults to False. Known limitation: refinement is a single,
        non-iterative pass with a 1-cell-wide halo around ~5-per-axis
        coarse nodes, so for grids much larger than ~20x20 per axis it
        under-counts the true accepted region (deep interior/exterior cells
        far from any coarse node are never evaluated and are force-excluded
        as a result). See the comment above the "Sparsification Array
        Tracing" block in this function's body for details.
    likelihood_type : str
        "binned" or "unbinned".
    S_mc_pool, B_mc_pool : array_like, optional
        Pool of events to bootstrap for unbinned toy generation.
    output_file, save_log, save_directory : str/bool
        I/O file structures. `save_directory` (default `"output/example_fc_output"`,
        relative to the current working directory) is where checkpoints and final
        results are written; see the README's "Outputs, Plots, and Checkpointing"
        section. `output_file` defaults to None, meaning no final `.npz`/`.json`
        archive is written -- only the in-memory `results` dict (this function's
        first return value) is populated; pass a string prefix (e.g. `"fc_results"`)
        to also write `{save_directory}/{output_file}.npz` and `.json`.
    use_finite_mc_correction_binned : bool, optional
        Toggle for Poisson-Gamma mixture likelihood.
    S_sumw2, B_sumw2 : array_like, optional
        Sum of squared MC weights (sumw2) per bin, for the finite MC correction.
    compute_1D_intervals, compute_2D_intervals : bool
        Switches for calculating 1D profiles or 2D joint contours.
    param_names : list of str, optional
        Names used for plot labels. Defaults to None, which auto-generates
        `["param1", "param2", ...]` matching `len(grids)` -- i.e. it always
        matches the actual number of parameters in this specific model,
        unlike hardcoding a fixed-length example list would.
    smooth_1d, smooth_2d : bool, optional
        Toggles interpolation smoothing for final plot outputs.
    bounds_func : callable, optional
        Advanced hook that lets the box bounds of the profiled (free)
        parameters depend on whatever parameter(s) are currently fixed by
        the scan, e.g. to express a joint constraint like f_e + f_mu <= 1
        without a user-side penalty function. Forwarded to whichever
        `conditional_fit_*`/`unconditional_fit_*` optimizer is active (scipy
        or ultranest strategies only; ignored for strategy="grid"). See the
        `optimizers.py` module docstring for the full signature contract and
        a worked example, and the README section "Handling joint/simplex-
        constrained parameters" for an end-to-end walkthrough. Limitation:
        this only handles constraints between a currently-FIXED test
        parameter and a free nuisance parameter -- it cannot express a
        constraint between two parameters that are BOTH free at the same
        time (use `constraints` for that instead).
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints among the FULL n_params vector, for expressing
        constraints between multiple simultaneously-free nuisance parameters
        (which `bounds_func` cannot express). Forwarded to whichever scipy/
        ultranest optimizer is active; ignored for strategy="grid". For the
        scipy path, `method` switches from 'L-BFGS-B' to 'SLSQP' whenever
        constraints are non-empty (required, since L-BFGS-B doesn't support
        `constraints`); for the ultranest path, a violated constraint simply
        yields a very large negative log-likelihood at that point. When
        None/empty (default), behavior and method are UNCHANGED from before
        this parameter existed. See the `optimizers.py` module docstring and
        the README section "Handling joint/simplex-constrained parameters".
    scipy_method : str, optional
        Explicit override for the `scipy.optimize.minimize` method (e.g.
        'trust-constr' for better-conditioned but slower constrained fits).
        Takes precedence over the constraints-based default.
    n_restarts : int, optional
        Forwarded to the scipy fit functions for the DATA fits (Phase 0's
        unconditional fit and the Phase 1/2 conditional fits) -- NOT the MC
        toy fits, which are already well-seeded from the conditional MLE
        (`true_params`). Number of distinct starting points to try per fit,
        keeping whichever converges to the lowest NLL (see
        `optimizers._minimize_with_restarts`). Default 1 preserves pre-
        FIX-4 behavior (a single fit per point), except that `res.success`
        is now always checked, with a cheap perturbed retry and warning on
        non-convergence -- this applies even at the default n_restarts=1,
        for every scipy fit call including the toy fits.
    extra_nll : callable, optional
        ``extra_nll(params) -> float``, added to the NLL at every evaluation --
        the hook for SOFT constraints on nuisance parameters, i.e. a parameter
        with a preferred value and a width, which is how an external
        measurement of a systematic enters a physics likelihood.

        UNITS -- READ THIS. PyFC's NLL is ``-2 ln L``, not ``-ln L``. A Gaussian
        constraint of width ``s`` about a centre ``c`` therefore contributes

            ((x - c) / s) ** 2          <-- correct here

        and NOT the ``0.5 * ((x - c) / s) ** 2`` that appears in a ``-ln L``
        convention. Getting this wrong is silent: the fit still converges and
        the interval still looks reasonable, but every constraint is weaker by
        a factor sqrt(2), so a stated 4.6% prior actually acts as 6.5%. The
        check is that a lone parameter constrained only by this term must give
        a test statistic of exactly 1.0 at ``c + s`` -- see
        ``tests/test_extra_nll.py``, which pins that oracle.

        Receives the FULL parameter vector (length ``len(grids)``), in the same
        order as ``grids``, with any scan-fixed values already substituted --
        the same convention `constraints` uses. Writing against the full vector
        means one function works for the unconditional fit, the 1D scan and the
        2D scan alike, since the free subspace differs between them. Must return
        a finite float. Called once per NLL evaluation, so it should be cheap.

        Applies to every fit the construction performs: unconditional,
        conditional, AND the Monte Carlo toys. The toys matter most -- if the
        term reached the data fit but not the toys, the observed test statistic
        and its critical value would be computed under different likelihoods,
        and the frequentist coverage the whole method exists to guarantee would
        be silently lost.

        Distinct from `bounds_func` and `constraints`, which express HARD
        geometric relations on the parameter box (a simplex ``a + b <= 1``, or a
        free parameter's bound depending on a fixed one). Those declare regions
        allowed or forbidden; this adds a term to the likelihood and leaves the
        model untouched. Distinct also from the hand-rolled penalty the module
        docstring warns against, which multiplies the RATE and reproduces the
        flat-gradient trap; this is additive in log-likelihood space.

        With ``strategy="grid"`` it must be a numba-jitted callable
        (``@njit``), because the grid scan runs inside @njit code that cannot
        call an arbitrary Python function; a plain callable raises TypeError
        with instructions. Every other strategy accepts any Python callable.
        Note also that with ``likelihood_type="unbinned"``, toys are dispatched
        to a ProcessPoolExecutor, so an unpicklable `extra_nll` (a lambda or a
        closure) makes that pool fail and fall back to threads -- correct, but
        slower; define it at module level to keep the process pool.
    neighbor_seeding : bool, optional
        When True (default) and strategy="scipy", the DATA fit (not the MC
        toys, which are already seeded from the conditional MLE) at each 1D/
        2D scan grid point is seeded from an adjacent, already-evaluated
        grid point's profiled parameters, instead of always starting fresh
        from the bounds midpoint -- these loops already iterate the grid in
        order, so neighboring points' solutions are informative starting
        guesses. Whenever a neighbor seed is used, the effective number of
        restarts for that specific call is raised to at least 2 (the
        neighbor-seeded start plus a fresh bounds-midpoint start, keeping
        whichever is better), so a bad neighbor optimum can't silently
        cascade forward across many subsequent grid points. Set False to
        disable and always start from the bounds midpoint (the original
        behavior for the data fit).

    Returns:
    --------
    results : dict
        A massive dictionary containing all t_data, thresholds, accepted masks, 
        and best-fit geometries evaluated across the parameter space.
    fig : matplotlib.figure.Figure or None
        The Matplotlib Figure object generated by generate_corner_plot, or None 
        if plotting was skipped.
    """
    
    if compute_rates_func is None:
        raise ValueError("You must provide a valid `compute_rates_func` to define your physical model.")
    if likelihood_type == "unbinned" and not pdf_components:
        raise ValueError("You must provide a non-empty `pdf_components` list (of callables) for likelihood_type='unbinned'.")

    # `strategy="grid"` runs its scan inside @njit functions, which cannot call an
    # arbitrary Python callable -- numba needs a CPUDispatcher, exactly as it already does
    # for `compute_rates_func`. Checked up front because the alternative is a numba
    # TypingError raised from deep inside the compiler: measured at 316 characters over 9
    # lines, mentioning neither `njit` nor which argument was wrong. A user would have no
    # way to know the fix is a decorator.
    if extra_nll is not None and strategy == "grid":
        try:
            import numba
            from numba.core.registry import CPUDispatcher
            # Under NUMBA_DISABLE_JIT=1 the @njit decorator is a no-op returning the plain
            # Python function, so the CPUDispatcher check below would reject a perfectly
            # correct penalty. Nothing is compiled in that mode -- the "jitted" grid
            # fitters are ordinary Python and can call anything -- so the requirement does
            # not apply. This is not hypothetical: scripts/run_coverage.sh sets that flag
            # for most of its run, and the guard failed two tests there before this branch
            # existed.
            jit_disabled = bool(numba.config.DISABLE_JIT)
        except ImportError:  # pragma: no cover - numba is a hard dependency
            CPUDispatcher, jit_disabled = (), False
        if not jit_disabled and not isinstance(extra_nll, CPUDispatcher):
            raise TypeError(
                "extra_nll must be a numba-jitted function when strategy='grid', because "
                "the grid scan runs inside @njit code that cannot call a plain Python "
                "callable. Decorate it, e.g.:\n"
                "    from numba import njit\n"
                "    @njit(fastmath=True, nogil=True)\n"
                "    def my_penalty(params):\n"
                # No 0.5 factor: PyFC's NLL is -2lnL, so a width-s Gaussian contributes
                # ((x-c)/s)**2. An example carrying the -lnL form would teach the very
                # mistake the docstring warns about, in the one place a confused user is
                # guaranteed to be reading.
                "        return ((params[0] - 1.0) / 0.05) ** 2\n"
                "Alternatively use strategy='scipy', 'ultranest' or 'hybrid', which accept "
                "any Python callable."
            )
    if likelihood_type == "binned" and pdf_components:
        warnings.warn("`pdf_components` was supplied but is unused for likelihood_type='binned'; ignoring it.")

    if cl is None:
        cl = [0.68, 0.90]
        
    os.makedirs(save_directory, exist_ok=True)
    n_params = len(grids)
    ckpt_path = os.path.join(save_directory, "checkpoint_fc.npz")
    
    # Setup Logging Architecture
    run_logger = logging.getLogger("FC_Orchestrator")
    run_logger.setLevel(logging.INFO)
    if save_log and not run_logger.handlers:
        fh = logging.FileHandler(os.path.join(save_directory, 'fc_run.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        run_logger.addHandler(fh)
        
    def log_print(msg):
        if verbose > 0:
            print(msg)
        if save_log:
            run_logger.info(msg)

    # Coerce input configurations
    if isinstance(cl, (float, int)):
        cl = [float(cl)]
    if ULTRANEST_AVAILABLE and verbose < 2:
        logging.getLogger("ultranest").setLevel(logging.WARNING)
    if NUMBA_AVAILABLE and num_cores is not None:
        set_num_threads(num_cores)

    # Target significance for adaptive_toys' early-stopping decision: the
    # *smallest* alpha (largest requested CL) is the hardest accept/reject
    # verdict to settle, so requiring it to be settled before stopping
    # guarantees every other requested CL's verdict is settled too (they
    # all come from the same toy sample). Only used when adaptive_toys=True
    # and only affects strategy in ("scipy", "ultranest", "hybrid") --
    # strategy="grid"'s toy generators are unaffected (see their own
    # docstrings for why).
    adaptive_alpha = 1.0 - max(cl)
        
    # Setup likelihood variances for Finite MC
    if likelihood_type == "binned":
        if S_sumw2 is None:
            S_sumw2 = np.zeros_like(data)
        if B_sumw2 is None:
            B_sumw2 = np.zeros_like(data)
    else:
        S_sumw2, B_sumw2 = None, None

    log_print(f"--- FC Construction Initiated ({n_params}-Parameter Model) ---")
    log_print(f"Modes -> 1D Intervals: {compute_1D_intervals} | 2D Intervals: {compute_2D_intervals}")
    log_print(f"Strategy: {strategy.upper()} | Cores: {num_cores if num_cores else 'Max'} | Likelihood: {likelihood_type.upper()}")
    
    bounds_list = [(g[0], g[-1]) for g in grids]
    disable_tqdm = (verbose == 0) or not TQDM_AVAILABLE
    
    # --- Pre-allocate Architecture with NaNs ---
    # We use np.nan to explicitly identify points in the grid that have not 
    # yet been evaluated (vital for the checkpointing and sparsification logic).
    results = {
        "best_fit": np.full(n_params, np.nan),
        "data_uncond_nll": np.nan
    }

    if compute_1D_intervals:
        for p_idx in range(n_params):
            grid_test = grids[p_idx]
            results[f"1d_test_p{p_idx+1}"] = grid_test
            results[f"1d_t_data_p{p_idx+1}"] = np.full(len(grid_test), np.nan)
            results[f"1d_prof_params_p{p_idx+1}"] = np.full((len(grid_test), n_params), np.nan)
            results[f"1d_t_critical_p{p_idx+1}"] = {c: np.full(len(grid_test), np.nan) for c in cl}
            results[f"1d_accepted_p{p_idx+1}"] = {c: np.full(len(grid_test), False) for c in cl}

    if compute_2D_intervals and n_params > 1:
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            gridA, gridB = grids[fix_A], grids[fix_B]
            results[f"2d_test_p{fix_A+1}_{pair_name}"] = gridA
            results[f"2d_test_p{fix_B+1}_{pair_name}"] = gridB
            results[f"2d_t_data_{pair_name}"] = np.full((len(gridA), len(gridB)), np.nan)
            results[f"2d_t_critical_{pair_name}"] = {c: np.full((len(gridA), len(gridB)), np.nan) for c in cl}
            results[f"2d_accepted_{pair_name}"] = {c: np.full((len(gridA), len(gridB)), False) for c in cl}

    # --- Resume Protocol Validation ---
    # Attempts to ingest an existing state dictionary from a previous incomplete run.
    # Checks strictly if the requested bounds/grids map precisely to the saved geometries.
    if warm_start and os.path.exists(ckpt_path):
        try:
            ckpt = np.load(ckpt_path, allow_pickle=True)
            loaded_grids = [ckpt[f"grid_p{i+1}"] for i in range(n_params) if f"grid_p{i+1}" in ckpt]
            
            grids_match = False
            if len(loaded_grids) == n_params:
                grids_match = all(np.array_equal(g1, g2) for g1, g2 in zip(loaded_grids, grids))
                
            if grids_match:
                log_print(f"Grids verified identically. Resuming checkpoint from {ckpt_path}.")
                if "best_fit" in ckpt:
                    results["best_fit"] = ckpt["best_fit"]
                if "data_uncond_nll" in ckpt:
                    results["data_uncond_nll"] = float(ckpt["data_uncond_nll"])
                
                # Restore 1D tracking arrays
                if compute_1D_intervals:
                    for p_idx in range(n_params):
                        if f"1d_t_data_p{p_idx+1}" in ckpt:
                            results[f"1d_t_data_p{p_idx+1}"] = ckpt[f"1d_t_data_p{p_idx+1}"]
                        if f"1d_prof_params_p{p_idx+1}" in ckpt:
                            results[f"1d_prof_params_p{p_idx+1}"] = ckpt[f"1d_prof_params_p{p_idx+1}"]
                        for c in cl:
                            if f"1d_t_critical_p{p_idx+1}_{c}" in ckpt:
                                results[f"1d_t_critical_p{p_idx+1}"][c] = ckpt[f"1d_t_critical_p{p_idx+1}_{c}"]
                            if f"1d_accepted_p{p_idx+1}_{c}" in ckpt:
                                results[f"1d_accepted_p{p_idx+1}"][c] = ckpt[f"1d_accepted_p{p_idx+1}_{c}"]
                            
                # Restore 2D tracking arrays
                if compute_2D_intervals and n_params > 1:
                    pairs = list(itertools.combinations(range(n_params), 2))
                    for fix_A, fix_B in pairs:
                        pair_name = f"p{fix_A+1}p{fix_B+1}"
                        if f"2d_t_data_{pair_name}" in ckpt:
                            results[f"2d_t_data_{pair_name}"] = ckpt[f"2d_t_data_{pair_name}"]
                        for c in cl:
                            if f"2d_t_critical_{pair_name}_{c}" in ckpt:
                                results[f"2d_t_critical_{pair_name}"][c] = ckpt[f"2d_t_critical_{pair_name}_{c}"]
                            if f"2d_accepted_{pair_name}_{c}" in ckpt:
                                results[f"2d_accepted_{pair_name}"][c] = ckpt[f"2d_accepted_{pair_name}_{c}"]
            else:
                log_print("Parameter grids strictly conflict with checkpoint environment. Initiating fresh start.")
        except Exception as e:
            log_print(f"Failed to process checkpoint geometry: {e}. Initiating fresh start.")

    # --- PHASE 0: Fit global unconditional data ONCE ---
    # Finds the absolute minimum NLL over the full parameter space. 
    # This forms the denominator of the PLR test statistic for all points.
    if strategy == "grid":
        full_grid_points = np.array(list(itertools.product(*grids)), dtype=np.float64)
    else:
        full_grid_points = None
    
    if np.isnan(results["data_uncond_nll"]):
        if strategy == "grid":
            if likelihood_type == "binned":
                data_uncond_nll, best_params = unconditional_fit_grid(data, full_grid_points, S_sumw2, B_sumw2, use_finite_mc_correction_binned, compute_rates_func, extra_nll)
            else:
                data_uncond_nll, best_params = unconditional_fit_grid_unbinned(data, pdf_components, full_grid_points, compute_rates_func, extra_nll)
        elif strategy in ["ultranest", "hybrid"]:
            data_uncond_nll, best_params = unconditional_fit_ultranest(data, n_params, bounds_list, compute_rates_func, verbose=verbose, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, bounds_func=bounds_func, constraints=constraints, extra_nll=extra_nll)
        elif strategy == "scipy":
            data_uncond_nll, best_params = unconditional_fit_scipy(data, n_params, bounds_list, compute_rates_func, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method, n_restarts=n_restarts, extra_nll=extra_nll)
        
        results["best_fit"] = best_params
        results["data_uncond_nll"] = data_uncond_nll
        _save_fc_archive(results, grids, ckpt_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
    else:
        log_print("Bypassing global unconditional fit execution (Loaded configuration found).")
        data_uncond_nll = results["data_uncond_nll"]

    # --- PHASE 1: Compute 1D Intervals ---
    if compute_1D_intervals:
        log_print("Executing 1D Parameter Scans...")
        for p_idx in range(n_params):
            grid_test = grids[p_idx]
            free_grids = [grids[i] for i in range(n_params) if i != p_idx]
            
            # Sub-grid for profiling nuisance parameters exhaustively
            if strategy == "grid":
                if not free_grids:
                    cond_grid_points = np.zeros((1, 0), dtype=np.float64)
                else:
                    cond_grid_points = np.array(list(itertools.product(*free_grids)), dtype=np.float64)
            else:
                cond_grid_points = None
                
            t_data_arr = results[f"1d_t_data_p{p_idx+1}"]
            prof_params_arr = results[f"1d_prof_params_p{p_idx+1}"]
            t_crit_dict = results[f"1d_t_critical_p{p_idx+1}"]
            
            # Sub-phase 1a: Calculate t_data across the 1D grid
            for i, pt in enumerate(tqdm(grid_test, desc=f"1D Data (p{p_idx+1})", disable=disable_tqdm)):
                if not np.isnan(t_data_arr[i]): 
                    continue
                    
                if strategy == "grid":
                    if likelihood_type == "binned":
                        cond_nll, prof_p = conditional_fit_grid_1d(pt, p_idx, n_params, data, cond_grid_points, S_sumw2, B_sumw2, use_finite_mc_correction_binned, compute_rates_func, extra_nll)
                    else:
                        cond_nll, prof_p = conditional_fit_grid_unbinned_1d(pt, p_idx, n_params, data, pdf_components, cond_grid_points, compute_rates_func, extra_nll)
                elif strategy in ["ultranest", "hybrid"]:
                    cond_nll, prof_p = conditional_fit_1d_ultranest(pt, p_idx, n_params, data, bounds_list, compute_rates_func, verbose=verbose, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, bounds_func=bounds_func, constraints=constraints, extra_nll=extra_nll)
                elif strategy == "scipy":
                    # Neighbor warm-start: seed this grid point's DATA fit from the
                    # immediately preceding (already-evaluated) grid point's profiled
                    # nuisance parameters, instead of always starting fresh from the
                    # bounds midpoint. Paired with a forced-up n_restarts (>= 2) so a
                    # bad neighbor optimum can't silently cascade forward -- the fresh
                    # bounds-midpoint start is always tried alongside it.
                    neighbor_seed = None
                    effective_n_restarts = n_restarts
                    if neighbor_seeding and i > 0 and not np.any(np.isnan(prof_params_arr[i - 1])):
                        neighbor_seed = [prof_params_arr[i - 1][k] for k in range(n_params) if k != p_idx]
                        effective_n_restarts = max(n_restarts, 2)
                    cond_nll, prof_p = conditional_fit_1d_scipy(pt, p_idx, n_params, data, bounds_list, compute_rates_func, seed=neighbor_seed, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method, n_restarts=effective_n_restarts, extra_nll=extra_nll)
                
                prof_params_arr[i] = prof_p
                # Evaluate the actual PLR data statistic (bounded at 0 to fix numerical floating point noise)
                t_data_arr[i] = max(0.0, cond_nll - data_uncond_nll)

            # Sub-phase 1b: Calculate t_critical via Toy Generation
            for i, pt in enumerate(tqdm(grid_test, desc=f"1D Toys (p{p_idx+1})", disable=disable_tqdm)):
                if not np.isnan(t_crit_dict[cl[0]][i]): 
                    continue
                    
                true_params = prof_params_arr[i]
                
                if strategy == "grid":
                    if likelihood_type == "binned":
                        t_stats = generate_and_fit_toys_grid_1d(pt, p_idx, true_params, n_params, full_grid_points, cond_grid_points, n_toys, S_sumw2, B_sumw2, use_finite_mc_correction_binned, compute_rates_func, extra_nll)
                    else:
                        t_stats = generate_and_fit_toys_grid_unbinned_1d(pt, p_idx, true_params, n_params, pdf_components, full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool, compute_rates_func, generate_toy_func, extra_nll)
                else:
                    t_stats = generate_and_fit_toys_python(true_params, n_params, "1d", p_idx, None, None, pt, None, bounds_list, n_toys, strategy, num_cores=num_cores, verbose=0, pdf_components=pdf_components, likelihood_type=likelihood_type, S_mc_pool=S_mc_pool, B_mc_pool=B_mc_pool, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, compute_rates_func=compute_rates_func, generate_toy_func=generate_toy_func, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method, toy_batch_size=toy_batch_size, adaptive_toys=adaptive_toys, t_data=t_data_arr[i], alpha=adaptive_alpha, extra_nll=extra_nll)

                t_stats.sort()
                # Quantile index uses len(t_stats), not n_toys: adaptive_toys
                # may have stopped this point early with fewer toys than
                # n_toys (strategy="grid" always returns exactly n_toys, so
                # this is a no-op there).
                n_generated = len(t_stats)
                for c in cl:
                    t_crit_dict[c][i] = t_stats[min(int(c * n_generated), n_generated - 1)]
                    
            results[f"1d_accepted_p{p_idx+1}"] = {c: t_data_arr <= t_crit_dict[c] for c in cl}
            
            # Record loop checkpoint 
            _save_fc_archive(results, grids, ckpt_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
            log_print(f"Checkpoint successfully written mapping 1D parameter {p_idx+1}")

    # --- PHASE 2: Compute 2D Intervals ---
    if compute_2D_intervals and n_params > 1:
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            gridA, gridB = grids[fix_A], grids[fix_B]
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            log_print(f"Executing 2D Grid Scan mapping {pair_name} ({len(gridA)}x{len(gridB)})...")
            
            free_grids = [grids[i] for i in range(n_params) if i != fix_A and i != fix_B]
            if strategy == "grid":
                if not free_grids:
                    cond_grid_points = np.zeros((1, 0), dtype=np.float64)
                else:
                    cond_grid_points = np.array(list(itertools.product(*free_grids)), dtype=np.float64)
            else:
                cond_grid_points = None

            # In-memory cache of already-evaluated (i, j) -> full profiled parameter
            # vector, scoped to this pair's scan, used for neighbor warm-starting
            # below (mirrors the 1D loop's use of prof_params_arr, which the 2D scan
            # has no persistent equivalent of).
            prof_params_2d = {}

            def _lookup_2d_neighbor_seed(i, j):
                if not neighbor_seeding:
                    return None
                for ni, nj in ((i - 1, j), (i, j - 1)):
                    if (ni, nj) in prof_params_2d:
                        neighbor_full = prof_params_2d[(ni, nj)]
                        return [neighbor_full[k] for k in range(n_params) if k not in (fix_A, fix_B)]
                return None

            # Helper function to evaluate toys for a specific (i, j) 2D coordinate
            def eval_2d_point(i, j, pair_name=pair_name, gridA=gridA, gridB=gridB, fix_A=fix_A, fix_B=fix_B, cond_grid_points=cond_grid_points):
                # Keyed off 2d_t_data, not 2d_t_critical: the coarse-to-fine
                # interpolation step below fills in 2d_t_critical for every cell
                # (coarse and refined alike) with an interpolated estimate, not a
                # real toy-evaluated one. Checking 2d_t_critical here would make
                # every refinement cell look "already done" and turn the boundary
                # refinement pass into a silent no-op. 2d_t_data is only ever set
                # by Step 1 below (an exact conditional fit), so its NaN-ness
                # accurately reflects whether this point has actually been
                # evaluated, regardless of the interpolation pass.
                if not np.isnan(results[f"2d_t_data_{pair_name}"][i, j]):
                    return

                p_A, p_B = gridA[i], gridB[j]

                # Neighbor warm-start: seed this cell's DATA fit from an adjacent,
                # already-evaluated cell's profiled nuisance parameters. Paired with
                # a forced-up n_restarts (>= 2) so a bad neighbor optimum can't
                # silently cascade -- a fresh bounds-midpoint start is always tried
                # alongside it.
                neighbor_seed = _lookup_2d_neighbor_seed(i, j) if strategy == "scipy" else None
                effective_n_restarts = max(n_restarts, 2) if neighbor_seed is not None else n_restarts

                # Step 1. Exact Data NLL Calculation (Profiling out remaining nuisance pars).
                # 2d_t_data is always NaN at this point -- the guard at the top of this
                # function already returned otherwise -- so there is only one case to
                # handle here, not a NaN/already-computed branch.
                if strategy == "grid":
                    if likelihood_type == "binned":
                        cond_nll, prof_p = conditional_fit_grid_2d(p_A, p_B, fix_A, fix_B, n_params, data, cond_grid_points, S_sumw2, B_sumw2, use_finite_mc_correction_binned, compute_rates_func, extra_nll)
                    else:
                        cond_nll, prof_p = conditional_fit_grid_unbinned_2d(p_A, p_B, fix_A, fix_B, n_params, data, pdf_components, cond_grid_points, compute_rates_func, extra_nll)
                elif strategy in ["ultranest", "hybrid"]:
                    cond_nll, prof_p = conditional_fit_2d_ultranest(p_A, p_B, fix_A, fix_B, n_params, data, bounds_list, compute_rates_func, verbose=0, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, bounds_func=bounds_func, constraints=constraints, extra_nll=extra_nll)
                elif strategy == "scipy":
                    cond_nll, prof_p = conditional_fit_2d_scipy(p_A, p_B, fix_A, fix_B, n_params, data, bounds_list, compute_rates_func, seed=neighbor_seed, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method, n_restarts=effective_n_restarts, extra_nll=extra_nll)

                results[f"2d_t_data_{pair_name}"][i, j] = max(0.0, cond_nll - data_uncond_nll)
                true_params = prof_p

                if strategy == "scipy":
                    prof_params_2d[(i, j)] = true_params

                # Step 2. Sequential Toy Assessment to get critical threshold for coverage
                if strategy == "grid":
                    if likelihood_type == "binned":
                        t_stats = generate_and_fit_toys_grid_2d(p_A, p_B, fix_A, fix_B, true_params, n_params, full_grid_points, cond_grid_points, n_toys, S_sumw2, B_sumw2, use_finite_mc_correction_binned, compute_rates_func, extra_nll)
                    else:
                        t_stats = generate_and_fit_toys_grid_unbinned_2d(p_A, p_B, fix_A, fix_B, true_params, n_params, pdf_components, full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool, compute_rates_func, generate_toy_func, extra_nll)
                else:
                    t_stats = generate_and_fit_toys_python(true_params, n_params, "2d", None, fix_A, fix_B, p_A, p_B, bounds_list, n_toys, strategy, num_cores=num_cores, verbose=0, pdf_components=pdf_components, likelihood_type=likelihood_type, S_mc_pool=S_mc_pool, B_mc_pool=B_mc_pool, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc_correction_binned, compute_rates_func=compute_rates_func, generate_toy_func=generate_toy_func, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method, toy_batch_size=toy_batch_size, adaptive_toys=adaptive_toys, t_data=results[f"2d_t_data_{pair_name}"][i, j], alpha=adaptive_alpha, extra_nll=extra_nll)

                t_stats.sort()
                # Quantile index uses len(t_stats), not n_toys -- see the 1D
                # loop's identical comment above.
                n_generated = len(t_stats)
                for c in cl:
                    results[f"2d_t_critical_{pair_name}"][c][i, j] = t_stats[min(int(c * n_generated), n_generated - 1)]

            # --- Sparsification Array Tracing ---
            # To dramatically reduce computation time in 2D grids, this algorithm:
            # 1. Evaluates a coarse subset of grid nodes.
            # 2. Uses Bivariate Spline interpolation to estimate the t_critical surface.
            # 3. Dynamically traces and evaluates only the specific high-resolution nodes
            #    where the condition (t_data <= t_critical) transitions state (the contour boundary).
            #
            # Known limitation (sparsify_grid=True, hence why it defaults to False):
            # eval_2d_point's memoization is keyed off 2d_t_data, which is only ever
            # set by an actual conditional fit -- so the refinement pass below
            # genuinely re-evaluates boundary cells (this used to be a silent no-op
            # when the guard was keyed off 2d_t_critical instead, which the coarse-
            # to-fine interpolation below fills in for every cell regardless of
            # whether it was really evaluated). However, boundary detection ("is
            # this coarse cell adjacent to a rejected cell") and the resulting
            # refinement halo are still only a single, non-iterative, 1-cell-wide
            # ring around the ~5-per-axis coarse nodes. For grids much larger than
            # roughly 20x20 per axis, that halo never reaches deep interior/exterior
            # cells far from any coarse node: their 2d_t_data stays NaN and they are
            # force-excluded from the accepted region regardless of their true
            # status (NaN comparisons are always False). This under-counts the true
            # accepted region at exactly the grid sizes this feature is meant for
            # (see docs/dev/FIXES_BRIEF_dev-no-templates.md for a worked example).
            # Making this reliable at scale needs an iterative refine loop and/or an
            # interpolated (not NaN) t_data estimate to trace the true contour --
            # tracked as follow-up work, not addressed here.
            if sparsify_grid:
                step_A = max(1, len(gridA) // 5)
                step_B = max(1, len(gridB) // 5)
                coarse_i = sorted(list(set(list(range(0, len(gridA), step_A)) + [len(gridA)-1])))
                coarse_j = sorted(list(set(list(range(0, len(gridB), step_B)) + [len(gridB)-1])))
                coarse_pts = [(i, j) for i in coarse_i for j in coarse_j]
                
                log_print(f"2D Sparsification: Coarse array pass across {len(coarse_pts)} structural points...")
                for pt in tqdm(coarse_pts, desc=f"2D Coarse {pair_name}", disable=disable_tqdm): 
                    eval_2d_point(*pt)
                    
                # Interpolate the coarse critical surface framework
                if SCIPY_AVAILABLE:
                    from scipy.interpolate import RectBivariateSpline
                    for c in cl:
                        z = results[f"2d_t_critical_{pair_name}"][c][np.ix_(coarse_i, coarse_j)]
                        interp = RectBivariateSpline(gridA[coarse_i], gridB[coarse_j], z)
                        results[f"2d_t_critical_{pair_name}"][c] = interp(gridA, gridB)
                else:
                    # Precompute nearest-neighbor mappings to avoid repeated lambda evaluations
                    nearest_i = [min(coarse_i, key=lambda x: abs(x - i)) for i in range(len(gridA))]
                    nearest_j = [min(coarse_j, key=lambda x: abs(x - j)) for j in range(len(gridB))]
                    for c in cl:
                        for i in range(len(gridA)):
                            for j in range(len(gridB)):
                                results[f"2d_t_critical_{pair_name}"][c][i, j] = results[f"2d_t_critical_{pair_name}"][c][nearest_i[i], nearest_j[j]]
                
                # Dynamic Perimeter Edge Tracing
                # Scans the interpolated map and flags cells that lie on the inclusion/exclusion border.
                refine_pts = set()
                for c in cl:
                    approx_acc = results[f"2d_t_data_{pair_name}"] <= results[f"2d_t_critical_{pair_name}"][c]
                    for i in range(len(gridA)):
                        for j in range(len(gridB)):
                            if approx_acc[i, j]:
                                is_bound = False
                                for di in [-1, 0, 1]:
                                    for dj in [-1, 0, 1]:
                                        ni = i + di
                                        nj = j + dj
                                        if 0 <= ni < len(gridA) and 0 <= nj < len(gridB):
                                            if not approx_acc[ni, nj]: 
                                                is_bound = True
                                                
                                if is_bound:
                                    for di in [-1, 0, 1]:
                                        for dj in [-1, 0, 1]:
                                            ni = i + di
                                            nj = j + dj
                                            if 0 <= ni < len(gridA) and 0 <= nj < len(gridB): 
                                                refine_pts.add((ni, nj))
                                                
                pts_to_eval = [p for p in refine_pts if p not in coarse_pts]
                log_print(f"2D Sparsification: Boundary resolved. Integrating {len(pts_to_eval)} exact refinement matrices...")
            else:
                pts_to_eval = [(i,j) for i in range(len(gridA)) for j in range(len(gridB))]
                
            for pt in tqdm(pts_to_eval, desc=f"2D Edge {pair_name}", disable=disable_tqdm): 
                eval_2d_point(*pt)
            
            # Formally calculate final boolean acceptance mask
            for c in cl: 
                results[f"2d_accepted_{pair_name}"][c] = results[f"2d_t_data_{pair_name}"] <= results[f"2d_t_critical_{pair_name}"][c]
                
            # Record loop checkpoint 
            _save_fc_archive(results, grids, ckpt_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
            log_print(f"Checkpoint successfully written mapping 2D sector {pair_name}")

    # --- Structural Archiving & Cleanup Operations ---
    if output_file is not None:
        final_path = os.path.join(save_directory, f"{output_file}.npz")
        final_path_json = os.path.join(save_directory, f"{output_file}.json")
        _save_fc_archive(results, grids, final_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
        _save_fc_json(results, final_path_json, cl, compute_1D_intervals, compute_2D_intervals, n_params)
        log_print(f"Results archived completely to storage matrix {final_path} and JSON {final_path_json}")
        
    # Eliminate interim tracking files now that final save is secure
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    fig = None
    if compute_1D_intervals or (compute_2D_intervals and n_params > 1):
        fig = generate_corner_plot(results, {
            "n_params": n_params,
            "compute_1D_intervals": compute_1D_intervals,
            "compute_2D_intervals": compute_2D_intervals,
            "param_names": param_names if param_names else [f"param{i+1}" for i in range(n_params)],
            "cl": cl,
            "save_directory": save_directory,
            "smooth_1d": smooth_1d,
            "smooth_2d": smooth_2d
        })

    return results, fig

if __name__ == "__main__":
    from numba import njit
    
    config = parse_arguments()
    
    grids = [
        np.linspace(0.5, 2.0, 50), # param1
        np.linspace(0.5, 2.0, 50), # param2
        np.linspace(0.5, 1.5, 50)  # param3
    ]
    
    if config["likelihood_type"] == "binned":
        print(f"\n--- Running BINNED Analysis Example ({len(grids)}-Parameter) | Modes -> 1D: {config['compute_1D_intervals']} | 2D: {config['compute_2D_intervals']} ---")
        
        # Fixed template arrays are referenced via module-level closure rather
        # than passed in as compute_rates_func arguments (S_model/B_model no
        # longer exist as a mechanism -- see the "BREAKING CHANGES" entry in
        # CHANGELOG.md). Must be defined before the @njit function below so
        # numba's global lookup resolves them.
        S_template = np.array([0.1, 0.5, 2.0, 5.0])
        B_template = np.array([15.0, 5.0, 1.0, 0.1])

        @njit(fastmath=True, nogil=True)
        def example_compute_rates_binned(params, S_sumw2, B_sumw2):
            """
            Computes the expected binned rates and variances for a non-degenerate
            3-parameter mock model.

            Parameters:
            -----------
            params : array_like
                params[0] = Signal strength multiplier
                params[1] = Flat background offset (breaks degeneracy)
                params[2] = Template background strength multiplier
            S_sumw2, B_sumw2 : array_like
                Sum of squared MC weights (sumw2) per bin.

            Returns:
            --------
            mu : array_like
                Total expected counts per bin.
            sigma2 : array_like
                Total variance per bin.
            """
            # Using independent linear contributions to break the params[0]*params[1] degeneracy
            mu = params[0] * S_template + params[1] + params[2] * B_template

            # The flat background is treated as exact (0 variance) for this example
            sigma2 = (params[0]**2) * S_sumw2 + (params[2]**2) * B_sumw2
            return mu, sigma2

        S_sumw2 = S_template.copy()
        B_sumw2 = B_template.copy()

        np.random.seed(42)
        # Updated to inject 1.0 for all three parameters
        N_data_binned = np.random.poisson(1.0 * S_template + 1.0 + 1.0 * B_template)
        print(f"Mock Observed Data (Binned Counts): {N_data_binned}")

        fc_results, fc_fig = compute_fc_intervals(
            data=N_data_binned, grids=grids,
            compute_rates_func=example_compute_rates_binned,
            cl=config["cl"], n_toys=config["n_toys"], strategy=config["strategy"],
            num_cores=config["num_cores"], verbose=config["verbose"],
            adaptive_toys=config["adaptive_toys"], toy_batch_size=config["toy_batch_size"],
            sparsify_grid=config["sparsify_grid"], warm_start=config["warm_start"],
            likelihood_type="binned",
            output_file=config["output_file"], save_log=config["save_log"], save_directory=config["save_directory"],
            use_finite_mc_correction_binned=config["use_finite_mc_correction_binned"],
            S_sumw2=S_sumw2, B_sumw2=B_sumw2,
            compute_1D_intervals=config["compute_1D_intervals"],
            compute_2D_intervals=config["compute_2D_intervals"],
            param_names=config["param_names"],
            smooth_1d=config["smooth_1d"],
            smooth_2d=config["smooth_2d"],
            scipy_method=config.get("scipy_method"),
            n_restarts=config.get("n_restarts", 1),
            neighbor_seeding=config.get("neighbor_seeding", True)
        )

    elif config["likelihood_type"] == "unbinned":
        print(f"\n--- Running UNBINNED Analysis Example ({len(grids)}-Parameter) | Modes -> 1D: {config['compute_1D_intervals']} | 2D: {config['compute_2D_intervals']} ---")
        from scipy.stats import expon, norm
        
        def s_pdf_mock(x): return norm.pdf(x, loc=5.0, scale=1.0)
        def b_pdf_mock(x): return expon.pdf(x, scale=2.0)
        
        def example_compute_rates_unbinned(params, probs):
            """
            Computes total expected events and pointwise likelihood probabilities.

            Parameters:
            -----------
            params : array_like
                params[0] = Signal strength multiplier
                params[1] = Flat background strength (breaks degeneracy)
                params[2] = Template background strength
            probs : list of array_like
                Pre-evaluated PDF probabilities for each event, one entry per
                `pdf_components[i]` (here: [s_pdf_mock, b_pdf_mock]).

            Returns:
            --------
            expected_total : float
                Total expected events across all sources.
            p_events : array_like
                Un-normalized likelihood densities evaluated for each event.
            """
            s_probs, b_probs = probs[0], probs[1]
            expected_total = params[0] + params[1] + params[2]

            if len(s_probs) == 0 and len(b_probs) == 0:
                return expected_total, np.array([])

            # Assume the flat background is distributed uniformly between 0 and 10 (PDF = 0.1)
            p_events = params[0] * s_probs + params[1] * 0.1 + params[2] * b_probs
            return expected_total, p_events

        def example_generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool):
            """
            Generates unbinned toy datasets based on the 3-parameter model.
            
            Parameters:
            -----------
            true_params : array_like
                The parameter point (hypothesis) to generate toys from.
            S_mc_pool, B_mc_pool : array_like
                Pre-generated MC events to draw from for the templates.
                
            Returns:
            --------
            array_like
                1D array of generated event features.
            """
            n_sig = np.random.poisson(true_params[0])
            n_flat = np.random.poisson(true_params[1])
            n_bkg = np.random.poisson(true_params[2])
            
            parts = []
            if n_sig > 0 and S_mc_pool is not None and len(S_mc_pool) > 0:
                parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
            if n_flat > 0:
                # Flat background distributed uniformly between 0 and 10
                parts.append(np.random.uniform(0, 10, size=n_flat))
            if n_bkg > 0 and B_mc_pool is not None and len(B_mc_pool) > 0:
                parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
                
            if parts:
                return np.concatenate(parts)
            return np.array([])
            
        s_mc_pool = np.random.normal(loc=5.0, scale=1.0, size=5000)
        b_mc_pool = np.random.exponential(scale=2.0, size=5000)
        
        # Updated to include events from the new flat background parameter
        unbinned_data = np.concatenate([
            np.random.choice(s_mc_pool, size=2), 
            np.random.uniform(0, 10, size=2),
            np.random.choice(b_mc_pool, size=3)  
        ])
        print(f"Mock Observed Unbinned Events: {np.round(unbinned_data, 2)}")
        
        fc_results, fc_fig = compute_fc_intervals(
            data=unbinned_data, grids=grids,
            compute_rates_func=example_compute_rates_unbinned,
            generate_toy_func=example_generate_unbinned_toy,
            pdf_components=[s_pdf_mock, b_pdf_mock],
            cl=config["cl"], n_toys=config["n_toys"], strategy=config["strategy"],
            num_cores=config["num_cores"], verbose=config["verbose"],
            adaptive_toys=config["adaptive_toys"], toy_batch_size=config["toy_batch_size"],
            sparsify_grid=config["sparsify_grid"], warm_start=config["warm_start"],
            likelihood_type="unbinned", S_mc_pool=s_mc_pool, B_mc_pool=b_mc_pool,
            output_file=config["output_file"], save_log=config["save_log"], save_directory=config["save_directory"],
            compute_1D_intervals=config["compute_1D_intervals"],
            compute_2D_intervals=config["compute_2D_intervals"],
            param_names=config["param_names"],
            smooth_1d=config["smooth_1d"],
            smooth_2d=config["smooth_2d"],
            scipy_method=config.get("scipy_method"),
            n_restarts=config.get("n_restarts", 1),
            neighbor_seeding=config.get("neighbor_seeding", True)
        )