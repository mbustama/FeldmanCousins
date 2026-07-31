"""
Monte Carlo Toy Generation and Fitting Module

This module manages the core frequentist computation of the PyFC package: 
the generation and parallel evaluation of pseudo-experiments (toys). 
To establish exact coverage in the Feldman-Cousins framework, the distribution 
of the test statistic must be empirically derived at each point in the parameter 
grid by simulating data under the null hypothesis, and fitting it both unconditionally 
and conditionally.

This file specifically handles the execution when using continuous optimizers 
(SciPy, UltraNest) and manages efficient parallelization by routing workloads 
to either Thread or Process pools depending on Python's Global Interpreter Lock (GIL) 
limitations for the requested likelihood type.

Created: v0.1.0 (July 24, 2026)
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import concurrent.futures
import math
import warnings

import numpy as np

from .optimizers import (
    conditional_fit_1d_scipy,
    conditional_fit_1d_ultranest,
    conditional_fit_2d_scipy,
    conditional_fit_2d_ultranest,
    unconditional_fit_scipy,
    unconditional_fit_ultranest,
)

# --- Adaptive early-stopping for toy generation ---
# Never consider stopping before this many toys have been generated at a
# given grid point -- small-sample binomial confidence intervals are
# unreliable, and the whole point of "definitively excluded" (README's own
# framing) is to only stop where there's overwhelming evidence either way.
ADAPTIVE_MIN_TOYS = 100

# Two-sided 99.9% normal quantile (not 95%) for the Wilson score interval
# below. The cost of a wrong early stop is a biased confidence interval, so
# the stopping rule itself is biased hard towards caution: it only fires
# when the accept/reject verdict is essentially certain not to flip with
# more toys, not merely "likely."
_ADAPTIVE_CI_Z = 3.290526731491989


def _wilson_score_interval(n_above, n_total, z=_ADAPTIVE_CI_Z):
    """
    Wilson score confidence interval for a binomial proportion.

    Parameters:
    -----------
    n_above : int
        Number of "successes" -- here, toys with t_toy >= t_data.
    n_total : int
        Total number of toys generated so far.
    z : float, optional
        Two-sided normal quantile for the desired confidence level.

    Returns:
    --------
    (lo, hi) : tuple of float
        The confidence interval on the true proportion, clipped to [0, 1].
    """
    p_hat = n_above / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    center = (p_hat + z2 / (2.0 * n_total)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n_total)) / n_total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _adaptive_stop_decision(n_above, n_total, alpha, min_toys=ADAPTIVE_MIN_TOYS):
    """
    Decides whether a grid point's accept/reject verdict at significance
    `alpha` is settled enough to stop generating more toys for it.

    Statistical basis: `n_above / n_total` is the running estimate of the
    p-value P(t_toy >= t_data). The point is accepted iff this p-value is
    >= alpha (equivalently, t_data is at or below the (1-alpha) quantile of
    the toy distribution -- see `compute_fc_intervals`'s own t_critical
    computation). If a strict (99.9%) Wilson confidence interval on that
    p-value estimate lies entirely on one side of `alpha`, the verdict
    cannot plausibly flip with more toys.

    Parameters:
    -----------
    n_above : int
        Number of toys so far with t_toy >= t_data.
    n_total : int
        Total number of toys generated so far.
    alpha : float
        Target significance (1 - CL) for the accept/reject decision. When
        multiple confidence levels are requested, callers should pass the
        *smallest* alpha (largest CL) -- the hardest case to resolve --
        so that stopping here also guarantees every less-stringent CL's
        decision is settled.
    min_toys : int, optional
        Minimum toy count before early-stopping is even considered.

    Returns:
    --------
    bool
        True if it is safe to stop generating more toys for this point.
    """
    if n_total < min_toys:
        return False
    lo, hi = _wilson_score_interval(n_above, n_total)
    return lo > alpha or hi < alpha


def _worker_unbinned_toy(args):
    """
    Module-level explicit worker function for unbinned toy generation and fitting.
    
    Architecture Note:
    This function must reside at the top level of the module to bypass Python's 
    `pickle` limitations. `concurrent.futures.ProcessPoolExecutor` must serialize 
    functions to pass them to worker processes. Closures or nested functions 
    (like `fit_single_toy` below) cannot be pickled natively.

    Parameters:
    -----------
    args : tuple
        A packed tuple containing all necessary arguments:
        (toy_index, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B,
         t_vA, t_vB, pdf_components, bounds_list, S_mc_pool, B_mc_pool, strategy,
         compute_rates_func, generate_toy_func, bounds_func, constraints, scipy_method)

    Returns:
    --------
    float
        The computed Profile Likelihood Ratio test statistic for this single toy.
    """
    # Unpack the monolithic argument tuple required for multiprocessing
    (t, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB,
     pdf_components, bounds_list, S_mc_pool, B_mc_pool, strategy,
     compute_rates_func, generate_toy_func, bounds_func, constraints, scipy_method) = args

    # Generate the simulated unbinned dataset by bootstrapping from the MC pools
    toy_data = generate_toy_func(true_params, S_mc_pool, B_mc_pool)

    if strategy == "scipy" or strategy == "hybrid":
        # Seeding strategy: Initialize the optimizer at the true parameters that generated
        # the toy. The global minimum is statistically guaranteed to be in this neighborhood.
        seed_p = true_params.copy()

        # 1. Unconditional fit (Denominator of the PLR)
        uncond_nll, _ = unconditional_fit_scipy(toy_data, n_params, bounds_list, compute_rates_func, seed=seed_p, pdf_components=pdf_components, likelihood_type="unbinned", bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method)

        # 2. Conditional fit (Numerator of the PLR)
        if fit_mode == "1d":
            seed_free_1d = [true_params[i] for i in range(n_params) if i != fix_idx] if len(true_params) > 1 else None
            cond_nll, _ = conditional_fit_1d_scipy(t_vA, fix_idx, n_params, toy_data, bounds_list, compute_rates_func, seed=seed_free_1d, pdf_components=pdf_components, likelihood_type="unbinned", bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method)
        elif fit_mode == "2d":
            seed_free_2d = [true_params[i] for i in range(n_params) if i not in (fix_A, fix_B)] if len(true_params) > 2 else None
            cond_nll, _ = conditional_fit_2d_scipy(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, bounds_list, compute_rates_func, seed=seed_free_2d, pdf_components=pdf_components, likelihood_type="unbinned", bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method)

    elif strategy == "ultranest":
        uncond_nll, _ = unconditional_fit_ultranest(toy_data, n_params, bounds_list, compute_rates_func, verbose=0, pdf_components=pdf_components, likelihood_type="unbinned", bounds_func=bounds_func, constraints=constraints)
        if fit_mode == "1d":
            cond_nll, _ = conditional_fit_1d_ultranest(t_vA, fix_idx, n_params, toy_data, bounds_list, compute_rates_func, verbose=0, pdf_components=pdf_components, likelihood_type="unbinned", bounds_func=bounds_func, constraints=constraints)
        elif fit_mode == "2d":
            cond_nll, _ = conditional_fit_2d_ultranest(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, bounds_list, compute_rates_func, verbose=0, pdf_components=pdf_components, likelihood_type="unbinned", bounds_func=bounds_func, constraints=constraints)

    return max(0.0, cond_nll - uncond_nll)


def generate_and_fit_toys_python(true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB,
                                 bounds_list, n_toys, strategy, num_cores=None, verbose=1,
                                 pdf_components=None,
                                 likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                                 S_sumw2=None, B_sumw2=None, use_finite_mc=False,
                                 compute_rates_func=None, generate_toy_func=None, bounds_func=None,
                                 constraints=None, scipy_method=None,
                                 toy_batch_size=None, adaptive_toys=False, t_data=None, alpha=None):
    """
    Handles threaded generation and fitting of MC toys for 1D profiling and 2D contours.
    
    Statistical Context:
    To evaluate if a grid point $t_{test}$ belongs in the confidence interval, we generate 
    `n_toys` pseudo-experiments assuming the null hypothesis ($\theta = t_{test}$, with 
    nuisance parameters at their conditional maximum likelihood values `true_params`). 
    Each toy is then fit unconditionally and conditionally to build the empirical distribution 
    of the Profile Likelihood Ratio.

    Parallelization Context:
    - Unbinned PDFs often rely on pure Python functions (like `scipy.stats.norm.pdf`). 
      These are strictly bound by the Global Interpreter Lock (GIL). Multithreading provides 
      zero speedup. Thus, we branch to a `ProcessPoolExecutor` to spawn distinct processes.
    - Binned operations (vectorized NumPy math) naturally release the GIL. Thus, we can 
      use a lighter-weight `ThreadPoolExecutor` and avoid the heavy overhead of process spawning 
      and memory IPC.

    Parameters:
    -----------
    true_params : array_like
        The physical parameters (test values + profiled nuisance values) used to generate toys.
    n_params : int
        Total number of parameters in the model.
    fit_mode : str
        Either "1d" or "2d".
    fix_idx, fix_A, fix_B : int or None
        Indices of the fixed parameters depending on the fit mode.
    t_vA, t_vB : float
        The test values to fix the parameters at during the conditional fit.
    bounds_list : various
        Boundary constraints.
    n_toys : int
        Number of pseudo-experiments to generate and evaluate.
    strategy : str
        Optimizer selection ("scipy", "ultranest", "hybrid").
    num_cores : int
        Number of parallel workers.
    verbose : int
        Logging level.
    pdf_components : list of callable, optional
        Probability density functions, one per model component. Required
        when `likelihood_type="unbinned"`; unused for `"binned"`.
    likelihood_type : str
        "binned" or "unbinned".
    S_mc_pool, B_mc_pool : array_like, optional
        Source pools for bootstrapping unbinned events.
    S_sumw2, B_sumw2 : array_like, optional
        Sum of squared MC weights (sumw2) per bin, for finite MC corrections.
    use_finite_mc : bool
        Toggle for Poisson-Gamma mixture likelihoods.
    compute_rates_func : callable
        User-provided expected rates mapping function.
    generate_toy_func : callable
        User-provided unbinned toy bootstrapper.
    bounds_func : callable, optional
        Advanced hook for joint/simplex parameter constraints, forwarded
        unchanged to the underlying `*_scipy`/`*_ultranest` fit calls (see
        `optimizers.py` module docstring). If None (default), behavior is
        unchanged from before this parameter existed.
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints, forwarded unchanged to the underlying
        `*_scipy`/`*_ultranest` fit calls. If None/empty (default), behavior
        is unchanged.
    scipy_method : str, optional
        Explicit `scipy.optimize.minimize` method override, forwarded to the
        scipy fit calls.
    toy_batch_size : int, optional
        Chunk size for dispatching toys to the executor: instead of
        submitting all `n_toys` tasks in one `executor.map` call, they are
        submitted/collected in batches of this size, bounding how many toy
        datasets / in-flight result objects are alive at once (matters
        most for `likelihood_type="unbinned"`, where each toy is a
        variable-size event array passed through `ProcessPoolExecutor`
        IPC; binned toys are small fixed-size per-bin arrays and don't
        have this memory concern). None or <= 0 means "one batch of the
        full `n_toys`" (previous behavior, unchanged numerics either way
        -- this only affects how toys are grouped for dispatch/early-stop
        checks, never how many are generated when `adaptive_toys=False`).
    adaptive_toys : bool, optional
        If True, stop generating toys for this point early once the
        accept/reject verdict at `alpha` is statistically settled (see
        `_adaptive_stop_decision`). Checked once per `toy_batch_size`
        batch, so a smaller `toy_batch_size` allows finer-grained (but not
        free -- still bounded by `ADAPTIVE_MIN_TOYS`) early stopping.
        Requires `t_data` and `alpha` to be provided; silently has no
        effect otherwise (falls back to always generating all `n_toys`).
    t_data : float, optional
        The observed data's test statistic at this grid point, required
        for `adaptive_toys` to have any effect.
    alpha : float, optional
        Target significance (1 - CL) for the `adaptive_toys` stopping
        decision. When multiple confidence levels are requested, callers
        should pass the smallest alpha (largest CL) -- see
        `_adaptive_stop_decision`'s docstring.

    Returns:
    --------
    t_stats : np.ndarray
        Array of the computed test statistics -- length `n_toys`, unless
        `adaptive_toys` stopped early, in which case it is shorter.
        Callers must index/quantile using `len(t_stats)`, not the original
        `n_toys`.
    """
    # max(n_toys, 1) guards against batch_size landing on 0 when n_toys=0
    # and toy_batch_size is unset -- range(0, 0, 0) raises ValueError, while
    # range(0, 0, 1) correctly yields zero iterations (n_toys=0 -> no toys
    # generated, an empty t_stats array, not a crash).
    batch_size = toy_batch_size if (toy_batch_size and toy_batch_size > 0) else max(n_toys, 1)
    do_adaptive = adaptive_toys and t_data is not None and alpha is not None

    # --- Branch 1: Unbinned Data (Process-based parallelism) ---
    if likelihood_type == "unbinned":
        # Deliberately NOT a `with` block. On the failure path below, `with` would call
        # shutdown(wait=True) on the way out and block until every worker exits -- but the
        # workers are exactly what may be wedged, so the recovery would hang instead of
        # recovering. Observed on Python 3.9: a broken pool left forked children that never
        # exited, and the run sat for two hours (until cancelled) rather than falling
        # through to the ThreadPoolExecutor retry a few lines down.
        #
        # So the lifetime is managed explicitly and the two exits differ: the success path
        # waits, because those results are the return value; the failure path does not,
        # because there is nothing left to collect and something to escape from.
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cores)
        try:
            t_stats = []
            n_above = 0
            for batch_start in range(0, n_toys, batch_size):
                batch_end = min(batch_start + batch_size, n_toys)
                args_batch = [(t, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB,
                              pdf_components, bounds_list, S_mc_pool, B_mc_pool, strategy,
                              compute_rates_func, generate_toy_func, bounds_func, constraints, scipy_method)
                              for t in range(batch_start, batch_end)]
                batch_results = list(executor.map(_worker_unbinned_toy, args_batch))
                t_stats.extend(batch_results)
                if do_adaptive:
                    n_above += sum(1 for v in batch_results if v >= t_data)
                    if _adaptive_stop_decision(n_above, len(t_stats), alpha):
                        break
            executor.shutdown(wait=True)
            return np.array(t_stats)
        except Exception as e:
            # This is deliberately broad -- pool-infrastructure failures
            # (a worker process crashing, an unpicklable compute_rates_func/
            # generate_toy_func/pdf_components closure) surface here as a
            # mix of concurrent.futures.process.BrokenProcessPool, TypeError,
            # and platform-specific OSError, with no single reliable type to
            # narrow to. But this also catches a genuine bug inside the
            # user's own functions (raised inside a worker and propagated
            # here by executor.map) just as readily -- the warning below
            # does not claim a specific cause for that reason. If the same
            # error recurs on the ThreadPoolExecutor fallback below (which
            # does NOT catch exceptions), it will surface uncaught there,
            # which is the actual signal that this was a user-code bug
            # rather than a pool problem.
            warnings.warn(f"Toy generation via ProcessPoolExecutor did not complete ({type(e).__name__}: {e}). Retrying via ThreadPoolExecutor -- if this same error recurs there, it is most likely a bug in your compute_rates_func/generate_toy_func/pdf_components, not a ProcessPoolExecutor/pickling issue.")
        finally:
            # Backstop, and the piece that replaces what `with` used to guarantee: release
            # the pool on every exit, including the BaseExceptions the `except` above does
            # not catch (KeyboardInterrupt being the one that matters -- Ctrl-C during a
            # long toy run). Non-blocking for the same reason as the failure path, and a
            # no-op when the success path has already shut down cleanly.
            executor.shutdown(wait=False)

    # --- Branch 2: Binned Data (Thread-based parallelism) or Unbinned Fallback ---
    if likelihood_type == "binned":
        mu_true, _ = compute_rates_func(true_params, S_sumw2, B_sumw2)
        toys_binned_data = np.random.poisson(mu_true, size=(n_toys,) + mu_true.shape)

    def fit_single_toy(t):
        if likelihood_type == "binned":
            toy_data = toys_binned_data[t]
        else:
            toy_data = generate_toy_func(true_params, S_mc_pool, B_mc_pool)

        if strategy == "scipy" or strategy == "hybrid":
            seed_p = true_params.copy()
            uncond_nll, _ = unconditional_fit_scipy(toy_data, n_params, bounds_list, compute_rates_func, seed=seed_p, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method)

            if fit_mode == "1d":
                seed_free_1d = [true_params[i] for i in range(n_params) if i != fix_idx] if len(true_params) > 1 else None
                cond_nll, _ = conditional_fit_1d_scipy(t_vA, fix_idx, n_params, toy_data, bounds_list, compute_rates_func, seed=seed_free_1d, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method)
            elif fit_mode == "2d":
                seed_free_2d = [true_params[i] for i in range(n_params) if i not in (fix_A, fix_B)] if len(true_params) > 2 else None
                cond_nll, _ = conditional_fit_2d_scipy(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, bounds_list, compute_rates_func, seed=seed_free_2d, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc, bounds_func=bounds_func, constraints=constraints, scipy_method=scipy_method)

        elif strategy == "ultranest":
            uncond_nll, _ = unconditional_fit_ultranest(toy_data, n_params, bounds_list, compute_rates_func, verbose=0, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc, bounds_func=bounds_func, constraints=constraints)

            if fit_mode == "1d":
                cond_nll, _ = conditional_fit_1d_ultranest(t_vA, fix_idx, n_params, toy_data, bounds_list, compute_rates_func, verbose=0, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc, bounds_func=bounds_func, constraints=constraints)
            elif fit_mode == "2d":
                cond_nll, _ = conditional_fit_2d_ultranest(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, bounds_list, compute_rates_func, verbose=0, pdf_components=pdf_components, likelihood_type=likelihood_type, S_sumw2=S_sumw2, B_sumw2=B_sumw2, use_finite_mc=use_finite_mc, bounds_func=bounds_func, constraints=constraints)

        return max(0.0, cond_nll - uncond_nll)

    t_stats = []
    n_above = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        for batch_start in range(0, n_toys, batch_size):
            batch_end = min(batch_start + batch_size, n_toys)
            batch_results = list(executor.map(fit_single_toy, range(batch_start, batch_end)))
            t_stats.extend(batch_results)
            if do_adaptive:
                n_above += sum(1 for v in batch_results if v >= t_data)
                if _adaptive_stop_decision(n_above, len(t_stats), alpha):
                    break

    return np.array(t_stats)