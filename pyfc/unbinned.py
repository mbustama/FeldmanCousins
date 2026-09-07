"""
Unbinned Likelihood and Grid Optimization Module

This file implements the Extended Unbinned Maximum Likelihood (EUML) routines 
for the PyFC package. Unlike binned analyses which rely on histograms, the unbinned 
method evaluates the probability density directly on an event-by-event basis. 
This is statistically powerful for small datasets where binning would cause 
unacceptable information loss.

The module also includes brute-force grid search optimizers for the unbinned 
likelihood, which serve as robust alternatives to gradient-based minimizers 
when the likelihood surface is highly irregular.

Created: v0.1.0 (July 24, 2026)
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np


# --- 1. Core Math (Unbinned) ---
def calc_nll_unbinned(params, len_obs, probs, compute_rates_func):
    """
    Computes the Extended Unbinned Negative Log-Likelihood (NLL).

    Mathematical Expression:
    -ln(L) = N_expected - sum(ln(lambda(x_i)))

    Note: The factorial term ln(N_obs!) is dropped from the standard EUML
    formula because it is a constant that depends only on the data,
    and therefore strictly cancels out when computing delta-NLL test statistics.

    Parameters:
    -----------
    params : array_like
        The physical parameters.
    len_obs : int
        The total number of observed events (N_obs).
    probs : list of np.ndarray
        The pre-evaluated PDFs (one per `pdf_components[i]`) for the observed events.
    compute_rates_func : callable
        User-provided function `(params, probs) -> (expected_total, p_events)` mapping
        parameters and the pre-evaluated component densities to the expected total
        event count and the per-event mixture density.

    Returns:
    --------
    float
        The calculated NLL value. For events where the model predicts an
        unphysical or near-zero density (p_events[k] <= 1e-12), that event's
        contribution is a smooth, continuous extension of -log(p) rather
        than a flat penalty
        (see note below).

    Note on unphysical (p_events <= p_floor = 1e-12) events:
    -----------------------------------------------------------
    Earlier versions of this function returned a flat penalty (1e10) the
    instant ANY single event had a non-positive predicted density, short-
    circuiting the entire likelihood. Because `scipy.optimize.minimize(...,
    method='L-BFGS-B')` estimates gradients via finite differences, a locally
    -constant NLL surface reads as "gradient ~ 0, already converged," which can
    permanently trap the optimizer. Instead, each offending event's density is
    clipped to a tiny positive floor for the -log(p) term (a continuous
    extension of the real likelihood), and a quadratic barrier proportional to
    how far below that floor p_events[k] actually is is added per-event, so the
    gradient stays informative instead of flat. This is applied elementwise
    (not collapsed via `np.any`), so multiple unphysical events each contribute
    independently. The trigger is `p_events <= p_floor` rather than
    `p_events <= 0` specifically so there is no razor-thin discontinuity in the
    tiny sliver `(0, p_floor)`: the real -log(p) term diverges as
    p_events -> 0+, while the floor-based term is constant (evaluated at
    p_floor, not at the actual p_events[k]), so triggering only at
    p_events <= 0 would make the NLL drop right at that boundary instead of
    continuing to climb. For all p_events[k] > p_floor (i.e. all physically
    meaningful values -- p_floor is astronomically small), behavior is
    practically identical to before this change (matches to within 1e-12
    relative/absolute tolerance; see tests/test_smoothing.py).
    """
    expected_total, p_events = compute_rates_func(params, probs)

    # Handle the zero-observation case perfectly
    if len_obs == 0:
        return expected_total

    # Triggered for p_events <= p_floor, not just p_events <= 0: the physical
    # branch's own -log(p) term below diverges to +inf as p_events -> 0+,
    # while this branch's floor term is constant (evaluated at the fixed
    # p_floor, not at the actual p_events). If the trigger were
    # p_events <= 0, the tiny physical sliver (0, p_floor) would still take
    # the diverging physical branch, so crossing from p_events = +epsilon to
    # p_events = -epsilon would make the NLL drop rather than keep climbing.
    # Extending the trigger to include that sliver removes the discontinuity:
    # at p_events == p_floor exactly, both formulations already agree
    # (barrier is 0 there by construction).
    # NaN entries in p_events are not "unphysical" in the sense the barrier
    # below handles -- `p_events <= p_floor` is False for NaN, so without
    # this check a NaN p_event would silently fall through into
    # np.log(p_events) below, producing a NaN nll with no barrier, no
    # warning, and no gradient direction for the optimizer to recover from
    # (a quadratic barrier needs a finite distance-from-floor to be
    # meaningful; NaN has none). A NaN p_event almost always means a bug in
    # the user's compute_rates_func, so fail loudly here instead of
    # corrupting the fit silently.
    if np.any(np.isnan(p_events)):
        raise ValueError("compute_rates_func returned a NaN per-event density (p_events). Check compute_rates_func for division by zero, sqrt/log of a negative number, or similar.")

    p_floor = 1e-12
    unphysical = p_events <= p_floor
    if np.any(unphysical):
        p_events_safe = np.where(unphysical, p_floor, p_events)
        barrier = np.where(unphysical, 1e6 * (p_floor - p_events)**2, 0.0)
        return expected_total - np.sum(np.log(p_events_safe)) + np.sum(barrier)

    return expected_total - np.sum(np.log(p_events))


# --- 2. Grid Search Optimizers (Unbinned) ---
def unconditional_fit_grid_unbinned(obs_events, pdf_components, full_grid_points, compute_rates_func, extra_nll=None):
    """
    Performs a brute-force global scan over the parameter space to find the unconditional MLE.

    Because evaluating a PDF function pdf(obs_events) for every parameter combination
    can be incredibly slow in Python, this function evaluates each PDF exactly ONCE for
    the observed dataset, and reuses those static arrays across the entire grid loop.

    Parameters:
    -----------
    obs_events : array_like
        The observed data events.
    pdf_components : list of callable
        Probability density functions, one per model component (e.g. signal, background).
    full_grid_points : array_like, 2D
        Matrix containing all parameter combinations to scan.
    compute_rates_func : callable
        User-provided expected rates function.

    Returns:
    --------
    min_nll : float
        The absolute minimum negative log-likelihood found.
    best_params : array_like, 1D
        The parameter combination corresponding to min_nll.
    """
    len_obs = len(obs_events)
    # Pre-compute structural PDFs to avoid massive redundant computation inside the loop
    probs = [pdf(obs_events) if len_obs > 0 else np.array([]) for pdf in pdf_components]

    min_nll = 1e10
    best_params = full_grid_points[0].copy()

    for row in range(len(full_grid_points)):
        p = full_grid_points[row]
        nll = calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
        if extra_nll is not None:
            nll += extra_nll(p)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()

    return min_nll, best_params

def conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, obs_events, pdf_components, cond_grid_points, compute_rates_func, extra_nll=None):
    """
    Performs a brute-force 1D conditional profile scan for the unbinned likelihood.
    Fixes one parameter of interest and scans the remaining grid.

    Parameters:
    -----------
    test_val : float
        The fixed value for the parameter of interest.
    fix_idx : int
        The index of the parameter to fix.
    n_params : int
        Total number of parameters.
    obs_events : array_like
        The observed data events.
    pdf_components : list of callable
        Probability density functions, one per model component.
    cond_grid_points : array_like, 2D
        The sub-grid of nuisance parameters to profile over.
    compute_rates_func : callable
        User-provided expected rates function.

    Returns:
    --------
    min_nll : float
        The minimum negative log-likelihood given the fixed parameter.
    best_params : array_like, 1D
        The parameter combination that produced min_nll.
    """
    len_obs = len(obs_events)
    probs = [pdf(obs_events) if len_obs > 0 else np.array([]) for pdf in pdf_components]

    min_nll = 1e10
    best_params = np.zeros(n_params)

    # Pre-allocate array and lock the parameter of interest
    p = np.zeros(n_params)
    p[fix_idx] = test_val

    for row in range(len(cond_grid_points)):
        free_p = cond_grid_points[row]
        free_i = 0

        # Inject the free nuisance parameters into the full parameter array
        for i in range(n_params):
            if i != fix_idx:
                p[i] = free_p[free_i]
                free_i += 1

        nll = calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
        if extra_nll is not None:
            nll += extra_nll(p)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()

    return min_nll, best_params

def conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, obs_events, pdf_components, cond_grid_points, compute_rates_func, extra_nll=None):
    """
    Performs a brute-force 2D conditional profile scan for the unbinned likelihood.
    Fixes two parameters of interest to map joint contours.

    Parameters:
    -----------
    test_vA, test_vB : float
        The fixed values for the two parameters of interest.
    fix_A, fix_B : int
        The indices of the fixed parameters.
    n_params : int
        Total number of parameters.
    obs_events : array_like
        The observed data events.
    pdf_components : list of callable
        Probability density functions, one per model component.
    cond_grid_points : array_like, 2D
        The sub-grid of nuisance parameters to profile over.
    compute_rates_func : callable
        User-provided expected rates function.

    Returns:
    --------
    min_nll : float
        The minimum negative log-likelihood given the two fixed parameters.
    best_params : array_like, 1D
        The parameter combination that produced min_nll.
    """
    len_obs = len(obs_events)
    probs = [pdf(obs_events) if len_obs > 0 else np.array([]) for pdf in pdf_components]

    min_nll = 1e10
    best_params = np.zeros(n_params)

    p = np.zeros(n_params)
    p[fix_A] = test_vA
    p[fix_B] = test_vB

    for row in range(len(cond_grid_points)):
        free_p = cond_grid_points[row]
        free_i = 0
        for i in range(n_params):
            if i != fix_A and i != fix_B:
                p[i] = free_p[free_i]
                free_i += 1

        nll = calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
        if extra_nll is not None:
            nll += extra_nll(p)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()

    return min_nll, best_params


# --- 3. Toy Generators (Unbinned) ---
def generate_and_fit_toys_grid_unbinned_1d(test_val, fix_idx, true_params, n_params, pdf_components,
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool,
                                           compute_rates_func, generate_toy_func, extra_nll=None):
    """
    Sequentially generates and evaluates 1D unbinned toys entirely using grid search.

    Used when gradient optimizers fail and external parallelization is disabled.
    It evaluates the Profile Likelihood Ratio (cond_nll - uncond_nll) for each
    simulated unbinned dataset.

    Parameters:
    -----------
    test_val : float
        The fixed value for the parameter of interest.
    fix_idx : int
        The index of the fixed parameter.
    true_params : array_like, 1D
        The physical parameters dictating the expected event yields.
    n_params : int
        Total number of parameters in the model.
    pdf_components : list of callable
        Probability density functions, one per model component.
    full_grid_points : array_like, 2D
        Grid for the unconditional fit.
    cond_grid_points : array_like, 2D
        Grid for the conditional (nuisance parameter) fit.
    n_toys : int
        Number of pseudo-experiments to generate.
    S_mc_pool, B_mc_pool : np.ndarray
        Pre-generated continuous events for bootstrapping.
    compute_rates_func : callable
        User-provided physics mapping function.
    generate_toy_func : callable
        User-provided function to bootstrap the toy data.

    Returns:
    --------
    t_statistics : array_like, 1D
        Array of length `n_toys` containing the test statistic for each toy.
    """
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_toy_func(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, pdf_components, full_grid_points, compute_rates_func, extra_nll)
        cond_nll, _ = conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, toy_events, pdf_components, cond_grid_points, compute_rates_func, extra_nll)

        # Bounded at 0.0 to correct floating point inaccuracies when cond_nll ≈ uncond_nll
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics

def generate_and_fit_toys_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, true_params, n_params, pdf_components,
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool,
                                           compute_rates_func, generate_toy_func, extra_nll=None):
    """
    Sequentially generates and evaluates 2D unbinned toys entirely using grid search.

    Parameters:
    -----------
    test_vA, test_vB : float
        The fixed values for the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    true_params : array_like, 1D
        The physical parameters dictating the expected event yields.
    n_params : int
        Total number of parameters in the model.
    pdf_components : list of callable
        Probability density functions, one per model component.
    full_grid_points : array_like, 2D
        Grid for the unconditional fit.
    cond_grid_points : array_like, 2D
        Grid for the conditional (nuisance parameter) fit.
    n_toys : int
        Number of pseudo-experiments to generate.
    S_mc_pool, B_mc_pool : np.ndarray
        Pre-generated continuous events for bootstrapping.
    compute_rates_func : callable
        User-provided physics mapping function.
    generate_toy_func : callable
        User-provided function to bootstrap the toy data.

    Returns:
    --------
    t_statistics : array_like, 1D
        Array of length `n_toys` containing the test statistic for each toy.
    """
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_toy_func(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, pdf_components, full_grid_points, compute_rates_func, extra_nll)
        cond_nll, _ = conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, toy_events, pdf_components, cond_grid_points, compute_rates_func, extra_nll)

        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics