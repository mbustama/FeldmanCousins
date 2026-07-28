"""
Continuous Optimization Module

This file contains the continuous optimization routines used to minimize the
Negative Log-Likelihood (NLL) functions for both binned and unbinned models.
It provides interfaces for gradient-based local optimization via SciPy (L-BFGS-B)
and robust global optimization via UltraNest (Nested Sampling). These optimizers
are essential for profiling out nuisance parameters and finding the global
and conditional likelihood maxima required for the Profile Likelihood Ratio
test statistic in the Feldman-Cousins construction.

A note on joint/simplex-constrained parameters:
------------------------------------------------
Every `bounds_list` entry here is an INDEPENDENT per-parameter box constraint
only -- `param_i` in `[lo_i, hi_i]`, with no notion of a relationship between
two different parameters. Neither L-BFGS-B/SLSQP nor UltraNest's prior
transform (which maps a unit hypercube onto `bounds_list` one coordinate at a
time) have any native concept of a joint constraint like a simplex
(`a + b <= 1`) or a sphere (`a^2 + b^2 <= 1`). If your physical model has such
a constraint, do not work around it with a hand-rolled penalty function
multiplying your rate -- that reproduces the exact flat-gradient trap
described in `calc_nll`'s docstring, just user-side instead of library-side.
Two purpose-built mechanisms exist instead, and can be used together:
  - `bounds_func` (accepted by every function below): tightens a free
    parameter's own box bound based on whatever OTHER parameter(s) the
    current scan step has fixed. Use this when the constraint only ever
    couples the scan's currently-FIXED test parameter(s) to a free nuisance
    parameter -- it's cheap, since it keeps the optimizer's search box (and
    its bounds-midpoint starting guess) always physical, with no boundary-
    adjacent evaluations needed at all.
  - `constraints` (accepted by `unconditional_fit_scipy`, the two
    `conditional_fit_*_scipy` functions, and the two `conditional_fit_*_
    ultranest` functions besides `unconditional_fit_ultranest`): a list of
    `scipy.optimize.LinearConstraint`/`NonlinearConstraint` objects,
    expressed in the FULL n_params space. Use this when the constraint
    couples multiple parameters that are SIMULTANEOUSLY free (e.g. neither
    is ever the scan's fixed test parameter) -- a case `bounds_func` cannot
    express. On the scipy side this switches `method` from 'L-BFGS-B' to
    'SLSQP' (the only two of PyFC's methods that support `constraints`
    together with bounds); on the UltraNest side, a violated constraint
    just yields a very large negative log-likelihood at that point, since
    nested sampling doesn't use gradients and isn't vulnerable to the
    flat-region trap that motivates the scipy-side machinery.
See the README section "Handling Joint/Simplex-Constrained Parameters" for a
full worked example (a neutrino flavor-fraction fit with `f_e + f_mu <= 1`).

Created: v0.1.0 (July 24, 2026)
Last modified: v0.10.0
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import warnings

import numpy as np

from .binned import calc_nll
from .unbinned import calc_nll_unbinned

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import ultranest
    ULTRANEST_AVAILABLE = True
except ImportError:
    ULTRANEST_AVAILABLE = False


# --- 0. Shared Constraint Helpers ---
# `scipy.optimize.minimize` only accepts `constraints=` for methods 'COBYLA',
# 'SLSQP', 'trust-constr' -- NOT the default 'L-BFGS-B'. These helpers let
# `constraints` (a list of LinearConstraint/NonlinearConstraint objects,
# expressed in the FULL n_params space) be supplied to the fit functions
# below without changing default behavior when no constraints are given.
def _resolve_scipy_method(constraints, scipy_method):
    """
    Picks the scipy.optimize.minimize method: an explicit `scipy_method`
    always wins; otherwise 'SLSQP' if constraints were supplied (fast,
    supports bounds+constraints together), else the original 'L-BFGS-B'
    (unchanged default when constraints is None/empty).
    """
    if scipy_method is not None:
        return scipy_method
    if constraints:
        return 'SLSQP'
    return 'L-BFGS-B'


def _project_linear_constraint(constraint, fixed_indices_values, free_indices):
    """
    Projects a single constraint expressed over the FULL n_params vector down
    to the reduced subspace of `free_indices`, substituting the currently-
    fixed parameter value(s) as a constant offset. Used by
    conditional_fit_1d_scipy/conditional_fit_2d_scipy, whose optimization
    vector only spans the free (profiled) parameters -- the fixed
    parameter(s) are not part of `x` at all.

    For a LinearConstraint (lb <= A x <= ub), splitting A's columns into the
    free and fixed parts gives A x = A_free x_free + A_fixed x_fixed, so the
    projected constraint is (lb - A_fixed @ x_fixed) <= A_free x_free <=
    (ub - A_fixed @ x_fixed). Sanity check: for a single linear constraint
    like f_e + f_mu <= 1 with f_e fixed, this reduces to exactly the
    tightened box bound produced by the `bounds_func` mechanism.

    For a NonlinearConstraint, the fixed values are substituted into a
    wrapper around the original `fun` (and `jac`, if one was supplied),
    reconstructing the full parameter vector before evaluating it.

    Parameters:
    -----------
    constraint : scipy.optimize.LinearConstraint or scipy.optimize.NonlinearConstraint
        A single constraint expressed over the full n_params vector.
    fixed_indices_values : dict[int, float]
        Mapping of fixed-parameter index -> its current test value.
    free_indices : list of int
        Indices (in the full n_params space) of the parameters being optimized,
        in the order they appear in the reduced optimization vector.

    Returns:
    --------
    A new constraint object of the same type, expressed over only
    `len(free_indices)` variables (in that order).
    """
    fixed_indices = list(fixed_indices_values.keys())
    fixed_values = np.array([fixed_indices_values[i] for i in fixed_indices], dtype=float)
    n_params = len(free_indices) + len(fixed_indices)

    if isinstance(constraint, optimize.LinearConstraint):
        A = np.atleast_2d(np.asarray(constraint.A, dtype=float))
        A_free = A[:, free_indices]
        offset = A[:, fixed_indices] @ fixed_values if fixed_indices else 0.0
        lb = np.asarray(constraint.lb, dtype=float) - offset
        ub = np.asarray(constraint.ub, dtype=float) - offset
        return optimize.LinearConstraint(A_free, lb, ub)

    if isinstance(constraint, optimize.NonlinearConstraint):
        def _expand(free_p):
            p = np.zeros(n_params)
            for idx, val in fixed_indices_values.items():
                p[idx] = val
            for idx, val in zip(free_indices, free_p):
                p[idx] = val
            return p

        def fun_free(free_p):
            return constraint.fun(_expand(free_p))

        kwargs = {}
        if callable(getattr(constraint, "jac", None)):
            def jac_free(free_p):
                return np.asarray(constraint.jac(_expand(free_p)))[:, free_indices]
            kwargs["jac"] = jac_free

        return optimize.NonlinearConstraint(fun_free, constraint.lb, constraint.ub, **kwargs)

    raise TypeError(f"Unsupported constraint type for projection: {type(constraint)!r}")


def _project_constraints(constraints, fixed_indices_values, free_indices):
    """Applies `_project_linear_constraint` to a list of constraints (or None)."""
    if not constraints:
        return []
    return [_project_linear_constraint(c, fixed_indices_values, free_indices) for c in constraints]


def _constraints_satisfied(constraints, p, tol=1e-8):
    """
    Evaluates a list of constraints (expressed over the full n_params vector)
    at the fully-assembled parameter point `p`. Used by the UltraNest fit
    functions: since nested sampling doesn't use gradients, it doesn't suffer
    the flat-gradient trap that motivates the scipy-side SLSQP/projection
    machinery above, so a violated constraint can simply be signaled with a
    very large negative log-likelihood.
    """
    if not constraints:
        return True
    for c in constraints:
        val = np.atleast_1d(c.A @ p if isinstance(c, optimize.LinearConstraint) else c.fun(p))
        lb = np.atleast_1d(c.lb)
        ub = np.atleast_1d(c.ub)
        if np.any(val < lb - tol) or np.any(val > ub + tol):
            return False
    return True


def _minimize_with_restarts(cost, x0, bounds, method, extra_kwargs, n_restarts=1, rng=None, label=""):
    """
    Runs `scipy.optimize.minimize`, surfacing and acting on `res.success`
    (previously ignored entirely -- only `res.fun`/`res.x` were read).

    With the default `n_restarts=1`, this still runs a single optimization
    from `x0` (identical to the pre-FIX-4 call), but now additionally: if
    that single attempt does not converge, it retries once from a randomly
    perturbed starting point (cheap -- it only costs extra evaluations on
    the fits that actually need it) and logs a warning if even the retry
    fails to converge, instead of silently returning a possibly poor fit.

    With `n_restarts > 1`, several starting points are tried -- `x0` itself,
    the bounds midpoint (a fresh start, paired with `x0` specifically so
    that a bad neighbor-seeded starting point can't silently propagate
    forward without a sanity-check alternative), and (if `n_restarts > 2`)
    additional randomized points within `bounds` -- keeping whichever
    converges to the lowest NLL.

    Parameters:
    -----------
    cost : callable
        The objective function to minimize.
    x0 : array_like
        The primary starting point (e.g. the bounds midpoint, or a
        neighbor-seeded guess).
    bounds : list of (lo, hi) tuples
        Box bounds passed to `scipy.optimize.minimize`.
    method : str
        The `scipy.optimize.minimize` method.
    extra_kwargs : dict
        Additional kwargs forwarded to `scipy.optimize.minimize` (e.g.
        `constraints`).
    n_restarts : int, optional
        Number of distinct starting points to try (default 1: unchanged
        pre-FIX-4 behavior, aside from the res.success check above).
    rng : np.random.Generator, optional
        Random generator for the randomized restart points.
    label : str, optional
        Human-readable context appended to any convergence warning.

    Returns:
    --------
    scipy.optimize.OptimizeResult
        The best result found (lowest `.fun` among converged attempts, or
        the least-bad attempt if none converged).
    """
    def _solve(x0_try):
        return optimize.minimize(cost, x0=x0_try, bounds=bounds, method=method, **extra_kwargs)

    candidates = [np.asarray(x0, dtype=float)]
    if n_restarts > 1:
        mid = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
        if not np.allclose(mid, candidates[0], atol=1e-12):
            candidates.append(mid)
        rng = rng if rng is not None else np.random.default_rng()
        while len(candidates) < n_restarts:
            candidates.append(np.array([rng.uniform(b[0], b[1]) for b in bounds]))
        candidates = candidates[:n_restarts]

    best = None
    for x0_try in candidates:
        res = _solve(x0_try)
        if best is None or (res.success and not best.success) or (res.success == best.success and res.fun < best.fun):
            best = res

    if not best.success:
        rng = rng if rng is not None else np.random.default_rng()
        perturb_scale = np.array([0.1 * (b[1] - b[0]) for b in bounds])
        perturbed = np.clip(candidates[0] + rng.normal(scale=perturb_scale), [b[0] for b in bounds], [b[1] for b in bounds])
        retry_res = _solve(perturbed)
        if retry_res.success or retry_res.fun < best.fun:
            best = retry_res
        if not best.success:
            warnings.warn(f"scipy.optimize.minimize did not converge{label} (status={best.status}): {best.message}")

    return best


# --- 1. Scipy Continuous Optimizers ---
def unconditional_fit_scipy(data, n_params, bounds_list, compute_rates_func, seed=None,
                            pdf_components=None,
                            likelihood_type="binned", S_sumw2=None, B_sumw2=None, use_finite_mc=False,
                            bounds_func=None, constraints=None, scipy_method=None, n_restarts=1):
    """
    Performs an unconditional global maximum likelihood fit using SciPy's L-BFGS-B.
    
    Statistical Theory & Assumptions:
    To evaluate the denominator of the Profile Likelihood Ratio (PLR), we must find 
    the parameter combination that yields the absolute minimum Negative Log-Likelihood 
    (NLL). 
    
    Mathematical Expression:
    NLL_uncond = min_{theta} NLL(theta | data)
    where theta represents the full set of N free parameters in the model.
    
    The L-BFGS-B algorithm is a quasi-Newton method that approximates the Broyden-Fletcher-Goldfarb-Shanno 
    (BFGS) update to the Hessian matrix, scaled for bounded parameter spaces. It assumes 
    the likelihood surface is sufficiently smooth and twice-differentiable. Because it is 
    a local optimizer, it can be sensitive to the initial seed.

    Parameters:
    -----------
    data : array_like
        The observed dataset (binned counts of any shape, or unbinned events).
    n_params : int
        Total number of parameters in the model.
    bounds_list : list of tuples
        A list of (min, max) bounds for each parameter.
    compute_rates_func : callable
        User-provided mapping function matching parameters to physical expectations.
    seed : array_like, 1D, optional
        Initial parameter guess. If None, the midpoint of the bounds is used.
    pdf_components : list of callable, optional
        Probability density functions, one per model component (e.g. signal,
        background). Required when `likelihood_type="unbinned"`; each is
        evaluated exactly once on `data` and the resulting arrays are reused
        across every NLL evaluation in this fit. Unused for `"binned"`.
    likelihood_type : str
        Either "binned" or "unbinned".
    S_sumw2, B_sumw2 : array_like, optional
        Sum of squared MC weights (sumw2) per bin, for finite MC binned likelihoods.
    use_finite_mc : bool
        Flag to enable the Poisson-Gamma mixture likelihood for binned data.
    bounds_func : callable, optional
        Advanced hook for joint/simplex parameter constraints (see module
        docstring). If provided, called as
        `bounds_func({}, list(range(n_params)), bounds_list)` -- an empty
        `fixed_values` dict since nothing is fixed in an unconditional fit --
        and must return one `(lo, hi)` tuple per parameter, in index order.
        If None (default), `bounds_list` is used unmodified.
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints among the full n_params vector (see module
        docstring). When non-empty, `method` defaults to 'SLSQP' instead of
        'L-BFGS-B' (required, since L-BFGS-B does not support `constraints`).
        When None/empty (default), behavior and method are UNCHANGED from
        before this parameter existed.
    scipy_method : str, optional
        Explicit override for the `scipy.optimize.minimize` method (e.g.
        'trust-constr' for better-conditioned but slower constrained fits).
        Takes precedence over the constraints-based default.
    n_restarts : int, optional
        Number of distinct starting points to try, keeping whichever
        converges to the lowest NLL (see `_minimize_with_restarts`). Default
        1 preserves pre-FIX-4 behavior (a single fit from `x0`), except that
        `res.success` is now always checked; a non-converged result
        triggers one cheap perturbed retry and a warning if that also fails.

    Returns:
    --------
    min_nll : float
        The minimal negative log-likelihood found by the optimizer.
    best_params : array_like, 1D
        The parameter combination corresponding to min_nll.
    """
    if likelihood_type == "unbinned":
        len_obs = len(data)
        probs = [pdf(data) if len_obs > 0 else np.array([]) for pdf in pdf_components]

        def cost(params):
            return calc_nll_unbinned(params, len_obs, probs, compute_rates_func)
    else:
        def cost(params):
            return calc_nll(params, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func)

    if bounds_func is not None:
        resolved_bounds = bounds_func({}, list(range(n_params)), bounds_list)
    else:
        resolved_bounds = bounds_list

    if seed is not None:
        eps = 1e-8
        lb = np.array([b[0] for b in resolved_bounds]) + eps
        ub = np.array([b[1] for b in resolved_bounds]) - eps
        x0 = np.clip(seed, lb, ub)
    else:
        x0 = [(b[0] + b[1]) / 2.0 for b in resolved_bounds]

    method = _resolve_scipy_method(constraints, scipy_method)
    extra_kwargs = {"constraints": constraints} if constraints else {}

    res = _minimize_with_restarts(cost, x0, resolved_bounds, method, extra_kwargs, n_restarts, label=" (unconditional fit)")
    return res.fun, res.x

def conditional_fit_1d_scipy(test_val, fix_idx, n_params, data, bounds_list, compute_rates_func, seed=None,
                             pdf_components=None,
                             likelihood_type="binned", S_sumw2=None, B_sumw2=None, use_finite_mc=False,
                             bounds_func=None, constraints=None, scipy_method=None, n_restarts=1):
    """
    Performs a conditional maximum likelihood fit (profiling 1 parameter) using SciPy.
    
    Statistical Theory:
    To evaluate the numerator of the Profile Likelihood Ratio for 1D intervals, 
    one parameter of interest is fixed to a specific 'test_val', and the NLL is 
    minimized over the remaining (N-1) nuisance parameters.
    
    Mathematical Expression:
    NLL_cond = min_{nu} NLL(theta_fixed, nu | data)
    where theta_fixed is the parameter of interest and nu are the nuisance parameters.

    Parameters:
    -----------
    test_val : float
        The value at which to fix the parameter of interest.
    fix_idx : int
        The index of the parameter being fixed.
    n_params : int
        Total number of parameters.
    data, bounds_list : various
        Dataset (binned counts of any shape, or unbinned events) and boundary
        constraints.
    compute_rates_func : callable
        User-provided mapping function matching parameters to physical expectations.
    seed : array_like, 1D, optional
        Initial guess for the *free* parameters.
    pdf_components : list of callable, optional
        Probability density functions, one per model component. Required
        when `likelihood_type="unbinned"`; unused for `"binned"`.
    likelihood_type, S_sumw2, B_sumw2, use_finite_mc : various
        Likelihood formulation configurations.
    bounds_func : callable, optional
        Advanced hook for expressing a joint/simplex constraint between the
        currently-fixed `test_val` and the free nuisance parameters (see
        module docstring). If provided, called as
        `bounds_func({fix_idx: test_val}, free_indices, bounds_list)` and must
        return one `(lo, hi)` tuple per entry of `free_indices`, in that
        order. This lets the box itself be tightened so it is always
        physical (e.g. `f_mu <= 1 - f_e_test`), which also fixes the default
        bounds-midpoint starting guess so it can never land in forbidden
        territory. If None (default), each free parameter uses its static
        `bounds_list[i]` unmodified -- identical to pre-FIX-2 behavior.
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints among the FULL n_params vector (see module
        docstring and `_project_linear_constraint`). Since `fix_idx` is not
        part of this function's optimization vector, each constraint is
        projected down to the free-parameter subspace before being passed to
        SciPy, substituting `test_val` as a constant offset. When non-empty,
        `method` defaults to 'SLSQP' instead of 'L-BFGS-B'. When None/empty
        (default), behavior and method are UNCHANGED from before this
        parameter existed. Use `constraints` (rather than `bounds_func`) for
        constraints among multiple simultaneously-FREE nuisance parameters,
        which `bounds_func` cannot express.
    scipy_method : str, optional
        Explicit override for the `scipy.optimize.minimize` method.
    n_restarts : int, optional
        Number of distinct starting points to try, keeping whichever
        converges to the lowest NLL (see `_minimize_with_restarts`). Default
        1 preserves pre-FIX-4 behavior (a single fit from `x0`), except that
        `res.success` is now always checked; a non-converged result
        triggers one cheap perturbed retry and a warning if that also fails.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the fixed test_val and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i != fix_idx]
    if bounds_func is not None:
        free_bounds = bounds_func({fix_idx: test_val}, free_indices, bounds_list)
    else:
        free_bounds = [bounds_list[i] for i in free_indices]
    projected_constraints = _project_constraints(constraints, {fix_idx: test_val}, free_indices)

    if likelihood_type == "unbinned":
        len_obs = len(data)
        probs = [pdf(data) if len_obs > 0 else np.array([]) for pdf in pdf_components]

        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
    else:
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll(p, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func)

    if seed is not None:
        eps = 1e-8
        lb = np.array([b[0] for b in free_bounds]) + eps
        ub = np.array([b[1] for b in free_bounds]) - eps
        x0 = np.clip(seed, lb, ub)
    else:
        x0 = [(b[0] + b[1]) / 2.0 for b in free_bounds]

    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_idx] = test_val
        # No free parameters left to optimize over, so `constraints` was
        # never projected into anything scipy could enforce (that only
        # happens inside _minimize_with_restarts below, which this branch
        # skips entirely). Check directly against the fully-assembled
        # point instead, same flat-penalty convention used on the
        # UltraNest side (module docstring) since there's no optimizer
        # step here to get stuck in a flat-gradient trap.
        if not _constraints_satisfied(constraints, p):
            return 1e10, p
        return cost(np.array([])), p

    method = _resolve_scipy_method(projected_constraints, scipy_method)
    extra_kwargs = {"constraints": projected_constraints} if projected_constraints else {}

    res = _minimize_with_restarts(cost, x0, free_bounds, method, extra_kwargs, n_restarts, label=f" (conditional 1D fit, fix_idx={fix_idx})")

    best_p = np.zeros(n_params)
    best_p[fix_idx] = test_val
    for idx, free_val in zip(free_indices, res.x):
        best_p[idx] = free_val

    return res.fun, best_p

def conditional_fit_2d_scipy(test_vA, test_vB, fix_A, fix_B, n_params, data, bounds_list, compute_rates_func, seed=None,
                             pdf_components=None,
                             likelihood_type="binned", S_sumw2=None, B_sumw2=None, use_finite_mc=False,
                             bounds_func=None, constraints=None, scipy_method=None, n_restarts=1):
    """
    Performs a conditional maximum likelihood fit (profiling 2 parameters) using SciPy.
    
    Statistical Theory:
    Analogous to the 1D profiling function, but computes the joint conditional 
    minimum for a 2D confidence region by fixing two parameters and optimizing 
    the remaining (N-2) nuisance parameters.

    Parameters:
    -----------
    test_vA, test_vB : float
        The values at which to fix the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    n_params : int
        Total number of parameters.
    data, bounds_list : various
        Dataset (binned counts of any shape, or unbinned events) and boundary
        constraints.
    compute_rates_func : callable
        User-provided mapping function matching parameters to physical expectations.
    seed : array_like, 1D, optional
        Initial guess for the *free* parameters.
    pdf_components : list of callable, optional
        Probability density functions, one per model component. Required
        when `likelihood_type="unbinned"`; unused for `"binned"`.
    likelihood_type, S_sumw2, B_sumw2, use_finite_mc : various
        Likelihood formulation configurations.
    bounds_func : callable, optional
        Advanced hook for expressing a joint/simplex constraint between the
        two currently-fixed test parameters and the free nuisance parameters
        (see module docstring). If provided, called as
        `bounds_func({fix_A: test_vA, fix_B: test_vB}, free_indices, bounds_list)`
        and must return one `(lo, hi)` tuple per entry of `free_indices`, in
        that order. If None (default), each free parameter uses its static
        `bounds_list[i]` unmodified -- identical to pre-FIX-2 behavior.
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints among the FULL n_params vector, projected down to
        the free-parameter subspace (substituting `test_vA`/`test_vB` as a
        constant offset) before being passed to SciPy. When non-empty,
        `method` defaults to 'SLSQP' instead of 'L-BFGS-B'. When None/empty
        (default), behavior and method are UNCHANGED from before this
        parameter existed.
    scipy_method : str, optional
        Explicit override for the `scipy.optimize.minimize` method.
    n_restarts : int, optional
        Number of distinct starting points to try, keeping whichever
        converges to the lowest NLL (see `_minimize_with_restarts`). Default
        1 preserves pre-FIX-4 behavior (a single fit from `x0`), except that
        `res.success` is now always checked; a non-converged result
        triggers one cheap perturbed retry and a warning if that also fails.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the two fixed values and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i not in (fix_A, fix_B)]
    if bounds_func is not None:
        free_bounds = bounds_func({fix_A: test_vA, fix_B: test_vB}, free_indices, bounds_list)
    else:
        free_bounds = [bounds_list[i] for i in free_indices]
    projected_constraints = _project_constraints(constraints, {fix_A: test_vA, fix_B: test_vB}, free_indices)

    if likelihood_type == "unbinned":
        len_obs = len(data)
        probs = [pdf(data) if len_obs > 0 else np.array([]) for pdf in pdf_components]

        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
    else:
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll(p, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func)

    if seed is not None:
        eps = 1e-8
        lb = np.array([b[0] for b in free_bounds]) + eps
        ub = np.array([b[1] for b in free_bounds]) - eps
        x0 = np.clip(seed, lb, ub)
    else:
        x0 = [(b[0] + b[1]) / 2.0 for b in free_bounds]

    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_A] = test_vA
        p[fix_B] = test_vB
        # See conditional_fit_1d_scipy's identical branch for why this
        # check is needed here: with no free parameters, `constraints` is
        # never projected/enforced by _minimize_with_restarts below, which
        # this branch skips entirely.
        if not _constraints_satisfied(constraints, p):
            return 1e10, p
        return cost(np.array([])), p

    method = _resolve_scipy_method(projected_constraints, scipy_method)
    extra_kwargs = {"constraints": projected_constraints} if projected_constraints else {}

    res = _minimize_with_restarts(cost, x0, free_bounds, method, extra_kwargs, n_restarts, label=f" (conditional 2D fit, fix_A={fix_A}, fix_B={fix_B})")

    best_p = np.zeros(n_params)
    best_p[fix_A] = test_vA
    best_p[fix_B] = test_vB
    for idx, free_val in zip(free_indices, res.x):
        best_p[idx] = free_val

    return res.fun, best_p


# --- 2. UltraNest Optimizers ---
def _run_ultranest_with_retry(param_names, log_likelihood, prior_transform, run_kwargs, max_attempts=3):
    """
    Constructs an `ultranest.ReactiveNestedSampler` and calls `.run(**run_kwargs)`,
    retrying with a fresh sampler instance on a specific, still-open UltraNest bug
    (confirmed present through at least v4.5.0 and the current GitHub master, in
    `ReactiveNestedSampler._find_strategy`): an internal random subsample
    (`itmax = np.random.choice(len(w), p=w)`) can occasionally select too few
    saved iterations, producing a degenerate 1D `logweights` array where 2D
    indexing is expected, and crashing with `IndexError: too many indices for
    array: array is 1-dimensional, but 2 were indexed`. This is triggered by
    UltraNest's own internal exploration randomness, not by anything in the
    likelihood/prior being sampled, so a fresh attempt reliably avoids the
    degenerate draw -- this mirrors `_minimize_with_restarts`'s retry-on-failure
    pattern for the scipy path.
    """
    last_exc = None
    for attempt in range(max_attempts):
        sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=None)
        try:
            return sampler.run(**run_kwargs)
        except IndexError as e:
            if "too many indices for array" not in str(e):
                raise
            last_exc = e
            if attempt < max_attempts - 1:
                warnings.warn(
                    f"UltraNest hit a known internal bug (attempt {attempt + 1}/{max_attempts}): {e}. "
                    "Retrying with a fresh sampler."
                )
    raise last_exc


def unconditional_fit_ultranest(data, n_params, bounds_list, compute_rates_func, verbose=1,
                                pdf_components=None,
                                likelihood_type="binned", S_sumw2=None, B_sumw2=None, use_finite_mc=False,
                                bounds_func=None, constraints=None):
    """
    Performs an unconditional global maximum likelihood fit using UltraNest.
    
    Statistical Theory & Assumptions:
    UltraNest relies on Reactive Nested Sampling, an algorithm traditionally used 
    for Bayesian evidence computation and posterior sampling. However, because it 
    exhaustively maps the likelihood surface across the entire prior volume without 
    relying on gradients, it acts as a highly robust global optimizer, virtually 
    immune to local minima traps. 
    
    The algorithm evaluates the likelihood by converting a uniformly sampled unit 
    hypercube [0, 1]^N into the physical parameter bounds via a prior transform. 
    The returned value is the Maximum Likelihood Estimate (MLE) discovered during 
    the integration phase.

    Parameters:
    -----------
    data : array_like
        The observed dataset (binned counts of any shape, or unbinned events).
    n_params : int
        Total number of parameters in the model.
    bounds_list : list of tuples
        A list of (min, max) bounds for each parameter.
    compute_rates_func : callable
        User-provided mapping function matching parameters to physical expectations.
    verbose : int
        Verbosity level controlling UltraNest output (2 for full tracking).
    pdf_components : list of callable, optional
        Probability density functions, one per model component. Required
        when `likelihood_type="unbinned"`; unused for `"binned"`.
    likelihood_type, S_sumw2, B_sumw2, use_finite_mc : various
        Likelihood formulation configurations.
    bounds_func : callable, optional
        Advanced hook for joint/simplex parameter constraints (see module
        docstring). If provided, called as
        `bounds_func({}, list(range(n_params)), bounds_list)` -- an empty
        `fixed_values` dict since nothing is fixed in an unconditional fit --
        and must return one `(lo, hi)` tuple per parameter, in index order.
        If None (default), `bounds_list` is used unmodified.
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints among the full n_params vector. Since nested
        sampling doesn't use gradients, it doesn't suffer the flat-gradient
        trap that motivates the scipy-side SLSQP/projection machinery: a
        violated constraint simply makes `log_likelihood` return a very
        large negative value (-1e10) at that point, no projection needed.
        If None/empty (default), behavior is unchanged.

    Returns:
    --------
    min_nll : float
        The absolute minimum negative log-likelihood (-1 * max_logL).
    best_params : array_like, 1D
        The parameter combination corresponding to the maximum log-likelihood point.
    """
    resolved_bounds = bounds_func({}, list(range(n_params)), bounds_list) if bounds_func is not None else bounds_list

    def prior_transform(cube):
        return np.array([cube[i] * (resolved_bounds[i][1] - resolved_bounds[i][0]) + resolved_bounds[i][0] for i in range(n_params)])

    if likelihood_type == "unbinned":
        len_obs = len(data)
        probs = [pdf(data) if len_obs > 0 else np.array([]) for pdf in pdf_components]

        def log_likelihood(p):
            if not _constraints_satisfied(constraints, p):
                return -1e10
            return -calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
    else:
        def log_likelihood(p):
            if not _constraints_satisfied(constraints, p):
                return -1e10
            return -calc_nll(p, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func)

    param_names = [f'p{i+1}' for i in range(n_params)]
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = _run_ultranest_with_retry(param_names, log_likelihood, prior_transform, run_kwargs)
    return -result['maximum_likelihood']['logl'], result['maximum_likelihood']['point']

def conditional_fit_1d_ultranest(test_val, fix_idx, n_params, data, bounds_list, compute_rates_func, verbose=1,
                                 pdf_components=None,
                                 likelihood_type="binned", S_sumw2=None, B_sumw2=None, use_finite_mc=False,
                                 bounds_func=None, constraints=None):
    """
    Performs a conditional maximum likelihood fit (profiling 1 parameter) using UltraNest.
    
    Statistical Theory:
    Executes Reactive Nested Sampling in the reduced (N-1) dimensional subspace 
    of nuisance parameters to robustly find the conditional maximum likelihood.
    This prevents false confidence interval exclusions that can occur if a gradient 
    optimizer falls into a local minimum when profiling.

    Parameters:
    -----------
    test_val : float
        The value at which to fix the parameter of interest.
    fix_idx : int
        The index of the parameter being fixed.
    n_params : int
        Total number of parameters in the model.
    data, bounds_list : various
        Dataset (binned counts of any shape, or unbinned events) and boundary
        constraints.
    compute_rates_func : callable
        User-provided mapping function matching parameters to physical expectations.
    verbose : int
        Verbosity level.
    pdf_components : list of callable, optional
        Probability density functions, one per model component. Required
        when `likelihood_type="unbinned"`; unused for `"binned"`.
    likelihood_type, S_sumw2, B_sumw2, use_finite_mc : various
        Likelihood formulation configurations.
    bounds_func : callable, optional
        Advanced hook for expressing a joint/simplex constraint between the
        currently-fixed `test_val` and the free nuisance parameters (see
        module docstring). If provided, called as
        `bounds_func({fix_idx: test_val}, free_indices, bounds_list)` and must
        return one `(lo, hi)` tuple per entry of `free_indices`, in that
        order. If None (default), each free parameter uses its static
        `bounds_list[i]` unmodified -- identical to pre-FIX-2 behavior.
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints among the full n_params vector, evaluated directly
        on the fully-assembled parameter point (no projection needed --
        nested sampling doesn't use gradients, so a violated constraint
        simply makes `log_likelihood` return -1e10 at that point). If
        None/empty (default), behavior is unchanged.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the fixed test_val and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i != fix_idx]
    if bounds_func is not None:
        free_bounds = bounds_func({fix_idx: test_val}, free_indices, bounds_list)
    else:
        free_bounds = [bounds_list[i] for i in free_indices]

    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_idx] = test_val
        # With no free parameters, `log_likelihood` below (which is where
        # `_constraints_satisfied` normally gets checked) never runs --
        # this branch returns directly instead. Check here too, same
        # flat-penalty convention as the module's other constraint checks.
        if not _constraints_satisfied(constraints, p):
            return 1e10, p
        if likelihood_type == "unbinned":
            probs = [pdf(data) if len(data) > 0 else np.array([]) for pdf in pdf_components]
            return calc_nll_unbinned(p, len(data), probs, compute_rates_func), p
        else:
            return calc_nll(p, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func), p

    def prior_transform(cube):
        return np.array([cube[i] * (free_bounds[i][1] - free_bounds[i][0]) + free_bounds[i][0] for i in range(len(free_bounds))])

    if likelihood_type == "unbinned":
        len_obs = len(data)
        probs = [pdf(data) if len_obs > 0 else np.array([]) for pdf in pdf_components]

        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            if not _constraints_satisfied(constraints, p):
                return -1e10
            return -calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
    else:
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            if not _constraints_satisfied(constraints, p):
                return -1e10
            return -calc_nll(p, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func)

    param_names = [f'f{i+1}' for i in range(len(free_bounds))]
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = _run_ultranest_with_retry(param_names, log_likelihood, prior_transform, run_kwargs)

    best_p = np.zeros(n_params)
    best_p[fix_idx] = test_val
    for idx, free_val in zip(free_indices, result['maximum_likelihood']['point']):
        best_p[idx] = free_val

    return -result['maximum_likelihood']['logl'], best_p

def conditional_fit_2d_ultranest(test_vA, test_vB, fix_A, fix_B, n_params, data, bounds_list, compute_rates_func, verbose=1,
                                 pdf_components=None,
                                 likelihood_type="binned", S_sumw2=None, B_sumw2=None, use_finite_mc=False,
                                 bounds_func=None, constraints=None):
    """
    Performs a conditional maximum likelihood fit (profiling 2 parameters) using UltraNest.
    
    Statistical Theory:
    Executes Reactive Nested Sampling in the (N-2) dimensional nuisance parameter 
    subspace to construct a robust 2D confidence region profile.

    Parameters:
    -----------
    test_vA, test_vB : float
        The values at which to fix the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    n_params : int
        Total number of parameters in the model.
    data, bounds_list : various
        Dataset (binned counts of any shape, or unbinned events) and boundary
        constraints.
    compute_rates_func : callable
        User-provided mapping function matching parameters to physical expectations.
    verbose : int
        Verbosity level.
    pdf_components : list of callable, optional
        Probability density functions, one per model component. Required
        when `likelihood_type="unbinned"`; unused for `"binned"`.
    likelihood_type, S_sumw2, B_sumw2, use_finite_mc : various
        Likelihood formulation configurations.
    bounds_func : callable, optional
        Advanced hook for expressing a joint/simplex constraint between the
        two currently-fixed test parameters and the free nuisance parameters
        (see module docstring). If provided, called as
        `bounds_func({fix_A: test_vA, fix_B: test_vB}, free_indices, bounds_list)`
        and must return one `(lo, hi)` tuple per entry of `free_indices`, in
        that order. If None (default), each free parameter uses its static
        `bounds_list[i]` unmodified -- identical to pre-FIX-2 behavior.
    constraints : list of scipy.optimize.LinearConstraint/NonlinearConstraint, optional
        Joint constraints among the full n_params vector, evaluated directly
        on the fully-assembled parameter point (no projection needed). If
        None/empty (default), behavior is unchanged.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the two fixed values and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i not in (fix_A, fix_B)]
    if bounds_func is not None:
        free_bounds = bounds_func({fix_A: test_vA, fix_B: test_vB}, free_indices, bounds_list)
    else:
        free_bounds = [bounds_list[i] for i in free_indices]

    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_A] = test_vA
        p[fix_B] = test_vB
        # See conditional_fit_1d_ultranest's identical branch: with no
        # free parameters, `log_likelihood` below (where
        # `_constraints_satisfied` is normally checked) never runs.
        if not _constraints_satisfied(constraints, p):
            return 1e10, p
        if likelihood_type == "unbinned":
            probs = [pdf(data) if len(data) > 0 else np.array([]) for pdf in pdf_components]
            return calc_nll_unbinned(p, len(data), probs, compute_rates_func), p
        else:
            return calc_nll(p, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func), p

    def prior_transform(cube):
        return np.array([cube[i] * (free_bounds[i][1] - free_bounds[i][0]) + free_bounds[i][0] for i in range(len(free_bounds))])

    if likelihood_type == "unbinned":
        len_obs = len(data)
        probs = [pdf(data) if len_obs > 0 else np.array([]) for pdf in pdf_components]

        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            if not _constraints_satisfied(constraints, p):
                return -1e10
            return -calc_nll_unbinned(p, len_obs, probs, compute_rates_func)
    else:
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            if not _constraints_satisfied(constraints, p):
                return -1e10
            return -calc_nll(p, data, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func)

    param_names = [f'f{i+1}' for i in range(len(free_bounds))]
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = _run_ultranest_with_retry(param_names, log_likelihood, prior_transform, run_kwargs)
    
    best_p = np.zeros(n_params)
    best_p[fix_A] = test_vA
    best_p[fix_B] = test_vB
    for idx, free_val in zip(free_indices, result['maximum_likelihood']['point']):
        best_p[idx] = free_val
        
    return -result['maximum_likelihood']['logl'], best_p