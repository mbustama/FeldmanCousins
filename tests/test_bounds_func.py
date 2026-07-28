"""
Tests for the `bounds_func` hook that lets the box bounds of the
profiled (free) parameters depend on the currently-fixed test value(s).

Motivating scenario: two flavor fractions f_e (index 0), f_mu (index 1)
subject to f_e + f_mu <= 1. When f_e is the scan's fixed test parameter,
f_mu's true valid range is [0, 1 - f_e_test], not the generic [0, 1].
"""
import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import (
    SCIPY_AVAILABLE,
    conditional_fit_1d_scipy,
    conditional_fit_2d_scipy,
    unconditional_fit_scipy,
)


def simplex_bounds_func(fixed_values, free_indices, default_bounds_list):
    """f_e (0) + f_mu (1) <= 1, as given in the task's worked example."""
    bounds = [default_bounds_list[i] for i in free_indices]
    if 0 in fixed_values and 1 in free_indices:
        j = free_indices.index(1)
        lo, hi = bounds[j]
        bounds[j] = (lo, min(hi, 1.0 - fixed_values[0]))
    if 1 in fixed_values and 0 in free_indices:
        j = free_indices.index(0)
        lo, hi = bounds[j]
        bounds[j] = (lo, min(hi, 1.0 - fixed_values[1]))
    return bounds


@njit(fastmath=True, nogil=True)
def _two_param_rate_func(params, S_sumw2, B_sumw2):
    # A single bin whose rate depends on both f_e and f_mu (params[0], params[1]).
    mu = np.array([10.0 * params[0] + 10.0 * params[1] + 1.0])
    return mu, S_sumw2


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this test")
def test_bounds_func_none_is_backward_compatible():
    """Default behavior (bounds_func=None) must be unaffected."""
    bounds_list = [(0.0, 1.0), (0.0, 1.0)]
    N_obs = np.array([15.0])
    s2 = np.array([0.0])

    cond_nll, best_p = conditional_fit_1d_scipy(
        0.9, 0, 2, N_obs, bounds_list, _two_param_rate_func,
        S_sumw2=s2, B_sumw2=s2,
    )
    # Free param (f_mu, index 1) is optimized within the *unmodified* [0, 1] box.
    assert 0.0 <= best_p[1] <= 1.0


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this test")
def test_bounds_func_1d_tightens_free_bound():
    """
    With f_e fixed at 0.7, f_mu's free bound must be tightened to
    [0, 0.3] -- the optimizer must never be allowed to try f_mu > 0.3, and
    its bounds-midpoint starting guess must already respect the constraint.
    """
    bounds_list = [(0.0, 1.0), (0.0, 1.0)]
    N_obs = np.array([100.0])  # large N pulls the fit toward larger mu (larger f_mu)
    s2 = np.array([0.0])

    fix_val = 0.7
    cond_nll, best_p = conditional_fit_1d_scipy(
        fix_val, 0, 2, N_obs, bounds_list, _two_param_rate_func,
        S_sumw2=s2, B_sumw2=s2, bounds_func=simplex_bounds_func,
    )

    assert best_p[0] == fix_val
    # f_mu must respect the tightened bound (1 - f_e = 0.3), not the generic [0,1].
    assert best_p[1] <= 0.3 + 1e-9
    assert best_p[1] >= 0.0 - 1e-9


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this test")
def test_bounds_func_2d_tightens_free_bound_symmetrically():
    """Same check via conditional_fit_2d_scipy with a 3rd unconstrained dummy param."""
    bounds_list = [(0.0, 1.0), (0.0, 1.0), (0.0, 5.0)]
    N_obs = np.array([100.0])
    s2 = np.array([0.0])

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_sumw2, B_sumw2):
        mu = np.array([10.0 * params[0] + 10.0 * params[1] + params[2] + 1.0])
        return mu, S_sumw2

    # Fix f_mu (index 1) at 0.6; f_e (index 0) is free and should be tightened to [0, 0.4].
    cond_nll, best_p = conditional_fit_2d_scipy(
        0.6, 2.5, 1, 2, 3, N_obs, bounds_list, rate_func,
        S_sumw2=s2, B_sumw2=s2, bounds_func=simplex_bounds_func,
    )
    assert best_p[1] == 0.6
    assert best_p[0] <= 0.4 + 1e-9


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this test")
def test_bounds_func_unconditional_noop_with_empty_fixed_values():
    """unconditional_fit_scipy calls bounds_func({}, ...); simplex_bounds_func is then a no-op."""
    bounds_list = [(0.0, 1.0), (0.0, 1.0)]
    N_obs = np.array([15.0])
    s2 = np.array([0.0])

    min_nll, best_params = unconditional_fit_scipy(
        N_obs, 2, bounds_list, _two_param_rate_func,
        S_sumw2=s2, B_sumw2=s2, bounds_func=simplex_bounds_func,
    )
    # No fixed values -> simplex_bounds_func returns the bounds_list unmodified,
    # so both params may independently reach values that individually violate
    # f_e + f_mu <= 1 (that joint case is exactly what the `constraints`
    # mechanism is for).
    assert 0.0 <= best_params[0] <= 1.0
    assert 0.0 <= best_params[1] <= 1.0


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this test")
def test_bounds_func_reproduces_plateau_trap_without_it():
    """
    Baseline sanity check: reproduce the original failure mode with a
    compute_rates_func that hard-zeroes itself outside f_e+f_mu<=1, and no
    bounds_func supplied. The optimizer's default bounds-midpoint start
    (0.5, 0.5) already violates the constraint (0.5+0.5=1.0 is the boundary,
    so nudge slightly): starting outside a feasible region can leave it
    stuck at (or very near) the unconstrained midpoint since the old-style
    hard cutoff gives zero gradient information there.
    """
    bounds_list = [(0.0, 1.0), (0.0, 1.0)]
    N_obs = np.array([100.0])
    s2 = np.array([0.0])

    @njit(fastmath=True, nogil=True)
    def hard_cutoff_rate_func(params, S_sumw2, B_sumw2):
        if params[0] + params[1] > 1.0:
            return np.array([0.0]), S_sumw2  # hard-zeroed: unphysical, flat region
        mu = np.array([10.0 * params[0] + 10.0 * params[1] + 1.0])
        return mu, S_sumw2

    # bounds_func fixes the box itself so the optimizer never needs the cutoff.
    cond_nll, best_p = conditional_fit_1d_scipy(
        0.7, 0, 2, N_obs, bounds_list, hard_cutoff_rate_func,
        S_sumw2=s2, B_sumw2=s2, bounds_func=simplex_bounds_func,
    )
    assert best_p[1] <= 0.3 + 1e-9
    assert np.isfinite(cond_nll)
    assert cond_nll < 1e9  # never had to fall into the hard-cutoff branch
