"""
Tests for FIX 3: native scipy LinearConstraint/NonlinearConstraint support
(and the shared `_project_linear_constraint` projection helper), for
expressing joint constraints among multiple SIMULTANEOUSLY-free nuisance
parameters -- the case `bounds_func` (FIX 2) cannot express.
"""
import time
from unittest.mock import patch

import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import (
    SCIPY_AVAILABLE,
    _constraints_satisfied,
    _project_linear_constraint,
    conditional_fit_1d_scipy,
    optimize,
    unconditional_fit_scipy,
)

pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for these tests")


def simplex_bounds_func(fixed_values, free_indices, default_bounds_list):
    """The FIX 2 example from the task, used here for the FIX2/FIX3 agreement check."""
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


# --- _project_linear_constraint unit tests ---
def test_project_linear_constraint_matches_fix2_tightened_box():
    """
    Sanity check called out in the task: for a single linear constraint like
    f_e + f_mu <= 1 with f_e fixed, projecting it down should reduce to
    exactly the tightened box bound produced by FIX 2's bounds_func.
    """
    constraint = optimize.LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)
    test_val = 0.7
    projected = _project_linear_constraint(constraint, {0: test_val}, free_indices=[1])

    # Original box for p1 was [0, 1]; FIX2 tightens it to [0, 1 - 0.7] = [0, 0.3].
    # The projected LinearConstraint should say: x1 <= 1 - 0.7 = 0.3 (lb still -inf).
    assert projected.A.shape == (1, 1)
    assert projected.A[0, 0] == pytest.approx(1.0)
    assert projected.ub[0] == pytest.approx(1.0 - test_val)
    assert projected.lb[0] == -np.inf


def test_project_linear_constraint_nonlinear_substitutes_fixed_value():
    def full_fun(p):
        return np.array([p[0]**2 + p[1]**2])

    constraint = optimize.NonlinearConstraint(full_fun, -np.inf, 1.0)
    projected = _project_linear_constraint(constraint, {0: 0.6}, free_indices=[1])

    # At free_p = [0.8], full point is [0.6, 0.8] -> 0.36 + 0.64 = 1.0
    val = projected.fun(np.array([0.8]))
    assert val[0] == pytest.approx(1.0)


# --- FIX2 vs FIX3 agreement (overlapping case) ---
@njit(fastmath=True, nogil=True)
def _two_param_rate_func(params, S_sigma2, B_sigma2):
    mu = np.array([10.0 * params[0] + 10.0 * params[1] + 1.0])
    return mu, S_sigma2


def test_bounds_func_and_constraints_agree_on_overlapping_case():
    """
    For a constraint between the fixed test parameter and a single free
    nuisance parameter, bounds_func (FIX 2) and constraints (FIX 3) should
    reach the same optimum -- this is the case where they overlap.
    """
    bounds_list = [(0.0, 1.0), (0.0, 1.0)]
    N_obs = np.array([100.0])
    s2 = np.array([0.0])
    fix_val = 0.7

    _, best_p_bf = conditional_fit_1d_scipy(
        fix_val, 0, 2, N_obs, bounds_list, _two_param_rate_func,
        S_sigma2=s2, B_sigma2=s2, bounds_func=simplex_bounds_func,
    )

    full_constraint = optimize.LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)
    _, best_p_c = conditional_fit_1d_scipy(
        fix_val, 0, 2, N_obs, bounds_list, _two_param_rate_func,
        S_sigma2=s2, B_sigma2=s2, constraints=[full_constraint],
    )

    assert best_p_bf[1] == pytest.approx(best_p_c[1], abs=1e-4)


# --- The case bounds_func cannot express: two simultaneously-free params ---
def test_constraints_handle_two_simultaneously_free_nuisance_params():
    """
    3-param model: p0 + p1 <= 1 must hold, but NEITHER p0 nor p1 is ever the
    scan's fixed test parameter (p2 is fixed instead) -- both are free at the
    same time. bounds_func cannot express this (FIX 2's own documented
    limitation); constraints can.
    """
    bounds_list = [(0.0, 1.0), (0.0, 1.0), (0.0, 5.0)]
    N_obs = np.array([100.0])
    s2 = np.array([0.0])

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_sigma2, B_sigma2):
        # Large N pulls both p0 and p1 toward their upper bounds independently.
        mu = np.array([10.0 * params[0] + 10.0 * params[1] + params[2] + 1.0])
        return mu, S_sigma2

    full_constraint = optimize.LinearConstraint(np.array([[1.0, 1.0, 0.0]]), -np.inf, 1.0)
    cond_nll, best_p = conditional_fit_1d_scipy(
        2.5, 2, 3, N_obs, bounds_list, rate_func,
        S_sigma2=s2, B_sigma2=s2, constraints=[full_constraint],
    )

    assert best_p[2] == 2.5
    assert best_p[0] + best_p[1] <= 1.0 + 1e-6
    assert np.isfinite(cond_nll)


# --- Method selection / backward compatibility ---
def test_method_defaults_to_lbfgsb_without_constraints():
    captured = {}
    original_minimize = optimize.minimize

    def spy(cost, x0, **kwargs):
        captured.update(kwargs)
        return original_minimize(cost, x0, **kwargs)

    bounds_list = [(0.0, 1.0)]
    N_obs = np.array([5.0])
    S_t = np.array([1.0])
    s2 = np.array([0.0])

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_sigma2, B_sigma2):
        return params[0] * S_t + 1.0, S_sigma2

    with patch("pyfc.optimizers.optimize.minimize", side_effect=spy):
        unconditional_fit_scipy(N_obs, 1, bounds_list, rate_func, S_sigma2=s2, B_sigma2=s2)

    assert captured["method"] == "L-BFGS-B"
    assert "constraints" not in captured


def test_method_switches_to_slsqp_with_constraints():
    captured = {}
    original_minimize = optimize.minimize

    def spy(cost, x0, **kwargs):
        captured.update(kwargs)
        return original_minimize(cost, x0, **kwargs)

    bounds_list = [(0.0, 1.0), (0.0, 1.0)]
    N_obs = np.array([10.0])
    s2 = np.array([0.0])
    full_constraint = optimize.LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)

    with patch("pyfc.optimizers.optimize.minimize", side_effect=spy):
        unconditional_fit_scipy(N_obs, 2, bounds_list, _two_param_rate_func,
                                 S_sigma2=s2, B_sigma2=s2, constraints=[full_constraint])

    assert captured["method"] == "SLSQP"
    assert captured["constraints"] == [full_constraint]


def test_scipy_method_override_takes_precedence():
    captured = {}
    original_minimize = optimize.minimize

    def spy(cost, x0, **kwargs):
        captured.update(kwargs)
        return original_minimize(cost, x0, **kwargs)

    bounds_list = [(0.0, 1.0)]
    N_obs = np.array([5.0])
    S_t = np.array([1.0])
    s2 = np.array([0.0])

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_sigma2, B_sigma2):
        return params[0] * S_t + 1.0, S_sigma2

    with patch("pyfc.optimizers.optimize.minimize", side_effect=spy):
        unconditional_fit_scipy(N_obs, 1, bounds_list, rate_func,
                                 S_sigma2=s2, B_sigma2=s2, scipy_method="trust-constr")

    assert captured["method"] == "trust-constr"


# --- _constraints_satisfied helper (used by the UltraNest path) ---
def test_constraints_satisfied_linear():
    c = optimize.LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)
    assert _constraints_satisfied([c], np.array([0.3, 0.3]))
    assert not _constraints_satisfied([c], np.array([0.7, 0.7]))


def test_constraints_satisfied_empty_is_true():
    assert _constraints_satisfied(None, np.array([1.0, 2.0]))
    assert _constraints_satisfied([], np.array([1.0, 2.0]))


# --- Benchmark: confirm no regression on the default (unconstrained) path,
# and characterize constrained overhead. Loose, informational assertions
# only (timing-based CI gates are flaky) -- printed for manual inspection.
def test_benchmark_unconstrained_vs_constrained_overhead():
    bounds_list = [(0.0, 1.0), (0.0, 1.0)]
    N_obs = np.array([100.0])
    s2 = np.array([0.0])
    n_reps = 20

    t0 = time.perf_counter()
    for _ in range(n_reps):
        unconditional_fit_scipy(N_obs, 2, bounds_list, _two_param_rate_func, S_sigma2=s2, B_sigma2=s2)
    t_unconstrained = (time.perf_counter() - t0) / n_reps

    full_constraint = optimize.LinearConstraint(np.array([[1.0, 1.0]]), -np.inf, 1.0)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        unconditional_fit_scipy(N_obs, 2, bounds_list, _two_param_rate_func, S_sigma2=s2, B_sigma2=s2,
                                 constraints=[full_constraint])
    t_constrained = (time.perf_counter() - t0) / n_reps

    print(f"\n[benchmark] unconstrained (L-BFGS-B) mean: {t_unconstrained*1000:.3f} ms/call")
    print(f"[benchmark] constrained (SLSQP) mean:      {t_constrained*1000:.3f} ms/call")

    # Loose sanity bounds -- not a strict performance regression gate.
    assert t_unconstrained < 1.0
    assert t_constrained < 1.0
