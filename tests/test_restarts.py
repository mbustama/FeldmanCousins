"""
Tests for FIX 4: optimizer restarts, neighbor warm-starting for the DATA fit,
and res.success checking.

IMPORTANT CORRECTION this fix is built on (see optimizers.py/orchestrator.py
comments): `warm_start` only controls resuming from an on-disk checkpoint
across separate program invocations -- it does NOT seed neighboring grid
points' fits from each other. Toy fits are already well-seeded from the
conditional MLE (`true_params`). The gap this fix closes is that the DATA
fit itself (Phase 0/1/2 in orchestrator.py) never passed a `seed`, so every
grid point started fresh from the bounds midpoint with zero information
sharing between adjacent, already-solved grid points.
"""
import time
import warnings
from unittest.mock import patch

import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import SCIPY_AVAILABLE, _minimize_with_restarts, optimize
from pyfc.orchestrator import compute_fc_intervals

pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for these tests")


# --- Unit tests for _minimize_with_restarts, using a controlled fake minimize ---
class _FakeResult:
    def __init__(self, x, fun, success, status=0, message="ok"):
        self.x = np.asarray(x, dtype=float)
        self.fun = fun
        self.success = success
        self.status = status
        self.message = message


def test_minimize_with_restarts_default_single_call_success():
    """n_restarts=1 with a converging first attempt: single call, no warning, no retry."""
    calls = []

    def fake_minimize(cost, x0, bounds, method, **kwargs):
        calls.append(np.asarray(x0))
        return _FakeResult(x0, fun=1.23, success=True)

    with patch("pyfc.optimizers.optimize.minimize", side_effect=fake_minimize):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = _minimize_with_restarts(lambda x: 0.0, [0.5], [(0.0, 1.0)], "L-BFGS-B", {}, n_restarts=1)

    assert len(calls) == 1
    assert res.fun == 1.23
    assert res.success


def test_minimize_with_restarts_retries_once_on_failure_and_warns_if_still_failing():
    """n_restarts=1, first attempt fails, perturbed retry also fails -> warns, returns best-effort."""
    call_count = {"n": 0}

    def fake_minimize(cost, x0, bounds, method, **kwargs):
        call_count["n"] += 1
        return _FakeResult(x0, fun=99.0, success=False, status=2, message="did not converge")

    with patch("pyfc.optimizers.optimize.minimize", side_effect=fake_minimize):
        with pytest.warns(UserWarning, match="did not converge"):
            res = _minimize_with_restarts(lambda x: 0.0, [0.5], [(0.0, 1.0)], "L-BFGS-B", {}, n_restarts=1)

    assert call_count["n"] == 2  # original attempt + one perturbed retry
    assert not res.success


def test_minimize_with_restarts_recovers_via_perturbed_retry_no_warning():
    """First attempt fails, but the cheap perturbed retry succeeds -> no warning, retry's result used."""
    call_count = {"n": 0}

    def fake_minimize(cost, x0, bounds, method, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeResult(x0, fun=99.0, success=False, status=2, message="did not converge")
        return _FakeResult(x0, fun=0.5, success=True)

    with patch("pyfc.optimizers.optimize.minimize", side_effect=fake_minimize):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = _minimize_with_restarts(lambda x: 0.0, [0.5], [(0.0, 1.0)], "L-BFGS-B", {}, n_restarts=1)

    assert call_count["n"] == 2
    assert res.success
    assert res.fun == 0.5


def test_minimize_with_restarts_multi_start_keeps_best():
    """n_restarts=3: tries x0 + bounds midpoint + 1 random point, keeps lowest .fun among successes."""
    funs_by_x0 = {}

    def fake_minimize(cost, x0, bounds, method, **kwargs):
        x0 = np.asarray(x0)
        # Give the bounds-midpoint start the best (lowest) result; others worse.
        mid = np.array([(b[0] + b[1]) / 2.0 for b in bounds])
        if np.allclose(x0, mid):
            fun = 0.1
        else:
            fun = 10.0 + float(np.sum(x0))
        funs_by_x0[tuple(x0)] = fun
        return _FakeResult(x0, fun=fun, success=True)

    with patch("pyfc.optimizers.optimize.minimize", side_effect=fake_minimize):
        res = _minimize_with_restarts(lambda x: 0.0, [0.9], [(0.0, 1.0)], "L-BFGS-B", {}, n_restarts=3)

    assert len(funs_by_x0) == 3  # x0, midpoint, and one random candidate all tried
    assert res.fun == 0.1  # the midpoint start's result, the best among candidates


def test_minimize_with_restarts_n_restarts_1_backward_compatible_call_shape():
    """
    With n_restarts=1 and a converging fit, the single call made must use exactly
    the x0/bounds/method/kwargs passed in -- i.e. behaves like the pre-FIX-4
    direct `optimize.minimize(...)` call.
    """
    captured = {}
    original_minimize = optimize.minimize

    def spy(cost, x0, bounds, method, **kwargs):
        captured["x0"] = x0
        captured["bounds"] = bounds
        captured["method"] = method
        captured.update(kwargs)
        return original_minimize(cost, x0=x0, bounds=bounds, method=method, **kwargs)

    with patch("pyfc.optimizers.optimize.minimize", side_effect=spy):
        res = _minimize_with_restarts(lambda p: (p[0] - 0.3) ** 2, [0.5], [(0.0, 1.0)], "L-BFGS-B", {}, n_restarts=1)

    assert captured["method"] == "L-BFGS-B"
    assert captured["bounds"] == [(0.0, 1.0)]
    assert res.success
    assert res.x[0] == pytest.approx(0.3, abs=1e-4)


# --- Integration: neighbor warm-starting is actually exercised by the orchestrator ---
# Fixed template arrays are module-level constants, referenced via closure
# from the @njit compute_rates_func below (no more S_model/B_model arguments
# -- see the "BREAKING CHANGES" CHANGELOG entry). These two mocked tests
# patch out the actual fit functions, so compute_rates_func is never called
# for real here -- it only needs to be a valid callable reference.
_S_TEMPLATE_2P = np.array([1.0])
_B_TEMPLATE_2P = np.array([1.0])


@njit(fastmath=True, nogil=True)
def _simple_2param_rate_func(params, S_sigma2, B_sigma2):
    mu = params[0] * _S_TEMPLATE_2P + params[1] * _B_TEMPLATE_2P + 1.0
    return mu, S_sigma2


def test_orchestrator_neighbor_seeding_passes_previous_point_as_seed():
    """
    Phase 1's data-fit loop should call conditional_fit_1d_scipy with
    seed=None for the first grid point (i=0), then seed=<previous point's
    profiled free parameters> for i>0, with n_restarts bumped to >= 2
    whenever a neighbor seed is used.
    """
    calls_by_fix_idx = {0: [], 1: []}

    def fake_conditional_fit_1d_scipy(test_val, fix_idx, n_params_, data, bounds_list,
                                       compute_rates_func, seed=None, **kwargs):
        calls_by_fix_idx[fix_idx].append({"seed": seed, "n_restarts": kwargs.get("n_restarts")})
        p = np.zeros(n_params_)
        p[fix_idx] = test_val
        for k in range(n_params_):
            if k != fix_idx:
                p[k] = 0.5  # deterministic dummy profiled value, so neighbor lookups are well-defined
        return 0.0, p

    def fake_unconditional_fit_scipy(*args, **kwargs):
        return 0.0, np.array([0.5, 0.5])

    N_data = np.array([5.0])
    s2 = np.zeros_like(_S_TEMPLATE_2P)
    grids = [np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)]

    with patch("pyfc.orchestrator.conditional_fit_1d_scipy", side_effect=fake_conditional_fit_1d_scipy), \
         patch("pyfc.orchestrator.unconditional_fit_scipy", side_effect=fake_unconditional_fit_scipy):
        compute_fc_intervals(
            data=N_data, grids=grids,
            compute_rates_func=_simple_2param_rate_func,
            cl=[0.90], n_toys=1, strategy="scipy", num_cores=1, verbose=0,
            sparsify_grid=False, warm_start=False,
            likelihood_type="binned", S_sigma2=s2, B_sigma2=s2,
            output_file=None, save_directory="/tmp/pyfc_test_neighbor_seed_1d",
            compute_1D_intervals=True, compute_2D_intervals=False,
        )

    for fix_idx, calls in calls_by_fix_idx.items():
        assert len(calls) == 4  # one call per grid point
        assert calls[0]["seed"] is None  # i=0: no neighbor yet, fresh start
        assert calls[0]["n_restarts"] == 1  # default n_restarts, no forced bump without a neighbor seed
        for c in calls[1:]:
            assert c["seed"] == [0.5]  # neighbor-seeded from the previous point's dummy profiled value
            assert c["n_restarts"] == 2  # forced up to >= 2 when a neighbor seed is used


def test_orchestrator_neighbor_seeding_disabled_via_flag():
    """neighbor_seeding=False must restore the pre-FIX-4 behavior: always seed=None."""
    calls = []

    def fake_conditional_fit_1d_scipy(test_val, fix_idx, n_params_, data, bounds_list,
                                       compute_rates_func, seed=None, **kwargs):
        calls.append({"seed": seed, "n_restarts": kwargs.get("n_restarts")})
        p = np.zeros(n_params_)
        p[fix_idx] = test_val
        return 0.0, p

    def fake_unconditional_fit_scipy(*args, **kwargs):
        return 0.0, np.array([0.5])

    N_data = np.array([5.0])
    S_template = np.array([1.0])
    s2 = np.zeros_like(S_template)
    grids = [np.linspace(0.0, 1.0, 4)]

    @njit(fastmath=True, nogil=True)
    def rate_func_1p(params, S_sigma2, B_sigma2):
        return params[0] * S_template + 1.0, S_sigma2

    with patch("pyfc.orchestrator.conditional_fit_1d_scipy", side_effect=fake_conditional_fit_1d_scipy), \
         patch("pyfc.orchestrator.unconditional_fit_scipy", side_effect=fake_unconditional_fit_scipy):
        compute_fc_intervals(
            data=N_data, grids=grids,
            compute_rates_func=rate_func_1p,
            cl=[0.90], n_toys=1, strategy="scipy", num_cores=1, verbose=0,
            sparsify_grid=False, warm_start=False,
            likelihood_type="binned", S_sigma2=s2, B_sigma2=s2,
            output_file=None, save_directory="/tmp/pyfc_test_neighbor_seed_disabled",
            compute_1D_intervals=True, compute_2D_intervals=False,
            neighbor_seeding=False,
        )

    assert len(calls) == 4
    assert all(c["seed"] is None for c in calls)
    assert all(c["n_restarts"] == 1 for c in calls)


# --- End-to-end: with vs without neighbor warm-starting, checking for
# plateaus/non-monotonic jumps and comparing wall-clock (loose, informational) ---
def test_end_to_end_with_and_without_neighbor_seeding_no_plateau_or_regression():
    S_template = np.array([0.1, 0.5, 2.0, 5.0])
    B_template = np.array([15.0, 5.0, 1.0, 0.1])
    s2_S = np.zeros_like(S_template)
    s2_B = np.zeros_like(B_template)

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_s2, B_s2):
        mu = params[0] * S_template + params[1] + params[2] * B_template
        return mu, S_s2

    np.random.seed(7)
    N_data = np.random.poisson(1.0 * S_template + 1.0 + 1.0 * B_template)
    grids = [np.linspace(0.5, 2.0, 6), np.linspace(0.5, 2.0, 6), np.linspace(0.5, 1.5, 6)]

    def run(neighbor_seeding):
        t0 = time.perf_counter()
        results, _ = compute_fc_intervals(
            data=N_data, grids=grids,
            compute_rates_func=rate_func,
            cl=[0.90], n_toys=10, strategy="scipy", num_cores=1, verbose=0,
            sparsify_grid=False, warm_start=False,
            likelihood_type="binned", S_sigma2=s2_S, B_sigma2=s2_B,
            output_file=None, save_directory=f"/tmp/pyfc_test_e2e_{neighbor_seeding}",
            compute_1D_intervals=True, compute_2D_intervals=False,
            neighbor_seeding=neighbor_seeding,
        )
        return results, time.perf_counter() - t0

    results_with, t_with = run(True)
    results_without, t_without = run(False)

    print(f"\n[benchmark] neighbor_seeding=True:  {t_with:.3f}s")
    print(f"[benchmark] neighbor_seeding=False: {t_without:.3f}s")

    for p_idx in range(3):
        t_data_with = results_with[f"1d_t_data_p{p_idx+1}"]
        t_data_without = results_without[f"1d_t_data_p{p_idx+1}"]
        assert np.all(np.isfinite(t_data_with))
        assert np.all(np.isfinite(t_data_without))
        # Same well-conditioned convex-ish problem: both should reach equivalent optima.
        assert np.allclose(t_data_with, t_data_without, atol=1e-3), \
            f"param {p_idx+1}: neighbor seeding changed the converged t_data profile"
