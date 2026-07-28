"""
Tests for making `adaptive_toys` and `toy_batch_size` functional in
`toys.generate_and_fit_toys_python` (the "scipy"/"ultranest"/"hybrid"
strategy toy-generation path).

`strategy="grid"`'s toy generators (`binned.py`/`unbinned.py`) are
unaffected by design -- their `numba` `@njit(parallel=True)`/sequential
loops are not touched by this fix; see their own docstrings and
`compute_fc_intervals`'s docstring for why.

Validation philosophy (per the risk this fix was flagged with): a wrong
adaptive-stopping implementation would silently bias confidence intervals,
which is worse than the previous do-nothing state. So these tests check
the *verdict* (accept/reject) adaptive_toys reaches, not just that it runs
without crashing -- both at the statistical-primitive level
(`_wilson_score_interval`/`_adaptive_stop_decision`) and end-to-end
against a non-adaptive reference.
"""
import numpy as np
import pytest
from numba import njit

from pyfc.orchestrator import compute_fc_intervals
from pyfc.toys import (
    ADAPTIVE_MIN_TOYS,
    _adaptive_stop_decision,
    _wilson_score_interval,
    generate_and_fit_toys_python,
)


# --- Statistical primitives ---

def test_wilson_score_interval_extreme_proportions():
    lo, hi = _wilson_score_interval(100, 100)
    assert lo > 0.85 and hi == pytest.approx(1.0)

    lo, hi = _wilson_score_interval(0, 100)
    assert lo == 0.0 and hi < 0.15


def test_wilson_score_interval_straddles_point_estimate():
    lo, hi = _wilson_score_interval(50, 100)
    assert lo < 0.5 < hi


def test_adaptive_stop_decision_never_before_min_toys():
    """Never stop before ADAPTIVE_MIN_TOYS, no matter how extreme the proportion."""
    assert _adaptive_stop_decision(0, ADAPTIVE_MIN_TOYS - 1, alpha=0.5) is False
    assert _adaptive_stop_decision(ADAPTIVE_MIN_TOYS - 1, ADAPTIVE_MIN_TOYS - 1, alpha=0.5) is False


def test_adaptive_stop_decision_stops_when_verdict_is_extreme():
    # p_hat=0.85, far from alpha=0.1 -- CI should clear alpha comfortably.
    assert _adaptive_stop_decision(85, 100, alpha=0.1) is True
    # p_hat=0.0, far below alpha=0.5.
    assert _adaptive_stop_decision(0, 100, alpha=0.5) is True


def test_adaptive_stop_decision_does_not_stop_near_boundary():
    # p_hat=0.12, close to alpha=0.1 -- CI should still straddle alpha.
    assert _adaptive_stop_decision(12, 100, alpha=0.1) is False
    # p_hat exactly at alpha.
    assert _adaptive_stop_decision(10, 100, alpha=0.1) is False


# --- toy_batch_size: pure dispatch-grouping change, zero effect on results ---

@njit(fastmath=True, nogil=True)
def _single_bin_rate_func(params, S_s2, B_s2):
    return np.array([params[0]]), np.array([0.0])


def test_toy_batch_size_alone_does_not_change_results():
    """
    For binned toys, np.random.poisson(mu_true, size=(n_toys,)+shape) is
    called once up front regardless of batching -- toy_batch_size only
    changes how the (already-generated) toys are dispatched to the
    executor, never what they are. With the same seed, results must be
    bit-for-bit identical regardless of toy_batch_size.
    """
    true_params = np.array([10.0])
    bounds_list = [(0.1, 30.0)]

    np.random.seed(123)
    t_unbatched = generate_and_fit_toys_python(
        true_params, 1, "1d", 0, None, None, 10.0, None, bounds_list, 200, "scipy",
        num_cores=1, verbose=0, likelihood_type="binned",
        S_sumw2=np.zeros(1), B_sumw2=np.zeros(1), use_finite_mc=False,
        compute_rates_func=_single_bin_rate_func, toy_batch_size=None, adaptive_toys=False,
    )
    np.random.seed(123)
    t_batched = generate_and_fit_toys_python(
        true_params, 1, "1d", 0, None, None, 10.0, None, bounds_list, 200, "scipy",
        num_cores=1, verbose=0, likelihood_type="binned",
        S_sumw2=np.zeros(1), B_sumw2=np.zeros(1), use_finite_mc=False,
        compute_rates_func=_single_bin_rate_func, toy_batch_size=37, adaptive_toys=False,
    )

    assert len(t_unbatched) == 200
    assert len(t_batched) == 200
    np.testing.assert_array_equal(t_unbatched, t_batched)


# --- adaptive_toys: early-stopping behavior and verdict correctness ---

def test_adaptive_toys_stops_early_for_extreme_points():
    """Well-inside/well-outside points should stop at ADAPTIVE_MIN_TOYS, not run all 300."""
    true_params = np.array([10.0])
    bounds_list = [(0.1, 30.0)]

    for t_data in (0.0, 1000.0):
        np.random.seed(7)
        t_stats = generate_and_fit_toys_python(
            true_params, 1, "1d", 0, None, None, 10.0, None, bounds_list, 300, "scipy",
            num_cores=2, verbose=0, likelihood_type="binned",
            S_sumw2=np.zeros(1), B_sumw2=np.zeros(1), use_finite_mc=False,
            compute_rates_func=_single_bin_rate_func, toy_batch_size=20,
            adaptive_toys=True, t_data=t_data, alpha=0.1,
        )
        assert len(t_stats) == ADAPTIVE_MIN_TOYS, (
            f"t_data={t_data}: expected early stop at {ADAPTIVE_MIN_TOYS}, got {len(t_stats)}"
        )


def test_adaptive_toys_runs_full_n_toys_near_boundary():
    """A point near the true critical value must not stop early."""
    true_params = np.array([10.0])
    bounds_list = [(0.1, 30.0)]

    np.random.seed(7)
    reference = generate_and_fit_toys_python(
        true_params, 1, "1d", 0, None, None, 10.0, None, bounds_list, 300, "scipy",
        num_cores=2, verbose=0, likelihood_type="binned",
        S_sumw2=np.zeros(1), B_sumw2=np.zeros(1), use_finite_mc=False,
        compute_rates_func=_single_bin_rate_func, toy_batch_size=None, adaptive_toys=False,
    )
    t_critical_approx = np.sort(reference)[int(0.90 * 300)]

    np.random.seed(7)
    t_stats = generate_and_fit_toys_python(
        true_params, 1, "1d", 0, None, None, 10.0, None, bounds_list, 300, "scipy",
        num_cores=2, verbose=0, likelihood_type="binned",
        S_sumw2=np.zeros(1), B_sumw2=np.zeros(1), use_finite_mc=False,
        compute_rates_func=_single_bin_rate_func, toy_batch_size=20,
        adaptive_toys=True, t_data=t_critical_approx, alpha=0.1,
    )
    assert len(t_stats) == 300


def test_adaptive_toys_verdict_matches_non_adaptive_reference_across_many_trials():
    """
    The real test of "did the stopping rule introduce bias": across many
    independent trials, the accept/reject verdict for well-inside and
    well-outside points under adaptive_toys must match a full-n_toys
    non-adaptive reference every time (these are far enough from the
    boundary that a correct stopping rule should never flip the verdict).
    """
    true_params = np.array([10.0])
    bounds_list = [(0.1, 30.0)]
    n_toys = 300
    alpha = 0.1
    n_trials = 20

    mismatches = 0
    for trial in range(n_trials):
        seed = 5000 + trial
        np.random.seed(seed)
        reference = generate_and_fit_toys_python(
            true_params, 1, "1d", 0, None, None, 10.0, None, bounds_list, n_toys, "scipy",
            num_cores=2, verbose=0, likelihood_type="binned",
            S_sumw2=np.zeros(1), B_sumw2=np.zeros(1), use_finite_mc=False,
            compute_rates_func=_single_bin_rate_func, toy_batch_size=None, adaptive_toys=False,
        )
        t_crit_ref = np.sort(reference)[int((1 - alpha) * n_toys)]

        for t_data in (0.0, 1000.0):
            accepted_ref = t_data <= t_crit_ref

            np.random.seed(seed)
            t_adapt = generate_and_fit_toys_python(
                true_params, 1, "1d", 0, None, None, 10.0, None, bounds_list, n_toys, "scipy",
                num_cores=2, verbose=0, likelihood_type="binned",
                S_sumw2=np.zeros(1), B_sumw2=np.zeros(1), use_finite_mc=False,
                compute_rates_func=_single_bin_rate_func, toy_batch_size=20,
                adaptive_toys=True, t_data=t_data, alpha=alpha,
            )
            n = len(t_adapt)
            t_crit_adapt = np.sort(t_adapt)[min(int((1 - alpha) * n), n - 1)]
            accepted_adapt = t_data <= t_crit_adapt

            if accepted_adapt != accepted_ref:
                mismatches += 1

    assert mismatches == 0, f"{mismatches}/{n_trials * 2} verdict mismatches between adaptive and reference"


# --- End-to-end: full compute_fc_intervals run, adaptive vs non-adaptive ---

S_TEMPLATE = np.array([0.1, 0.5, 2.0, 5.0])
B_TEMPLATE = np.array([15.0, 5.0, 1.0, 0.1])


@njit(fastmath=True, nogil=True)
def _compute_rates(params, S_sumw2, B_sumw2):
    mu = params[0] * S_TEMPLATE + params[1] + params[2] * B_TEMPLATE
    sigma2 = (params[0] ** 2) * S_sumw2 + (params[2] ** 2) * B_sumw2
    return mu, sigma2


def test_compute_fc_intervals_adaptive_matches_non_adaptive_end_to_end(tmp_path):
    """
    Full compute_fc_intervals run (not just the toy-generation function in
    isolation): adaptive_toys=True with n_toys=300 must reach the same
    1D accepted verdicts as adaptive_toys=False with the same n_toys cap,
    for a real (if small) grid scan.
    """
    np.random.seed(99)
    n_data = np.random.poisson(1.0 * S_TEMPLATE + 1.0 + 1.0 * B_TEMPLATE)
    grids = [np.linspace(0.5, 2.0, 6), np.linspace(0.5, 2.0, 6), np.linspace(0.5, 1.5, 6)]

    np.random.seed(42)
    results_ref, _ = compute_fc_intervals(
        data=n_data, grids=grids, compute_rates_func=_compute_rates,
        cl=[0.90], n_toys=300, strategy="scipy", num_cores=2, verbose=0,
        adaptive_toys=False, toy_batch_size=20,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        S_sumw2=S_TEMPLATE.copy(), B_sumw2=B_TEMPLATE.copy(),
        compute_1D_intervals=True, compute_2D_intervals=False,
        save_directory=str(tmp_path / "adaptive_off"),
    )

    np.random.seed(42)
    results_adapt, _ = compute_fc_intervals(
        data=n_data, grids=grids, compute_rates_func=_compute_rates,
        cl=[0.90], n_toys=300, strategy="scipy", num_cores=2, verbose=0,
        adaptive_toys=True, toy_batch_size=20,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        S_sumw2=S_TEMPLATE.copy(), B_sumw2=B_TEMPLATE.copy(),
        compute_1D_intervals=True, compute_2D_intervals=False,
        save_directory=str(tmp_path / "adaptive_on"),
    )

    for p_idx in range(3):
        acc_ref = results_ref[f"1d_accepted_p{p_idx+1}"][0.90]
        acc_adapt = results_adapt[f"1d_accepted_p{p_idx+1}"][0.90]
        mismatches = np.sum(acc_ref != acc_adapt)
        assert mismatches == 0, (
            f"param{p_idx+1}: {mismatches}/{len(acc_ref)} accept/reject "
            f"verdicts differ between adaptive_toys=True and False"
        )
