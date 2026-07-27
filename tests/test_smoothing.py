"""
Tests for the smooth unphysical-rate penalty (replacing the old flat 1e10
early-return) in `binned.calc_nll` and `unbinned.calc_nll_unbinned`.

These guard two properties:
  1. Backward compatibility: for every bin/event where the model prediction
     is physical (mu_i > 0 / p_events[k] > 0), the NLL must be practically
     identical to the pre-fix formula (matches an independent reimplementation
     to within 1e-12 relative/absolute tolerance).
  2. The fix itself: scanning a parameter through the unphysical region must
     produce a smooth, monotonically-worsening NLL rather than a flat
     constant plateau (the "flat gradient traps L-BFGS-B" failure mode).
"""
import math

import numpy as np
import pytest
from numba import njit

from pyfc.binned import calc_nll
from pyfc.unbinned import calc_nll_unbinned


# --- Binned: pre-fix reference formula (for the physical branch only) ---
def _old_binned_nll_physical(mu, N_obs, S_sumw2_arr, B_sumw2_arr, use_finite_mc):
    """Reimplementation of the pre-fix physical-branch formula for cross-checking."""
    nll = 0.0
    for i in range(len(N_obs)):
        mu_i = mu[i]
        n_obs = float(N_obs[i])
        assert mu_i > 0, "This reference helper only covers the physical branch"
        if use_finite_mc:
            sigma2 = S_sumw2_arr[i]
            if sigma2 > 1e-10:
                alpha = (mu_i**2) / sigma2 + 1.0
                beta = max(mu_i / sigma2, 1e-300)
                lnL = (alpha * math.log(beta)
                       + math.lgamma(n_obs + alpha)
                       - (n_obs + alpha) * math.log(1.0 + beta)
                       - math.lgamma(alpha))
                nll += -2.0 * lnL
            else:
                lnL_poisson = n_obs * math.log(mu_i) - mu_i
                nll += -2.0 * lnL_poisson
        else:
            if n_obs > 0:
                nll += 2.0 * (mu_i - n_obs + n_obs * math.log(n_obs / mu_i))
            else:
                nll += 2.0 * mu_i
    return nll


@njit(fastmath=True, nogil=True)
def _linear_rate_func(params, S_sumw2, B_sumw2):
    """mu = params[0] (single bin), used to sweep a rate through mu=0."""
    mu = np.array([params[0]])
    sigma2 = np.array([0.0])
    return mu, sigma2


def test_binned_calc_nll_identical_for_physical_bins():
    """For mu_i > 0 in all bins, output must match the pre-fix formula exactly."""
    N_obs = np.array([3.0, 0.0, 7.0])
    S_template = np.array([1.0, 1.0, 1.0])
    S_sumw2 = np.zeros_like(S_template)
    B_sumw2 = np.zeros_like(S_template)

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_s2, B_s2):
        mu = params[0] * S_template + 2.0
        return mu, S_s2

    for use_finite_mc in (False, True):
        params = np.array([3.0])
        mu, _ = rate_func(params, S_sumw2, B_sumw2)
        assert np.all(mu > 0)

        got = calc_nll(params, N_obs, S_sumw2, B_sumw2,
                        use_finite_mc, rate_func)
        expected = _old_binned_nll_physical(mu, N_obs, S_sumw2, B_sumw2, use_finite_mc)
        assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_binned_calc_nll_smooth_through_unphysical_region():
    """
    Scanning params[0] from physical (mu>0) into unphysical territory (mu<=0)
    must produce a smooth, monotonically non-decreasing sequence -- not a flat
    plateau at a constant value.
    """
    N_obs = np.array([5.0])
    S_sumw2 = np.zeros(1)
    B_sumw2 = np.zeros(1)

    xs = np.linspace(0.5, -3.0, 25)  # crosses mu=0 partway through
    nlls = []
    for x in xs:
        params = np.array([x])
        nll = calc_nll(params, N_obs, S_sumw2, B_sumw2,
                        False, _linear_rate_func)
        nlls.append(nll)
        assert np.isfinite(nll)

    unphysical_vals = [n for x, n in zip(xs, nlls) if x <= 0]
    # Not a flat plateau: consecutive unphysical values must differ.
    diffs = np.diff(unphysical_vals)
    assert np.all(diffs != 0.0), "Unphysical branch is flat -- gradient trap not fixed"
    # Monotonically worsening (non-decreasing) as mu_i moves further below zero.
    assert np.all(diffs >= 0.0), "Unphysical branch is non-monotonic"
    # Must not just be the old constant 1e10 plateau.
    assert not np.allclose(unphysical_vals, 1e10)


def test_binned_calc_nll_no_early_return_preserves_other_bins():
    """
    A violation in one bin must not discard the NLL contribution already
    accumulated from other (physical) bins -- i.e. no early return.
    """
    N_obs = np.array([5.0, 5.0])
    S_sumw2 = np.zeros(2)
    B_sumw2 = np.zeros(2)

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_s2, B_s2):
        # bin 0 physical, bin 1 unphysical
        mu = np.array([3.0, -1.0])
        return mu, S_s2

    nll = calc_nll(np.array([0.0]), N_obs, S_sumw2, B_sumw2,
                    False, rate_func)
    # physical bin-0 contribution alone
    bin0_only = 2.0 * (3.0 - 5.0 + 5.0 * math.log(5.0 / 3.0))
    assert nll > bin0_only, "Total NLL should include bin-0 contribution plus a penalty, not just the penalty"
    assert nll < 1e9, "Should not fall back to the old flat 1e10-scale penalty"


# --- Unbinned tests ---
def test_unbinned_calc_nll_identical_for_physical_events():
    def rate_func(params, probs):
        s_probs, b_probs = probs[0], probs[1]
        p_events = params[0] * s_probs + params[1] * b_probs
        return params[0] + params[1], p_events

    s_probs = np.array([0.5, 0.2, 0.9])
    b_probs = np.array([0.1, 0.3, 0.05])
    params = np.array([2.0, 1.0])

    expected_total, p_events = rate_func(params, [s_probs, b_probs])
    assert np.all(p_events > 0)

    got = calc_nll_unbinned(params, len(s_probs), [s_probs, b_probs], rate_func)
    expected = expected_total - np.sum(np.log(p_events))
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_unbinned_calc_nll_smooth_through_unphysical_region():
    def rate_func(params, probs):
        # Single event whose density is exactly params[0]
        return 5.0, np.array([params[0]])

    xs = np.linspace(0.5, -3.0, 25)
    nlls = []
    for x in xs:
        nll = calc_nll_unbinned(np.array([x]), 1, [np.array([1.0]), np.array([1.0])], rate_func)
        nlls.append(nll)
        assert np.isfinite(nll)

    unphysical_vals = [n for x, n in zip(xs, nlls) if x <= 0]
    diffs = np.diff(unphysical_vals)
    assert np.all(diffs != 0.0), "Unphysical branch is flat -- gradient trap not fixed"
    assert np.all(diffs >= 0.0), "Unphysical branch is non-monotonic"
    assert not np.allclose(unphysical_vals, 1e10)


def test_unbinned_calc_nll_elementwise_not_any_collapsed():
    """
    Two independent unphysical events should each contribute according to how
    far below the floor they are, not be collapsed to a single flat constant.
    """
    def rate_func(params, probs):
        return 5.0, np.array([params[0], params[1]])

    nll_a = calc_nll_unbinned(np.array([-0.1, -0.1]), 2, [np.array([1.0, 1.0]), np.array([1.0, 1.0])], rate_func)
    nll_b = calc_nll_unbinned(np.array([-0.1, -2.0]), 2, [np.array([1.0, 1.0]), np.array([1.0, 1.0])], rate_func)
    assert nll_a != nll_b
    assert nll_b > nll_a  # second event further into unphysical territory -> larger penalty
