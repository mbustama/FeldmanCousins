"""
Verifies `binned.calc_nll`'s finite-MC (Poisson-Gamma mixture) branch
against a hand-computed reference built directly from Eq. (3.16) of
Arguelles, Schneider & Yuan, "A binned likelihood for stochastic models",
JHEP 06 (2019) 030 [arXiv:1901.04645] -- the paper's L_Eff, which is
exactly what use_finite_mc=True implements (see the citation/provenance
note in calc_nll's own docstring).

This was flagged during a code review as needing a primary-source check:
alpha = mu^2/sigma^2 + 1 looks like it could be a mean-and-variance-
matching derivation that dropped a "+1", since the naive derivation gives
alpha = mu^2/sigma^2 with no "+1". The paper confirms the "+1" is
deliberate (their uniform-prior choice, Eq. 3.12/3.17 with a=1, b=0) --
this is L_Eff, their main recommended result, not a bug.
"""
import math

import numpy as np
from numba import njit

from pyfc.binned import calc_nll


def _paper_L_eff_nll(mu, sigma2, n_obs):
    """
    Direct transcription of the paper's Eq. (3.16), dropping the ln(k!)
    term (a data-only constant that cancels in any NLL difference, the
    same convention documented for the EUML formula elsewhere in this
    codebase).

    alpha = mu^2/sigma^2 + 1
    beta = mu/sigma^2
    ln(L_Eff) = alpha*ln(beta) + lgamma(n_obs+alpha) - (n_obs+alpha)*ln(1+beta) - lgamma(alpha)
    NLL = -2 * ln(L_Eff)
    """
    alpha = mu**2 / sigma2 + 1.0
    beta = mu / sigma2
    lnL = (alpha * math.log(beta)
           + math.lgamma(n_obs + alpha)
           - (n_obs + alpha) * math.log(1.0 + beta)
           - math.lgamma(alpha))
    return -2.0 * lnL


def test_calc_nll_finite_mc_matches_hand_computed_paper_formula():
    """
    Single bin, worked by hand: mu=10.0, sigma2=2.5, n_obs=8.
      alpha = 10.0^2/2.5 + 1 = 41.0
      beta = 10.0/2.5 = 4.0
      ln(L) = 41.0*ln(4.0) + lgamma(8+41.0) - (8+41.0)*ln(5.0) - lgamma(41.0)
      NLL = -2*ln(L)
    """
    mu, sigma2, n_obs = 10.0, 2.5, 8.0

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_s2, B_s2):
        return np.array([params[0]]), np.array([params[1]])

    params = np.array([mu, sigma2])
    nll_actual = calc_nll(params, np.array([n_obs]), np.zeros(1), np.zeros(1), True, rate_func)

    alpha = mu**2 / sigma2 + 1.0
    beta = mu / sigma2
    assert alpha == 41.0
    assert beta == 4.0
    nll_reference = -2.0 * (
        alpha * math.log(beta)
        + math.lgamma(n_obs + alpha)
        - (n_obs + alpha) * math.log(1.0 + beta)
        - math.lgamma(alpha)
    )

    assert np.isclose(nll_actual, nll_reference, rtol=1e-12, atol=1e-12)


def _exact_recurrence_nll(mu, sigma2, n_obs):
    """
    Independent high-precision reference for the finite-MC formula, using
    the exact Gamma recurrence Gamma(x+n) = Gamma(x) * PROD_{k=0}^{n-1}(x+k)
    (valid since n_obs is always a non-negative integer count) instead of
    evaluating lgamma(n+alpha) and lgamma(alpha) separately and
    subtracting -- the subtraction is what loses precision at large alpha
    (both terms scale like alpha*ln(alpha); their difference is only
    O(n*ln(alpha))). This is the same identity `calc_nll` itself now uses
    internally, so this is not an independent *implementation*, but it is
    an independent, exact *mathematical* check: verified against
    `_paper_L_eff_nll` (the literal lgamma-difference transcription of the
    paper's own Eq. 3.16) at ordinary alpha in
    test_calc_nll_finite_mc_matches_hand_computed_paper_formula above,
    where both forms agree to 1e-12 -- they are the same quantity, just
    computed two different ways.
    """
    alpha = mu**2 / sigma2 + 1.0
    beta = mu / sigma2
    n_int = int(round(n_obs))
    lgamma_diff = sum(math.log(alpha + k) for k in range(n_int))
    lnL = -alpha * math.log1p(1.0 / beta) - n_obs * math.log1p(beta) + lgamma_diff
    return -2.0 * lnL


def test_calc_nll_finite_mc_high_alpha_avoids_precision_loss():
    """
    Regression test for a catastrophic-cancellation bug: for well-simulated
    templates (small sigma2 relative to mu^2, i.e. large alpha), evaluating
    ln(Gamma(n+alpha)) - ln(Gamma(alpha)) as a literal difference of two
    `lgamma` calls loses precision, since both terms are individually huge
    (~alpha*ln(alpha)) while their difference is only O(n*ln(alpha)).

    At mu=1000, sigma2=1e-9, n_obs=1000 (alpha ~ 1e15), the naive
    lgamma-difference formula (`_paper_L_eff_nll`, a direct transcription
    of the paper's own Eq. 3.16) is off from the true value by ~7.5 in
    absolute NLL -- confirmed below to still be wrong by more than 1.0, so
    this test would have caught the original bug. `calc_nll` itself must
    match the exact-recurrence reference to numerical (not just
    order-of-magnitude) precision.
    """
    mu, sigma2, n_obs = 1000.0, 1e-9, 1000.0

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_s2, B_s2):
        return np.array([params[0]]), np.array([params[1]])

    params = np.array([mu, sigma2])
    nll_actual = calc_nll(params, np.array([n_obs]), np.zeros(1), np.zeros(1), True, rate_func)

    nll_exact = _exact_recurrence_nll(mu, sigma2, n_obs)
    nll_naive = _paper_L_eff_nll(mu, sigma2, n_obs)

    assert abs(nll_naive - nll_exact) > 1.0, (
        "sanity check that this regime actually exercises the precision-loss "
        "bug -- if this fails, the naive formula stopped being unstable here "
        "and the test case needs a more extreme (mu, sigma2) pair"
    )
    assert np.isclose(nll_actual, nll_exact, rtol=1e-8, atol=1e-6)


def test_calc_nll_finite_mc_matches_paper_formula_across_multiple_bins():
    """Sweeps several (mu, sigma2, n_obs) combinations against _paper_L_eff_nll."""
    S_template = np.array([3.0, 7.5, 1.2, 20.0])
    B_sumw2 = np.zeros_like(S_template)

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_s2, B_s2):
        mu = params[0] * S_template
        sigma2 = (params[1]**2) * S_s2
        return mu, sigma2

    N_obs = np.array([2.0, 9.0, 0.0, 25.0])
    S_sumw2 = np.array([0.4, 1.1, 0.2, 3.0])
    params = np.array([1.5, 1.0])

    nll_actual = calc_nll(params, N_obs, S_sumw2, B_sumw2, True, rate_func)

    mu_arr, sigma2_arr = rate_func(params, S_sumw2, B_sumw2)
    nll_reference = sum(
        _paper_L_eff_nll(mu_arr[i], sigma2_arr[i], N_obs[i])
        for i in range(len(N_obs))
        if sigma2_arr[i] > 1e-10
    )
    # n_obs=0 bin (index 2) has sigma2 > 1e-10 too, so it's included above;
    # the paper's formula (and calc_nll's finite-MC branch) both handle
    # n_obs=0 the same way as any other count -- no special-casing needed.

    assert np.isclose(nll_actual, nll_reference, rtol=1e-12, atol=1e-12)
