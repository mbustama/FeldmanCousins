"""
Tests that actually run the three UltraNest fit functions.

WHY THIS FILE EXISTS: with `ultranest` installed, `optimizers.py` still reported 63%, and
reading the cold lines showed why. Nothing ran the UltraNest fits. `test_ultranest_retry.py`
exercises `_run_ultranest_with_retry` against a mocked sampler, and `test_constraints.py`
calls the two conditional fits only in the zero-free-parameters case, which returns before
the sampler is ever invoked. So `unconditional_fit_ultranest`'s body was entirely unrun,
and the conditional pair were entered but never sampled -- roughly the last third of the
module, shipped and reachable through `strategy="ultranest"`/`"hybrid"`, executed by
nothing.

Nested sampling is stochastic, so these assert what must hold for any correct sampler
rather than exact values. Tolerances are set from measurement, not guessed: over repeated
runs on the Asimov dataset below the MLE lands within 0.004 of the truth, and the test
statistic at the true value comes out around -1e-4 -- slightly negative, from sampler
noise, which is why the profile-likelihood assertions carry a small negative tolerance
rather than requiring exactly >= 0. (`compute_fc_intervals` clamps the same quantity with
`max(0.0, ...)` for the same reason.)

The model is deliberately non-degenerate: bin 1 constrains p0, bin 2 constrains p1, bin 3
mixes them. An earlier draft used two bins whose equations reduced to the same constraint,
so a whole family of parameter pairs fit the data exactly and "recovers the truth" was not
a meaningful claim.
"""

import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import (
    SCIPY_AVAILABLE,
    ULTRANEST_AVAILABLE,
    conditional_fit_1d_scipy,
    conditional_fit_1d_ultranest,
    conditional_fit_2d_scipy,
    conditional_fit_2d_ultranest,
    unconditional_fit_scipy,
    unconditional_fit_ultranest,
)

pytestmark = pytest.mark.skipif(
    not ULTRANEST_AVAILABLE, reason="UltraNest is required for these tests"
)

TRUE_PARAMS = np.array([1.0, 1.0])
BOUNDS = [(0.3, 2.5), (0.3, 2.5)]


@njit(fastmath=True, nogil=True)
def _rates(params, S_sumw2, B_sumw2):
    """Three bins, non-degenerate in the two parameters."""
    mu = np.array([
        20.0 * params[0] + 1.0,
        15.0 * params[1] + 1.0,
        6.0 * params[0] + 9.0 * params[1] + 1.0,
    ])
    return mu, S_sumw2


# Asimov dataset: the observed counts ARE the expectation at TRUE_PARAMS, so the maximum
# likelihood estimate must land on TRUE_PARAMS and the NLL there must be ~0. That makes the
# truth an oracle rather than something read back out of the fit.
ASIMOV = np.array([21.0, 16.0, 16.0])
SUMW2 = np.zeros(3)
_KW = dict(verbose=0, S_sumw2=SUMW2, B_sumw2=SUMW2)


def test_unconditional_fit_ultranest_recovers_the_asimov_truth():
    """
    The unconditional fit must find the parameters that generated the data.

    This is the whole contract of the function and it had never been checked: the fit was
    only ever reached through `compute_fc_intervals`, which no test drove with
    `strategy="ultranest"`.
    """
    nll, best = unconditional_fit_ultranest(ASIMOV, 2, BOUNDS, _rates, **_KW)
    best = np.asarray(best, dtype=float)

    assert best.shape == (2,)
    np.testing.assert_allclose(best, TRUE_PARAMS, atol=0.05)
    assert nll == pytest.approx(0.0, abs=1e-2), (
        f"NLL at the Asimov best fit should be ~0, got {nll}"
    )


def test_conditional_fit_1d_ultranest_pins_the_tested_parameter():
    """The parameter under test is held exactly; the other is profiled."""
    nll, params = conditional_fit_1d_ultranest(1.7, 0, 2, ASIMOV, BOUNDS, _rates, **_KW)
    params = np.asarray(params, dtype=float)

    assert params[0] == pytest.approx(1.7, abs=1e-12), "the tested parameter was not pinned"
    assert BOUNDS[1][0] <= params[1] <= BOUNDS[1][1], "the profiled parameter left its bounds"
    assert np.isfinite(nll)


def test_conditional_fit_2d_ultranest_pins_both_tested_parameters():
    """
    Both parameters of interest are held while a third is profiled. Uses a genuinely
    3-parameter call so there is something left to sample over -- with only two parameters
    this function short-circuits before reaching UltraNest, which is exactly the path
    `test_constraints.py` already covers and the reason the sampling path stayed cold.
    """
    @njit(fastmath=True, nogil=True)
    def rates3(params, S_sumw2, B_sumw2):
        mu = np.array([
            20.0 * params[0] + 1.0,
            15.0 * params[1] + 1.0,
            6.0 * params[0] + 9.0 * params[1] + 3.0 * params[2] + 1.0,
        ])
        return mu, S_sumw2

    data = np.array([21.0, 16.0, 19.0])
    nll, params = conditional_fit_2d_ultranest(
        1.0, 1.0, 0, 1, 3, data, BOUNDS + [(0.3, 2.5)], rates3,
        verbose=0, S_sumw2=SUMW2, B_sumw2=SUMW2,
    )
    params = np.asarray(params, dtype=float)

    assert params[0] == pytest.approx(1.0, abs=1e-12), "parameter A was not pinned"
    assert params[1] == pytest.approx(1.0, abs=1e-12), "parameter B was not pinned"
    assert 0.3 <= params[2] <= 2.5, "the profiled parameter left its bounds"
    assert np.isfinite(nll)


def test_conditional_nll_never_undercuts_the_unconditional_minimum():
    """
    The property that makes the Feldman-Cousins test statistic well-defined: constraining a
    parameter cannot improve the fit, so `cond - uncond` is never negative and the statistic
    is never below zero. Swept across the grid rather than checked at one point.

    The tolerance is sampler noise, not slack in the claim -- measured at ~1e-4 at the true
    value, where the two minima genuinely coincide.
    """
    uncond, _ = unconditional_fit_ultranest(ASIMOV, 2, BOUNDS, _rates, **_KW)

    for test_val in (0.5, 0.8, 1.0, 1.5, 2.0):
        cond, _ = conditional_fit_1d_ultranest(test_val, 0, 2, ASIMOV, BOUNDS, _rates, **_KW)
        assert cond - uncond >= -1e-2, (
            f"conditional NLL {cond} undercut the unconditional minimum {uncond} at "
            f"p0={test_val}, by more than sampler noise explains"
        )


def test_test_statistic_is_zero_at_the_true_value_and_grows_away_from_it():
    """
    The complement of the inequality above. At the generating value the constraint costs
    nothing, so the statistic is ~0; away from it the fit is genuinely worse, so it grows.

    A fit that ignored `test_val` entirely would satisfy the inequality everywhere and fail
    here, since every point would return the unconditional minimum.
    """
    uncond, _ = unconditional_fit_ultranest(ASIMOV, 2, BOUNDS, _rates, **_KW)

    t = {}
    for test_val in (1.0, 1.5, 2.0):
        cond, _ = conditional_fit_1d_ultranest(test_val, 0, 2, ASIMOV, BOUNDS, _rates, **_KW)
        t[test_val] = cond - uncond

    assert abs(t[1.0]) < 1e-2, f"statistic at the true value should be ~0, got {t[1.0]}"
    assert t[1.5] > 1.0, f"statistic should be clearly positive away from truth, got {t[1.5]}"
    assert t[2.0] > t[1.5], "statistic should grow with distance from the true value"


def test_unconditional_fit_ultranest_honours_bounds_func():
    """
    `bounds_func` is the hook for joint constraints a per-parameter box cannot express, and
    on this path it is called with an empty `fixed_values` dict because nothing is fixed in
    an unconditional fit. Nothing exercised it here.

    The narrowed box excludes the true value, so a fit that ignored `bounds_func` would come
    back at ~1.0 and fail.
    """
    def narrow_bounds(fixed_values, free_indices, default_bounds_list):
        assert fixed_values == {}, (
            f"unconditional fit passed a non-empty fixed_values: {fixed_values}"
        )
        assert list(free_indices) == [0, 1]
        return [(0.3, 0.6), (0.3, 2.5)]

    _, best = unconditional_fit_ultranest(
        ASIMOV, 2, BOUNDS, _rates, bounds_func=narrow_bounds, **_KW
    )
    best = np.asarray(best, dtype=float)

    assert 0.3 <= best[0] <= 0.6 + 1e-6, (
        f"p0={best[0]} is outside the box bounds_func returned; bounds_func was ignored"
    )


# --- The unbinned likelihood through UltraNest ----------------------------------------
#
# Each of the three UltraNest fits branches on `likelihood_type`, and only the binned side
# was ever reached. The unbinned side is a separate code path building `probs` from
# `pdf_components` and calling `calc_nll_unbinned`, and it is reachable in normal use --
# `compute_fc_intervals(likelihood_type="unbinned", strategy="ultranest")` is a documented
# combination. Nothing ran it.


def _unbinned_rates(params, probs):
    """Extended unbinned model: each parameter scales one component's yield."""
    s_probs, b_probs = probs[0], probs[1]
    expected_total = params[0] * 12.0 + params[1] * 6.0
    if len(s_probs) == 0 and len(b_probs) == 0:
        return expected_total, np.array([])
    return expected_total, params[0] * 12.0 * s_probs + params[1] * 6.0 * b_probs


def _s_pdf(x):
    return np.exp(-0.5 * (x - 5.0) ** 2)


def _b_pdf(x):
    return np.full_like(x, 0.1)


def _generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool):
    """Module-level so it survives pickling into a ProcessPoolExecutor worker."""
    n_sig = np.random.poisson(true_params[0] * 12.0)
    n_bkg = np.random.poisson(true_params[1] * 6.0)
    parts = []
    if n_sig > 0:
        parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
    if n_bkg > 0:
        parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
    return np.concatenate(parts) if parts else np.array([])


@pytest.fixture(scope="module")
def unbinned_events():
    rng = np.random.default_rng(17)
    return np.concatenate([rng.normal(5.0, 1.0, size=40), rng.exponential(3.0, size=20)])


def test_unconditional_fit_ultranest_handles_the_unbinned_likelihood(unbinned_events):
    """The unbinned branch runs and returns a finite optimum inside the box."""
    nll, best = unconditional_fit_ultranest(
        unbinned_events, 2, BOUNDS, _unbinned_rates, verbose=0,
        pdf_components=[_s_pdf, _b_pdf], likelihood_type="unbinned",
    )
    best = np.asarray(best, dtype=float)

    assert np.isfinite(nll)
    assert best.shape == (2,)
    for value, (lo, hi) in zip(best, BOUNDS):
        assert lo <= value <= hi, f"fitted parameter {value} escaped its bounds ({lo}, {hi})"


def test_conditional_fits_ultranest_handle_the_unbinned_likelihood(unbinned_events):
    """
    Both conditional fits on the unbinned path: the tested parameters stay pinned, and the
    profile-likelihood inequality still holds against the unconditional unbinned fit.
    """
    kw = dict(verbose=0, pdf_components=[_s_pdf, _b_pdf], likelihood_type="unbinned")
    uncond, _ = unconditional_fit_ultranest(unbinned_events, 2, BOUNDS, _unbinned_rates, **kw)

    cond1, p1 = conditional_fit_1d_ultranest(
        1.4, 0, 2, unbinned_events, BOUNDS, _unbinned_rates, **kw
    )
    assert np.asarray(p1)[0] == pytest.approx(1.4, abs=1e-12)
    assert np.isfinite(cond1)
    assert cond1 - uncond >= -1e-2, "unbinned conditional NLL undercut the unconditional one"

    cond2, p2 = conditional_fit_2d_ultranest(
        1.4, 1.2, 0, 1, 3, unbinned_events, BOUNDS + [(0.3, 2.5)],
        lambda params, probs: _unbinned_rates(params, probs), **kw
    )
    p2 = np.asarray(p2)
    assert p2[0] == pytest.approx(1.4, abs=1e-12)
    assert p2[1] == pytest.approx(1.2, abs=1e-12)
    assert np.isfinite(cond2)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this comparison")
def test_ultranest_and_scipy_agree_on_the_asimov_optimum():
    """
    Two independent optimizers -- gradient descent and nested sampling -- must find the same
    minimum of the same likelihood. Neither is the reference; agreement is the evidence.

    This is the check that would catch a likelihood wired up differently on the two paths,
    which no single-backend test can see.
    """
    nll_un, best_un = unconditional_fit_ultranest(ASIMOV, 2, BOUNDS, _rates, **_KW)
    nll_sp, best_sp = unconditional_fit_scipy(
        ASIMOV, 2, BOUNDS, _rates, S_sumw2=SUMW2, B_sumw2=SUMW2
    )

    assert nll_un == pytest.approx(nll_sp, abs=1e-2), (
        f"the two backends disagree on the minimum NLL: ultranest={nll_un}, scipy={nll_sp}"
    )
    np.testing.assert_allclose(
        np.asarray(best_un, dtype=float), np.asarray(best_sp, dtype=float), atol=0.05
    )


# --- Toy generation dispatching to UltraNest ------------------------------------------
#
# `toys.py` has two `strategy == "ultranest"` branches, one per pool: `_worker_unbinned_toy`
# (processes, unbinned) and `fit_single_toy` (threads, binned). Both were cold for the same
# reason as the fits themselves -- nothing drove a toy run through UltraNest.


def test_binned_toys_dispatch_to_ultranest():
    """
    `strategy="ultranest"` through the binned (thread pool) toy path.

    Deliberately tiny -- two toys on a three-bin model -- because each one runs a full
    nested-sampling fit twice, unconditionally and conditionally. The assertions are the
    same invariants the statistic must satisfy however it was produced.
    """
    from pyfc.toys import generate_and_fit_toys_python

    t_stats = generate_and_fit_toys_python(
        true_params=TRUE_PARAMS, n_params=2, fit_mode="1d",
        fix_idx=0, fix_A=None, fix_B=None, t_vA=1.0, t_vB=None,
        bounds_list=BOUNDS, n_toys=2, strategy="ultranest",
        num_cores=1, verbose=0, likelihood_type="binned",
        compute_rates_func=_rates, S_sumw2=SUMW2, B_sumw2=SUMW2,
        use_finite_mc=False,
    )

    t_stats = np.asarray(t_stats, dtype=float)
    assert t_stats.shape == (2,)
    assert np.all(np.isfinite(t_stats))
    assert np.all(t_stats >= 0.0), (
        "a negative test statistic escaped the max(0.0, ...) clamp on the ultranest path"
    )


def test_unbinned_toys_dispatch_to_ultranest(unbinned_events):
    """
    The same for the unbinned (process pool) path, which reaches
    `_worker_unbinned_toy`'s ultranest branch.

    Everything handed to the pool has to be picklable, which is why the model functions
    here are module-level rather than closures -- a lambda would send this down the
    ThreadPoolExecutor fallback instead and quietly test the wrong branch.
    """
    from pyfc.toys import generate_and_fit_toys_python

    pool = unbinned_events
    t_stats = generate_and_fit_toys_python(
        true_params=TRUE_PARAMS, n_params=2, fit_mode="1d",
        fix_idx=0, fix_A=None, fix_B=None, t_vA=1.0, t_vB=None,
        bounds_list=BOUNDS, n_toys=2, strategy="ultranest",
        num_cores=1, verbose=0, likelihood_type="unbinned",
        pdf_components=[_s_pdf, _b_pdf],
        compute_rates_func=_unbinned_rates,
        generate_toy_func=_generate_unbinned_toy,
        S_mc_pool=pool, B_mc_pool=pool,
    )

    t_stats = np.asarray(t_stats, dtype=float)
    assert t_stats.shape == (2,)
    assert np.all(np.isfinite(t_stats))
    assert np.all(t_stats >= 0.0)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this comparison")
def test_every_fit_function_returns_array_like_parameters():
    """
    All six fit functions are drop-in alternatives selected by `strategy`, and
    `compute_fc_intervals` assigns whichever it calls straight into `results["best_fit"]`,
    which it initialises as an ndarray. They must therefore agree on what they return.

    They did not: `unconditional_fit_ultranest` alone returned a plain Python list, because
    it passes UltraNest's result dict straight through. Nothing broke -- the value is only
    indexed and serialised -- but the type of `results["best_fit"]` silently depended on
    which strategy ran, so any future numpy operation on it would have worked under three
    strategies and failed under the fourth.
    """
    calls = [
        ("unconditional_fit_scipy", unconditional_fit_scipy,
         (ASIMOV, 2, BOUNDS, _rates), dict(S_sumw2=SUMW2, B_sumw2=SUMW2)),
        ("unconditional_fit_ultranest", unconditional_fit_ultranest,
         (ASIMOV, 2, BOUNDS, _rates), _KW),
        ("conditional_fit_1d_scipy", conditional_fit_1d_scipy,
         (1.0, 0, 2, ASIMOV, BOUNDS, _rates), dict(S_sumw2=SUMW2, B_sumw2=SUMW2)),
        ("conditional_fit_1d_ultranest", conditional_fit_1d_ultranest,
         (1.0, 0, 2, ASIMOV, BOUNDS, _rates), _KW),
        ("conditional_fit_2d_scipy", conditional_fit_2d_scipy,
         (1.0, 1.0, 0, 1, 2, ASIMOV, BOUNDS, _rates), dict(S_sumw2=SUMW2, B_sumw2=SUMW2)),
        ("conditional_fit_2d_ultranest", conditional_fit_2d_ultranest,
         (1.0, 1.0, 0, 1, 2, ASIMOV, BOUNDS, _rates), _KW),
    ]

    offenders = []
    for name, fn, args, kwargs in calls:
        _, params = fn(*args, **kwargs)
        if not isinstance(params, np.ndarray):
            offenders.append(f"  {name} -> {type(params).__name__}")

    assert not offenders, (
        "these fit functions do not return np.ndarray, so results['best_fit'] changes "
        "type depending on which strategy ran:\n" + "\n".join(offenders)
    )
