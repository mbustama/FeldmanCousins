"""
Tests for the unbinned grid-search path (`pyfc/unbinned.py`).

WHY THIS FILE EXISTS: coverage put `unbinned.py` at 34%, and reading which lines were cold
showed the gap was not scattered -- it was four whole functions,
`conditional_fit_grid_unbinned_1d`, `conditional_fit_grid_unbinned_2d`,
`generate_and_fit_toys_grid_unbinned_1d` and `generate_and_fit_toys_grid_unbinned_2d`,
executed by nothing at all. Every one is imported and dispatched to by
`compute_fc_intervals`, so all four ship and are reachable by users; the suite simply never
combined `likelihood_type="unbinned"` with `strategy="grid"`. Across the whole test suite
`likelihood_type="unbinned"` appeared exactly once, and that one test uses
`strategy="scipy"`.

That combination is the one place in PyFC where the profiling is exhaustive rather than
numerical, which makes it both the easiest path to verify exactly and the one where a
silent error is least likely to be noticed: a brute-force scan that quietly profiles over
the wrong axis still returns plausible, finite, monotonic-looking numbers.

So these tests do not assert "it ran and produced finite output". Each one compares against
an oracle computed independently in the test body -- an explicit Python loop over the same
grid calling `calc_nll_unbinned` directly -- or against a statistical invariant that must
hold regardless of the data. Every test here was verified to fail against a deliberately
broken version of the code before being kept.
"""

import itertools

import numpy as np
import pytest

from pyfc.orchestrator import compute_fc_intervals
from pyfc.unbinned import (
    calc_nll_unbinned,
    conditional_fit_grid_unbinned_1d,
    conditional_fit_grid_unbinned_2d,
    generate_and_fit_toys_grid_unbinned_1d,
    generate_and_fit_toys_grid_unbinned_2d,
    unconditional_fit_grid_unbinned,
)


def _s_pdf(x):
    """Gaussian signal density, peaked at 5."""
    return np.exp(-0.5 * (x - 5.0) ** 2)


def _b_pdf(x):
    """Flat background density."""
    return np.full_like(x, 0.1)


def _compute_rates(params, probs):
    """
    Two-component extended unbinned model: each parameter scales one component's yield.

    Any parameter beyond the first two is a pure nuisance that does not enter the rate,
    which is deliberate -- it gives the 2D conditional fit a genuine third axis to profile
    over without changing the likelihood's shape in the two tested ones.
    """
    s_probs, b_probs = probs[0], probs[1]
    expected_total = params[0] * 10.0 + params[1] * 5.0
    if len(s_probs) == 0 and len(b_probs) == 0:
        return expected_total, np.array([])
    p_events = params[0] * 10.0 * s_probs + params[1] * 5.0 * b_probs
    return expected_total, p_events


def _generate_toy(true_params, S_mc_pool, B_mc_pool):
    """Parametric bootstrap: Poisson yields resampled from the supplied MC pools."""
    n_sig = np.random.poisson(true_params[0] * 10.0)
    n_bkg = np.random.poisson(true_params[1] * 5.0)
    parts = []
    if n_sig > 0:
        parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
    if n_bkg > 0:
        parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
    return np.concatenate(parts) if parts else np.array([])


@pytest.fixture(scope="module")
def pools():
    rng = np.random.default_rng(7)
    return (rng.normal(loc=5.0, scale=1.0, size=500),
            rng.exponential(scale=3.0, size=500))


@pytest.fixture(scope="module")
def observed(pools):
    S_mc_pool, B_mc_pool = pools
    np.random.seed(3)
    return _generate_toy(np.array([1.0, 1.0]), S_mc_pool, B_mc_pool)


@pytest.fixture(scope="module")
def pdf_components():
    return [_s_pdf, _b_pdf]


def _grid_points(grids):
    """Exactly how the orchestrator builds its scan grid (itertools.product)."""
    return np.array(list(itertools.product(*grids)), dtype=np.float64)


def _nll_at(params, obs_events, pdf_components):
    """Independent oracle: evaluate the NLL at one parameter point, from scratch."""
    len_obs = len(obs_events)
    probs = [pdf(obs_events) if len_obs > 0 else np.array([]) for pdf in pdf_components]
    return calc_nll_unbinned(np.asarray(params, dtype=np.float64), len_obs, probs,
                             _compute_rates)


def _brute_force_min(candidates, obs_events, pdf_components):
    """Oracle: scan every candidate point in plain Python and keep the lowest NLL."""
    best_nll, best_p = np.inf, None
    for p in candidates:
        nll = _nll_at(p, obs_events, pdf_components)
        if nll < best_nll:
            best_nll, best_p = nll, np.asarray(p, dtype=np.float64)
    return best_nll, best_p


def test_unconditional_fit_grid_unbinned_finds_the_true_grid_minimum(observed, pdf_components):
    """
    The unconditional fit must return the lowest NLL on the grid, and the point attaining
    it. Compared against an explicit scan, not against itself.
    """
    grids = [np.linspace(0.4, 2.0, 5), np.linspace(0.4, 2.0, 5)]
    full = _grid_points(grids)

    nll, params = unconditional_fit_grid_unbinned(observed, pdf_components, full,
                                                  _compute_rates)
    exp_nll, exp_params = _brute_force_min(full, observed, pdf_components)

    assert nll == pytest.approx(exp_nll, rel=1e-12)
    np.testing.assert_allclose(params, exp_params, rtol=1e-12)


def test_conditional_fit_grid_unbinned_1d_pins_the_tested_parameter(observed, pdf_components):
    """
    The 1D conditional fit must hold the parameter of interest exactly at the tested value
    while profiling the rest, and must return the minimum over the nuisance sub-grid.

    Pinning is asserted exactly rather than approximately: the tested value is written into
    the parameter vector, never optimized, so anything but equality means the wrong axis is
    being held.
    """
    grids = [np.linspace(0.4, 2.0, 5), np.linspace(0.4, 2.0, 5)]
    fix_idx, test_val = 0, grids[0][2]
    free_grids = [grids[i] for i in range(2) if i != fix_idx]
    cond = _grid_points(free_grids)

    nll, params = conditional_fit_grid_unbinned_1d(test_val, fix_idx, 2, observed,
                                                   pdf_components, cond, _compute_rates)

    assert params[fix_idx] == test_val, "the parameter under test was not held fixed"

    candidates = [[test_val, free] for free in free_grids[0]]
    exp_nll, exp_params = _brute_force_min(candidates, observed, pdf_components)
    assert nll == pytest.approx(exp_nll, rel=1e-12)
    np.testing.assert_allclose(params, exp_params, rtol=1e-12)


def test_conditional_fit_grid_unbinned_2d_pins_both_tested_parameters(observed, pdf_components):
    """
    The 2D conditional fit holds two parameters and profiles the third. Uses a genuinely
    3-parameter model so there is something left to profile -- with only two parameters the
    sub-grid is empty and the test would pass without exercising the injection loop.
    """
    grids = [np.linspace(0.4, 2.0, 4), np.linspace(0.4, 2.0, 4), np.linspace(0.5, 1.5, 3)]
    fix_A, fix_B = 0, 1
    test_vA, test_vB = grids[0][1], grids[1][2]
    free_grids = [grids[i] for i in range(3) if i not in (fix_A, fix_B)]
    cond = _grid_points(free_grids)

    nll, params = conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, 3,
                                                   observed, pdf_components, cond,
                                                   _compute_rates)

    assert params[fix_A] == test_vA, "parameter A was not held fixed"
    assert params[fix_B] == test_vB, "parameter B was not held fixed"

    candidates = [[test_vA, test_vB, free] for free in free_grids[0]]
    exp_nll, exp_params = _brute_force_min(candidates, observed, pdf_components)
    assert nll == pytest.approx(exp_nll, rel=1e-12)
    np.testing.assert_allclose(params, exp_params, rtol=1e-12)


def test_conditional_nll_never_undercuts_the_unconditional_minimum(observed, pdf_components):
    """
    The defining property of a profile likelihood ratio: constraining a parameter cannot
    improve the fit, so the conditional NLL is >= the unconditional one at every tested
    point, and the test statistic `cond - uncond` is therefore never negative.

    Swept across the whole grid rather than at a single point, so an error that only shows
    up away from the best-fit value cannot slip through. This is what makes the FC test
    statistic well-defined; if it failed, every interval the package produces would be
    suspect.
    """
    grids = [np.linspace(0.4, 2.0, 5), np.linspace(0.4, 2.0, 5)]
    full = _grid_points(grids)
    uncond_nll, _ = unconditional_fit_grid_unbinned(observed, pdf_components, full,
                                                    _compute_rates)

    for fix_idx in (0, 1):
        free_grids = [grids[i] for i in range(2) if i != fix_idx]
        cond = _grid_points(free_grids)
        for test_val in grids[fix_idx]:
            cond_nll, _ = conditional_fit_grid_unbinned_1d(test_val, fix_idx, 2, observed,
                                                           pdf_components, cond,
                                                           _compute_rates)
            assert cond_nll >= uncond_nll - 1e-9, (
                f"conditional NLL {cond_nll} undercut the unconditional minimum "
                f"{uncond_nll} at p{fix_idx}={test_val}"
            )


def test_conditional_fit_reaches_the_unconditional_minimum_at_the_best_fit_point(
    observed, pdf_components
):
    """
    The complement of the inequality above: at the grid point where the unconditional fit
    actually sits, constraining that parameter costs nothing, so the two minima must
    coincide and the test statistic must be exactly zero.

    An implementation that profiled over the wrong axis, or that silently excluded the
    best-fit value from its sub-grid, would satisfy the inequality everywhere but fail
    here.
    """
    grids = [np.linspace(0.4, 2.0, 5), np.linspace(0.4, 2.0, 5)]
    full = _grid_points(grids)
    uncond_nll, best = unconditional_fit_grid_unbinned(observed, pdf_components, full,
                                                       _compute_rates)

    for fix_idx in (0, 1):
        free_grids = [grids[i] for i in range(2) if i != fix_idx]
        cond = _grid_points(free_grids)
        cond_nll, _ = conditional_fit_grid_unbinned_1d(best[fix_idx], fix_idx, 2, observed,
                                                       pdf_components, cond, _compute_rates)
        assert cond_nll == pytest.approx(uncond_nll, rel=1e-12), (
            f"fixing p{fix_idx} at its own best-fit value changed the minimum NLL"
        )


def test_toys_grid_unbinned_1d_returns_one_nonnegative_finite_statistic_per_toy(
    pools, pdf_components
):
    """
    The 1D grid toy generator returns exactly `n_toys` test statistics, each finite and
    clamped at zero. The clamp matters: `cond - uncond` is mathematically non-negative but
    can go slightly negative in floating point when the two coincide, and a negative
    entry would corrupt the empirical critical value the whole interval depends on.
    """
    S_mc_pool, B_mc_pool = pools
    grids = [np.linspace(0.5, 1.5, 3), np.linspace(0.5, 1.5, 3)]
    full = _grid_points(grids)
    cond = _grid_points([grids[1]])

    np.random.seed(19)
    t_stats = generate_and_fit_toys_grid_unbinned_1d(
        grids[0][1], 0, np.array([1.0, 1.0]), 2, pdf_components, full, cond, 5,
        S_mc_pool, B_mc_pool, _compute_rates, _generate_toy,
    )

    assert t_stats.shape == (5,)
    assert np.all(np.isfinite(t_stats))
    assert np.all(t_stats >= 0.0)


def test_toys_grid_unbinned_2d_returns_one_nonnegative_finite_statistic_per_toy(
    pools, pdf_components
):
    """As above for the 2D generator, which profiles a third parameter."""
    S_mc_pool, B_mc_pool = pools
    grids = [np.linspace(0.5, 1.5, 3), np.linspace(0.5, 1.5, 3), np.linspace(0.5, 1.5, 2)]
    full = _grid_points(grids)
    cond = _grid_points([grids[2]])

    np.random.seed(23)
    t_stats = generate_and_fit_toys_grid_unbinned_2d(
        grids[0][1], grids[1][1], 0, 1, np.array([1.0, 1.0, 1.0]), 3, pdf_components,
        full, cond, 4, S_mc_pool, B_mc_pool, _compute_rates, _generate_toy,
    )

    assert t_stats.shape == (4,)
    assert np.all(np.isfinite(t_stats))
    assert np.all(t_stats >= 0.0)


def test_calc_nll_unbinned_with_zero_observations_returns_the_expected_total(pdf_components):
    """
    An unbinned toy can legitimately contain zero events when the expected yield is small.
    The extended likelihood then reduces to its normalisation term alone -- there are no
    per-event densities to sum -- so the NLL is exactly the expected total. Getting this
    wrong would bias the critical value in precisely the low-yield regime where
    Feldman-Cousins matters most.

    The second case is what actually pins the `len_obs == 0` guard. With a well-behaved
    `compute_rates_func` the guard is invisible: it returns an empty `p_events`, and
    `np.sum(np.log([]))` is 0.0, so falling through to the per-event branch happens to give
    the same answer. The guard only earns its keep when the user's function does *not*
    special-case the empty input and hands back densities anyway -- a very easy thing to
    write by accident, since nothing in the signature suggests the array can be empty.
    The contract is that those densities are ignored outright, not summed over.
    """
    params = np.array([0.5, 0.5])
    expected_total = params[0] * 10.0 + params[1] * 5.0

    assert calc_nll_unbinned(params, 0, [np.array([]), np.array([])],
                             _compute_rates) == pytest.approx(expected_total)

    def _careless_rates(p, probs):
        """Returns per-event densities regardless of whether any events were observed."""
        return expected_total, np.array([0.0, 1e-30])

    assert calc_nll_unbinned(params, 0, [np.array([]), np.array([])],
                             _careless_rates) == pytest.approx(expected_total), (
        "with zero observations the per-event densities must not be consulted at all; "
        "summing them here would return a barrier-inflated NLL for an empty dataset"
    )


def test_calc_nll_unbinned_rejects_nan_per_event_densities():
    """
    A NaN density means the user's `compute_rates_func` divided by zero, or took the log or
    square root of a negative number. It must fail loudly.

    NaN is not "unphysical" in the sense the barrier below handles: `p_events <= p_floor`
    is False for NaN, so without this guard a NaN would fall straight through into
    `np.log`, yielding a NaN NLL with no barrier, no warning, and no gradient for the
    optimizer to recover from. The fit would then wander and report a plausible interval
    built on nothing. `binned.py`'s `calc_nll` has had this guard tested since v0.10.0;
    the unbinned twin did not.
    """
    def _nan_rates(p, probs):
        return 10.0, np.array([0.5, np.nan, 0.3])

    with pytest.raises(ValueError, match="NaN"):
        calc_nll_unbinned(np.array([1.0, 1.0]), 3, [np.zeros(3), np.zeros(3)], _nan_rates)


def test_calc_nll_unbinned_applies_a_finite_barrier_to_unphysical_densities():
    """
    Densities at or below the floor take the quadratic-barrier branch instead of
    `-log(p)`, which would diverge. The result must stay finite, must exceed the
    unpenalised NLL, and must grow as the density becomes more negative -- that monotonic
    slope is the whole point, since it is what points the optimizer back toward the
    physical region rather than leaving it on a flat or inverted surface.
    """
    def _rates_at(value):
        def _f(p, probs):
            return 10.0, np.array([1.0, value])
        return _f

    params, probs = np.array([1.0, 1.0]), [np.zeros(2), np.zeros(2)]

    physical = calc_nll_unbinned(params, 2, probs, _rates_at(1.0))
    mild = calc_nll_unbinned(params, 2, probs, _rates_at(-1e-6))
    severe = calc_nll_unbinned(params, 2, probs, _rates_at(-1e-3))

    assert np.isfinite(mild) and np.isfinite(severe), "the barrier produced a non-finite NLL"
    assert mild > physical, "an unphysical density was not penalised at all"
    assert severe > mild, (
        "the barrier is not monotonic in the violation, so it gives the optimizer no "
        "direction back to the physical region"
    )


def test_compute_fc_intervals_unbinned_grid_end_to_end(tmp_path, pools):
    """
    The integration test the four functions above were missing: a real
    `compute_fc_intervals` run with `likelihood_type="unbinned"` and `strategy="grid"`
    together, which is the only combination that reaches them through the orchestrator's
    own dispatch rather than by direct import.

    The observed test statistics are checked against an oracle rather than merely for
    finiteness. Under `strategy="grid"` the DATA fits carry no randomness -- only the toys
    do -- so `t_data` is exactly reproducible here from the same grid functions, and any
    mis-wiring in the orchestrator's dispatch shows up as a numeric disagreement. Asserting
    only that the values are finite and non-negative would not: `t_data` is clamped by
    `max(0.0, ...)`, so a dispatch bug that shifted the conditional NLL would silently
    produce a valid-looking array of zeros.
    """
    S_mc_pool, B_mc_pool = pools
    np.random.seed(31)
    observed = _generate_toy(np.array([1.0, 1.0]), S_mc_pool, B_mc_pool)
    grids = [np.linspace(0.5, 1.5, 3), np.linspace(0.5, 1.5, 3)]

    results, _ = compute_fc_intervals(
        data=observed, grids=grids,
        compute_rates_func=_compute_rates,
        generate_toy_func=_generate_toy,
        pdf_components=[_s_pdf, _b_pdf],
        cl=[0.90], n_toys=4, strategy="grid", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="unbinned", S_mc_pool=S_mc_pool, B_mc_pool=B_mc_pool,
        output_file=None, save_directory=str(tmp_path),
        compute_1D_intervals=True, compute_2D_intervals=True,
    )

    # Oracle, rebuilt from the same grid the orchestrator uses.
    full = _grid_points(grids)
    uncond_nll, _ = unconditional_fit_grid_unbinned(observed, [_s_pdf, _b_pdf], full,
                                                    _compute_rates)

    for p_idx in range(2):
        t_data = results[f"1d_t_data_p{p_idx+1}"]
        cond = _grid_points([grids[i] for i in range(2) if i != p_idx])
        expected = np.array([
            max(0.0, conditional_fit_grid_unbinned_1d(pt, p_idx, 2, observed,
                                                      [_s_pdf, _b_pdf], cond,
                                                      _compute_rates)[0] - uncond_nll)
            for pt in grids[p_idx]
        ])
        np.testing.assert_allclose(t_data, expected, rtol=1e-9, atol=1e-12)

        # Guard the oracle itself: if every entry were zero the comparison above would
        # hold for almost any bug, so require the scan to have real dynamic range.
        assert expected.max() > 1e-6, "degenerate grid -- t_data is flat, so this proves little"

        accepted = results[f"1d_accepted_p{p_idx+1}"][0.90]
        assert accepted.dtype == bool
        assert accepted.shape == t_data.shape

    t_data_2d = results["2d_t_data_p1p2"]
    cond_2d = np.zeros((1, 0), dtype=np.float64)  # 2 params, both fixed -> nothing to profile
    expected_2d = np.array([
        [max(0.0, conditional_fit_grid_unbinned_2d(vA, vB, 0, 1, 2, observed,
                                                   [_s_pdf, _b_pdf], cond_2d,
                                                   _compute_rates)[0] - uncond_nll)
         for vB in grids[1]]
        for vA in grids[0]
    ])
    np.testing.assert_allclose(t_data_2d, expected_2d, rtol=1e-9, atol=1e-12)
    assert results["2d_accepted_p1p2"][0.90].shape == t_data_2d.shape
