"""
Tests for `extra_nll`, the hook for soft (Gaussian) constraints on nuisance parameters.

WHY THIS FILE EXISTS: `bounds_func` and `constraints` express HARD geometry on the
parameter box. Nothing expressed a SOFT term -- "this parameter is 1.0 +- 4.6%, because
someone measured it" -- which is how every external measurement of a systematic enters a
physics likelihood. `extra_nll` adds that term to the NLL at every evaluation.

The defect these tests are built around is not "the feature does not work". It is "the
feature works everywhere except one path", because that failure is invisible: the run
converges, the interval is well-formed, and it is wrong. The requesting analysis measured
that removing its constraints turns a 4.47e-09 s/eV limit into no limit at all, so a
penalty that silently fails to reach some fits produces a confidently unbounded answer.

The single most important case is the toys. Feldman-Cousins earns its coverage by
comparing the observed test statistic against a distribution built from Monte Carlo. If
the penalty reaches the data fit but not the toys, the statistic and its critical value
come from DIFFERENT likelihoods and the coverage guarantee is gone, with no symptom.
"""

import numba as _numba
import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import (
    SCIPY_AVAILABLE,
    ULTRANEST_AVAILABLE,
    conditional_fit_1d_scipy,
    conditional_fit_1d_ultranest,
    conditional_fit_2d_scipy,
    unconditional_fit_scipy,
    unconditional_fit_ultranest,
)
from pyfc.orchestrator import compute_fc_intervals

pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required")

# CENTRE is deliberately NOT 1.0, which is the midpoint of BOUNDS_1P and therefore where
# L-BFGS-B starts and where a degenerate (rate-independent) fit would sit anyway. With
# CENTRE at the midpoint, "the fit moved to the penalty centre" is true whether or not the
# penalty exists -- a mutation dropping it entirely went undetected until this changed.
CENTRE, WIDTH = 1.2, 0.05
N_OBS = np.array([50.0])
S2 = np.array([0.0])
BOUNDS_1P = [(0.5, 1.5)]


@njit(fastmath=True, nogil=True)
def _flat_rate(params, S_sumw2, B_sumw2):
    """Rate independent of the parameter: ONLY extra_nll can constrain it.

    That isolation is the point -- it makes the analytic oracle below exact rather than
    approximate, because the data contributes nothing to the parameter's curvature.
    """
    return np.array([50.0]), S_sumw2


@njit(fastmath=True, nogil=True)
def _gaussian_penalty(params):
    """((x - c)/s)**2 -- the -2lnL form. See the oracle test for why not 0.5*(...)."""
    return ((params[0] - CENTRE) / WIDTH) ** 2


def _t_statistic(test_val, penalty, n_params=1, bounds=None, rate=None):
    """The profile-likelihood test statistic PyFC itself forms: cond_nll - uncond_nll."""
    bounds = bounds or BOUNDS_1P
    rate = rate or _flat_rate
    uncond, _ = unconditional_fit_scipy(
        N_OBS, n_params, bounds, rate, S_sumw2=S2, B_sumw2=S2, extra_nll=penalty)
    cond, _ = conditional_fit_1d_scipy(
        test_val, 0, n_params, N_OBS, bounds, rate, S_sumw2=S2, B_sumw2=S2, extra_nll=penalty)
    return cond - uncond


# --- The analytic oracle -------------------------------------------------------------


def test_penalty_reproduces_the_closed_form_gaussian_interval():
    """
    A parameter constrained ONLY by `extra_nll` must reproduce c +- z*s exactly.

    With no information from the data, the NLL is the penalty alone, so the test statistic
    is t(x) = ((x - c)/s)**2 and t = z**2 precisely at x = c + z*s. This is a closed-form
    target, not a regression value: it cannot drift, and it pins the UNITS.

    PyFC's NLL is -2lnL (verify: calc_nll's delta is exactly twice the Poisson -lnL
    delta). A Gaussian of width s therefore contributes ((x-c)/s)**2, NOT the
    0.5*((x-c)/s)**2 of a -lnL convention. With the 0.5 the 1-sigma point lands at
    c + 1.4142*s instead of c + s -- every constraint silently sqrt(2) too weak, so a
    stated 4.6% prior would act as 6.5%. Nothing about that failure is visible in a
    converged fit, which is why it is pinned here.
    """
    for z in (1.0, 2.0, 3.0):
        t = _t_statistic(CENTRE + z * WIDTH, _gaussian_penalty)
        assert t == pytest.approx(z ** 2, abs=2e-3), (
            f"t at c + {z}sigma should be {z**2} in -2lnL units, got {t}"
        )

    # And the wrong convention is genuinely distinguishable, not a rounding difference.
    def half_penalty(p):
        return 0.5 * ((p[0] - CENTRE) / WIDTH) ** 2

    t_half = _t_statistic(CENTRE + WIDTH, half_penalty)
    assert t_half == pytest.approx(0.5, abs=2e-3), (
        "the -lnL convention should give t=0.5 at 1 sigma; if this now reads 1.0 the "
        "library's NLL units changed and the docstring must be corrected"
    )


def test_penalty_is_symmetric_about_its_centre():
    """A symmetric penalty must constrain both directions equally."""
    lo = _t_statistic(CENTRE - 2 * WIDTH, _gaussian_penalty)
    hi = _t_statistic(CENTRE + 2 * WIDTH, _gaussian_penalty)
    assert lo == pytest.approx(hi, abs=2e-3)
    assert lo == pytest.approx(4.0, abs=2e-3)


# --- Does it reach each fit separately? ----------------------------------------------


def test_penalty_reaches_the_unconditional_fit():
    """The unconditional fit must be pulled to the penalty's centre, not left free."""
    free, _ = unconditional_fit_scipy(
        N_OBS, 1, BOUNDS_1P, _flat_rate, S_sumw2=S2, B_sumw2=S2)
    pulled, best = unconditional_fit_scipy(
        N_OBS, 1, BOUNDS_1P, _flat_rate, S_sumw2=S2, B_sumw2=S2, extra_nll=_gaussian_penalty)
    assert best[0] == pytest.approx(CENTRE, abs=1e-3), (
        "the unconditional fit ignored the penalty and did not settle at its centre"
    )
    assert pulled >= free - 1e-9


def test_penalty_reaches_the_conditional_fit():
    """
    Away from the centre the conditional fit must pay the penalty. Without it, fixing a
    parameter that the rate ignores costs nothing and t is identically zero.
    """
    t_with = _t_statistic(CENTRE + 3 * WIDTH, _gaussian_penalty)
    uncond, _ = unconditional_fit_scipy(
        N_OBS, 1, BOUNDS_1P, _flat_rate, S_sumw2=S2, B_sumw2=S2)
    cond, _ = conditional_fit_1d_scipy(
        CENTRE + 3 * WIDTH, 0, 1, N_OBS, BOUNDS_1P, _flat_rate, S_sumw2=S2, B_sumw2=S2)
    assert (cond - uncond) == pytest.approx(0.0, abs=1e-6), "baseline should be flat"
    assert t_with > 8.0, f"conditional fit did not pay the penalty (t={t_with})"


@pytest.mark.skipif(not ULTRANEST_AVAILABLE, reason="UltraNest is required")
def test_penalty_sign_is_right_on_the_ultranest_paths():
    """
    UltraNest maximises a LOG-likelihood, so those call sites return a NEGATED NLL. The
    penalty must be negated with it. A sign error there does not crash -- it inverts the
    constraint, actively pushing the parameter AWAY from its measured value, which is
    worse than omitting it.
    """
    _, best = unconditional_fit_ultranest(
        N_OBS, 1, BOUNDS_1P, _flat_rate, verbose=0, S_sumw2=S2, B_sumw2=S2,
        extra_nll=_gaussian_penalty)
    assert np.asarray(best)[0] == pytest.approx(CENTRE, abs=0.02), (
        "UltraNest's optimum is not at the penalty centre; with the sign flipped it would "
        "be driven to a bound instead"
    )

    uncond, _ = unconditional_fit_ultranest(
        N_OBS, 1, BOUNDS_1P, _flat_rate, verbose=0, S_sumw2=S2, B_sumw2=S2,
        extra_nll=_gaussian_penalty)
    cond, _ = conditional_fit_1d_ultranest(
        CENTRE + 3 * WIDTH, 0, 1, N_OBS, BOUNDS_1P, _flat_rate, verbose=0,
        S_sumw2=S2, B_sumw2=S2, extra_nll=_gaussian_penalty)
    assert cond - uncond > 4.0, (
        f"constraining 3 sigma away should cost ~9 in -2lnL units, got {cond - uncond}"
    )


# --- The toy path: the defect most likely to survive ---------------------------------


@njit(fastmath=True, nogil=True)
def _signal_rate(params, S_sumw2, B_sumw2):
    """
    A rate the data constrains, with the two parameters COUPLED in the first bin.

    The coupling is essential and was learned the hard way. With a separable rate
    (bin 1 depending only on p0, bin 2 only on p1) a penalty on the nuisance contributes
    the SAME constant to the conditional and unconditional fits, so it cancels exactly in
    t_data = cond - uncond -- correctly, but invisibly, and a test asserting the statistic
    moves would fail against a flawless implementation.

    Coupling is also the physically interesting case: it is precisely because an
    unconstrained normalisation can absorb signal that the requesting analysis loses its
    limit entirely when its constraints are removed.
    """
    return np.array([20.0 * params[0] + 8.0 * params[1] + 5.0,
                     8.0 * params[1] + 3.0]), S_sumw2


# Centred at 1.15, NOT at 1.0. The data below pins the nuisance to exactly 1.0 (bin 2 is
# 8*p1 + 3 and reads 11), so a penalty centred there would add exactly zero at the optimum
# and every assertion about it would hold whether or not the hook worked at all. An earlier
# draft did exactly that and reported "extra_nll did not reach the DATA fits" against a
# perfectly correct implementation. The penalty has to actually fight the data to be
# visible.
_NUISANCE_CENTRE = 1.15


@njit(fastmath=True, nogil=True)
def _nuisance_penalty(params):
    """Constrains only p1 (the nuisance), leaving p0 (the parameter of interest) free."""
    return ((params[1] - _NUISANCE_CENTRE) / 0.05) ** 2


def _fc_run(tmp_path, extra_nll, strategy="scipy", n_toys=30, seed=7):
    np.random.seed(seed)
    data = np.array([33.0, 11.0])
    s2 = np.zeros(2)
    grids = [np.linspace(0.6, 1.4, 5), np.linspace(0.8, 1.2, 3)]
    results, _ = compute_fc_intervals(
        data=data, grids=grids, compute_rates_func=_signal_rate,
        cl=[0.90], n_toys=n_toys, strategy=strategy, num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        S_sumw2=s2, B_sumw2=s2, use_finite_mc_correction_binned=False,
        output_file=None, save_directory=str(tmp_path),
        compute_1D_intervals=True, compute_2D_intervals=False,
        extra_nll=extra_nll,
    )
    return results


def test_penalty_reaches_the_toy_fits_and_moves_the_critical_value(tmp_path):
    """
    THE test in this file. The penalty must enter the toy fits, not just the data fits.

    t_critical is built entirely from Monte Carlo. If `extra_nll` reached the data fit but
    not the toys, the observed statistic and its threshold would come from different
    likelihoods -- the interval would still be produced, still look plausible, and no
    longer have the coverage Feldman-Cousins exists to provide.

    Constraining the nuisance parameter reduces how far toys can wander while profiling,
    so the toy statistics shrink and the critical value must FALL. Comparing thresholds is
    the only way to see this: the returned interval alone cannot distinguish it.
    """
    free = _fc_run(tmp_path / "free", None)
    tied = _fc_run(tmp_path / "tied", _nuisance_penalty)

    tc_free = np.asarray(free["1d_t_critical_p1"][0.90], dtype=float)
    tc_tied = np.asarray(tied["1d_t_critical_p1"][0.90], dtype=float)

    assert np.all(np.isfinite(tc_free)) and np.all(np.isfinite(tc_tied))
    assert not np.allclose(tc_tied, tc_free), (
        "the critical values are identical with and without the constraint, which means "
        "extra_nll never reached the toy fits -- the data statistic and its threshold are "
        "being computed under different likelihoods"
    )
    assert tc_tied.mean() < tc_free.mean(), (
        f"constraining a nuisance parameter should shrink the toy statistic distribution; "
        f"mean t_critical went {tc_free.mean():.3f} -> {tc_tied.mean():.3f}"
    )


def test_penalty_changes_the_observed_statistic_too(tmp_path):
    """The data statistic must move as well -- the complement of the toy check above."""
    free = _fc_run(tmp_path / "free", None)
    tied = _fc_run(tmp_path / "tied", _nuisance_penalty)
    assert not np.allclose(tied["1d_t_data_p1"], free["1d_t_data_p1"]), (
        "extra_nll did not reach the DATA fits"
    )


# --- Full-vector reassembly order ----------------------------------------------------


@njit(fastmath=True, nogil=True)
def _asymmetric_penalty(params):
    """
    Deliberately asymmetric in its two arguments, with very different widths.

    A conditional fit must rebuild the full vector before calling the hook, putting the
    scan-fixed value at its own index. Swapping two entries is invisible to a symmetric
    penalty and invisible with one parameter -- it needs both an asymmetric term and two
    distinguishable positions to show up at all.
    """
    return ((params[0] - 1.0) / 0.5) ** 2 + ((params[1] - 1.0) / 0.02) ** 2


def test_conditional_fit_reassembles_the_full_vector_in_the_right_order():
    """
    The penalty sees the FULL vector in `grids` order, with the fixed value in its own
    slot. Fixing p0 far from its (loose) centre must cost far less than fixing p1 far from
    its (tight) centre -- if the two were transposed the costs would swap.
    """
    @njit(fastmath=True, nogil=True)
    def flat2(params, S_sumw2, B_sumw2):
        return np.array([50.0, 50.0]), S_sumw2

    n_obs = np.array([50.0, 50.0])
    s2 = np.zeros(2)
    bounds = [(0.5, 1.5), (0.5, 1.5)]
    off = 0.1

    uncond, _ = unconditional_fit_scipy(
        n_obs, 2, bounds, flat2, S_sumw2=s2, B_sumw2=s2, extra_nll=_asymmetric_penalty)
    cond_p0, _ = conditional_fit_1d_scipy(
        1.0 + off, 0, 2, n_obs, bounds, flat2, S_sumw2=s2, B_sumw2=s2,
        extra_nll=_asymmetric_penalty)
    cond_p1, _ = conditional_fit_1d_scipy(
        1.0 + off, 1, 2, n_obs, bounds, flat2, S_sumw2=s2, B_sumw2=s2,
        extra_nll=_asymmetric_penalty)

    t0, t1 = cond_p0 - uncond, cond_p1 - uncond
    assert t0 == pytest.approx((off / 0.5) ** 2, abs=1e-2), (
        f"fixing p0 should cost (0.1/0.5)^2 = 0.04; got {t0}. If this reads ~25 the full "
        "vector is being reassembled with the indices transposed."
    )
    assert t1 == pytest.approx((off / 0.02) ** 2, rel=2e-2), (
        f"fixing p1 should cost (0.1/0.02)^2 = 25; got {t1}"
    )


def test_2d_conditional_fit_places_both_fixed_values_correctly():
    """The same check for the 2D scan, which substitutes two fixed values at once."""
    @njit(fastmath=True, nogil=True)
    def flat3(params, S_sumw2, B_sumw2):
        return np.array([50.0]), S_sumw2

    @njit(fastmath=True, nogil=True)
    def pen3(params):
        return (((params[0] - 1.0) / 0.5) ** 2
                + ((params[1] - 1.0) / 0.02) ** 2
                + ((params[2] - 1.0) / 0.5) ** 2)

    n_obs = np.array([50.0]); s2 = np.array([0.0])
    bounds = [(0.5, 1.5)] * 3
    uncond, _ = unconditional_fit_scipy(
        n_obs, 3, bounds, flat3, S_sumw2=s2, B_sumw2=s2, extra_nll=pen3)
    cond, _ = conditional_fit_2d_scipy(
        1.1, 1.1, 0, 1, 3, n_obs, bounds, flat3, S_sumw2=s2, B_sumw2=s2, extra_nll=pen3)
    # p0 fixed 0.1 away with width 0.5 -> 0.04 ; p1 fixed 0.1 away with width 0.02 -> 25
    assert (cond - uncond) == pytest.approx(0.04 + 25.0, rel=2e-2), (
        f"expected 25.04 from the two fixed slots; got {cond - uncond}. A transposition "
        "here would give a very different number."
    )


# --- strategy="grid": the @njit path -------------------------------------------------


def test_grid_strategy_accepts_a_jitted_penalty(tmp_path):
    """
    `strategy="grid"` scans inside @njit code, which cannot call an arbitrary Python
    callable -- but numba CAN call a CPUDispatcher passed as an argument, which is already
    how `compute_rates_func` reaches those same functions. Verified here end to end,
    including through the @njit(parallel=True) toy generators.
    """
    tied = _fc_run(tmp_path / "tied", _nuisance_penalty, strategy="grid", n_toys=5)
    free = _fc_run(tmp_path / "free", None, strategy="grid", n_toys=5)

    assert np.all(np.isfinite(tied["1d_t_data_p1"]))
    assert np.all(tied["1d_t_data_p1"] >= 0.0)
    # Asserting only finiteness would pass just as well if the @njit fitters ignored the
    # penalty outright -- a mutation that did exactly that went undetected until this
    # comparison was added.
    assert not np.allclose(tied["1d_t_data_p1"], free["1d_t_data_p1"]), (
        "the grid strategy produced identical statistics with and without the penalty, so "
        "the jitted fitters are not applying extra_nll"
    )


@pytest.mark.skipif(
    bool(_numba.config.DISABLE_JIT),
    reason="with NUMBA_DISABLE_JIT=1 the guard deliberately stands down -- @njit returns a "
           "plain function and nothing is compiled, so the requirement does not apply. "
           "scripts/run_coverage.sh runs most of the suite under that flag.",
)
def test_grid_strategy_rejects_a_plain_callable_with_an_actionable_message(tmp_path):
    """
    A user who forgets @njit must be told what to do. Left to numba the failure is a
    316-character, 9-line TypingError naming neither `njit` nor the offending argument --
    measured, not guessed. The guard turns that into an instruction.
    """
    def plain_python_penalty(params):
        return ((params[1] - 1.15) / 0.05) ** 2

    with pytest.raises(TypeError) as excinfo:
        _fc_run(tmp_path, plain_python_penalty, strategy="grid", n_toys=3)

    msg = str(excinfo.value)
    assert "njit" in msg, "the error must name the decorator that fixes it"
    assert "extra_nll" in msg, "the error must name the offending argument"
    assert "grid" in msg, "the error must say which strategy imposed the requirement"


def test_grid_guard_does_not_fire_for_other_strategies(tmp_path):
    """The requirement is specific to grid; a plain callable is fine everywhere else."""
    def plain_python_penalty(params):
        return ((params[1] - 1.15) / 0.05) ** 2

    results = _fc_run(tmp_path, plain_python_penalty, strategy="scipy", n_toys=5)
    assert np.all(np.isfinite(results["1d_t_data_p1"]))


# --- unbinned ------------------------------------------------------------------------


def _s_pdf(x):
    return np.exp(-0.5 * (x - 5.0) ** 2)


def _b_pdf(x):
    return np.full_like(x, 0.1)


def _unbinned_rates(params, probs):
    s_probs, b_probs = probs[0], probs[1]
    total = params[0] * 12.0 + params[1] * 6.0
    if len(s_probs) == 0 and len(b_probs) == 0:
        return total, np.array([])
    return total, params[0] * 12.0 * s_probs + params[1] * 6.0 * b_probs


def test_penalty_applies_to_the_unbinned_likelihood_too():
    """
    The brief asked only for the binned likelihood, but `extra_nll(params) -> float` is
    likelihood-agnostic and a Gaussian nuisance constraint means the same thing either
    way. Binned-only support would mean an unbinned caller's penalty is silently ignored,
    which is exactly the hazard the grid guard exists to prevent elsewhere.
    """
    rng = np.random.default_rng(17)
    events = np.concatenate([rng.normal(5.0, 1.0, size=40), rng.exponential(3.0, size=20)])
    bounds = [(0.3, 2.5), (0.3, 2.5)]
    kw = dict(pdf_components=[_s_pdf, _b_pdf], likelihood_type="unbinned")

    def penalty(params):
        return ((params[1] - 2.0) / 0.02) ** 2   # tight, and far from the data's optimum

    free, best_free = unconditional_fit_scipy(events, 2, bounds, _unbinned_rates, **kw)
    tied, best_tied = unconditional_fit_scipy(
        events, 2, bounds, _unbinned_rates, extra_nll=penalty, **kw)

    assert np.asarray(best_tied)[1] == pytest.approx(2.0, abs=0.05), (
        "the unbinned fit ignored extra_nll; its nuisance did not move to the penalty "
        f"centre (got {np.asarray(best_tied)[1]})"
    )
    assert not np.isclose(np.asarray(best_free)[1], np.asarray(best_tied)[1])


# --- the no-op guarantee -------------------------------------------------------------


def test_extra_nll_none_reproduces_not_passing_it(tmp_path):
    """
    Every existing caller must be unaffected: `extra_nll=None` has to reproduce what the
    library did before, or this feature is a silent regression for the entire existing
    user base.

    The tolerance is not slack. An earlier draft asserted bit-for-bit equality and failed
    intermittently -- but so does the UNMODIFIED library: two identically-seeded runs in
    one process were measured to differ by up to 2e-15, because `fastmath=True` lets the
    compiler reassociate and the toy pool's scheduling varies. PyFC is reproducible to
    about 1e-15, never exactly, and asserting more than that tests numba's codegen
    stability rather than this feature. 1e-12 is still four orders tighter than any
    difference a real penalty term would make.
    """
    explicit = _fc_run(tmp_path / "explicit", None)
    np.random.seed(7)
    data = np.array([33.0, 11.0])
    s2 = np.zeros(2)
    grids = [np.linspace(0.6, 1.4, 5), np.linspace(0.8, 1.2, 3)]
    omitted, _ = compute_fc_intervals(
        data=data, grids=grids, compute_rates_func=_signal_rate,
        cl=[0.90], n_toys=30, strategy="scipy", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        S_sumw2=s2, B_sumw2=s2, use_finite_mc_correction_binned=False,
        output_file=None, save_directory=str(tmp_path / "omitted"),
        compute_1D_intervals=True, compute_2D_intervals=False,
    )  # extra_nll not passed at all

    np.testing.assert_allclose(
        explicit["1d_t_data_p1"], omitted["1d_t_data_p1"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(explicit["1d_t_critical_p1"][0.90], dtype=float),
        np.asarray(omitted["1d_t_critical_p1"][0.90], dtype=float),
        rtol=1e-12, atol=1e-12)


def test_penalty_reaches_the_toy_FITS_specifically():
    """
    Isolates the toy fits from everything upstream of them.

    The FC-level comparison above cannot do this on its own. Toys are generated at
    `true_params` taken from the conditional fit, which itself depends on `extra_nll`, so
    the toy statistics shift even when the toy FITS ignore the penalty entirely -- a
    mutation removing extra_nll from the toy dispatch passed that test. Here
    `generate_and_fit_toys_python` is called directly with identical `true_params` and an
    identical seed, so the only thing that can differ is whether the penalty entered the
    fits.

    This is the brief's stated worst case: the interval is still produced and still looks
    plausible, and only the critical value knows anything is wrong.
    """
    from pyfc.toys import generate_and_fit_toys_python

    common = dict(
        true_params=np.array([1.0, 1.0]), n_params=2, fit_mode="1d",
        fix_idx=0, fix_A=None, fix_B=None, t_vA=1.0, t_vB=None,
        bounds_list=[(0.6, 1.4), (0.8, 1.2)], n_toys=12, strategy="scipy",
        num_cores=1, verbose=0, likelihood_type="binned",
        compute_rates_func=_signal_rate, S_sumw2=np.zeros(2), B_sumw2=np.zeros(2),
        use_finite_mc=False,
    )

    np.random.seed(3)
    free = np.asarray(generate_and_fit_toys_python(**common), dtype=float)
    np.random.seed(3)
    tied = np.asarray(generate_and_fit_toys_python(extra_nll=_nuisance_penalty, **common),
                      dtype=float)

    assert free.shape == tied.shape == (12,)
    assert np.all(np.isfinite(tied)) and np.all(tied >= 0.0)
    assert not np.allclose(free, tied), (
        "toy test statistics are identical with and without extra_nll, from identical "
        "true_params and an identical seed -- the penalty is not entering the toy fits, "
        "so t_critical would be computed under a different likelihood than t_data"
    )


def test_orchestrator_forwards_extra_nll_to_the_toy_generators(tmp_path):
    """
    Pins the threading contract itself: `compute_fc_intervals` must hand `extra_nll` to
    the toy machinery, for every strategy.

    This is separate from the test above for a reason neither can cover alone. That one
    calls `generate_and_fit_toys_python` directly, so it proves the function honours the
    penalty but is blind to the orchestrator forgetting to pass it. The FC-level
    comparison is blind too, because toys are generated at `true_params` from the
    conditional fit -- so removing the penalty from the toy dispatch still shifts the
    numbers, and the run still looks different-but-fine.

    Intercepting the call is the only way to see the difference. The mutation that
    survived both other tests -- deleting `extra_nll=extra_nll` from the orchestrator's
    toy dispatch -- fails here immediately.
    """
    import pyfc.orchestrator as orch

    seen = {"python": [], "grid": []}
    real_python = orch.generate_and_fit_toys_python
    real_grid_1d = orch.generate_and_fit_toys_grid_1d

    def spy_python(*a, **kw):
        seen["python"].append(kw.get("extra_nll", "ABSENT"))
        return real_python(*a, **kw)

    def spy_grid_1d(*a, **kw):
        # njit signature: extra_nll is the final positional argument
        seen["grid"].append(a[-1] if a else "ABSENT")
        return real_grid_1d(*a, **kw)

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(orch, "generate_and_fit_toys_python", spy_python)
        _fc_run(tmp_path / "scipy", _nuisance_penalty, strategy="scipy", n_toys=4)
    finally:
        monkey.undo()

    assert seen["python"], "the scipy path never called generate_and_fit_toys_python"
    assert all(f is _nuisance_penalty for f in seen["python"]), (
        f"the orchestrator did not forward extra_nll to the toy generator: {seen['python']}"
    )

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(orch, "generate_and_fit_toys_grid_1d", spy_grid_1d)
        _fc_run(tmp_path / "grid", _nuisance_penalty, strategy="grid", n_toys=4)
    finally:
        monkey.undo()

    assert seen["grid"], "the grid path never called generate_and_fit_toys_grid_1d"
    assert all(f is _nuisance_penalty for f in seen["grid"]), (
        f"the orchestrator did not forward extra_nll to the jitted toy generator: "
        f"{seen['grid']}"
    )


@njit(fastmath=True, nogil=True)
def _nuisance_penalty_at_truth(params):
    """Centred on the toys' own true params, so a correct implementation sees it cancel."""
    return ((params[1] - 1.0) / 0.05) ** 2


@njit(fastmath=True, nogil=True)
def _seed_numba_rng(s):
    """Numba keeps its own RNG state; np.random.seed() from Python does not touch it."""
    np.random.seed(s)


def test_penalty_reaches_the_JITTED_toy_fits_specifically():
    """
    The grid strategy's equivalent of the direct toy-fit test.

    `generate_and_fit_toys_grid_1d` is @njit(parallel=True) and reaches `calc_nll` only
    through the three grid fitters, so it has to forward `extra_nll` to them by hand. A
    mutation dropping that forwarding is invisible to every other test here: `t_data` comes
    from the DATA fits, which are unaffected, so the FC-level grid comparison still passes.

    Seeded through a jitted helper because numba's RNG is separate from NumPy's -- calling
    np.random.seed() from Python would leave the toys unmatched and the comparison
    meaningless.
    """
    from pyfc.binned import generate_and_fit_toys_grid_1d

    grids = [np.linspace(0.6, 1.4, 5), np.linspace(0.8, 1.2, 3)]
    full = np.array([[a, b] for a in grids[0] for b in grids[1]], dtype=np.float64)
    cond = np.array([[b] for b in grids[1]], dtype=np.float64)
    s2 = np.zeros(2)
    args = (1.0, 0, np.array([1.0, 1.0]), 2, full, cond, 12, s2, s2, False, _signal_rate)

    _seed_numba_rng(11)
    free = np.asarray(generate_and_fit_toys_grid_1d(*args), dtype=float)
    _seed_numba_rng(11)
    tied = np.asarray(generate_and_fit_toys_grid_1d(*args, _nuisance_penalty), dtype=float)

    assert free.shape == tied.shape == (12,)
    assert np.all(np.isfinite(tied)) and np.all(tied >= 0.0)
    assert not np.allclose(free, tied), (
        "the jitted toy generator produced identical statistics with and without the "
        "penalty from an identical RNG seed -- it is not forwarding extra_nll to the grid "
        "fitters, so t_critical and t_data would come from different likelihoods"
    )

    # "Differs" is not enough on its own. The generator forwards the penalty to TWO fits,
    # unconditional and conditional, and dropping it from just one still changes the
    # numbers -- so the assertion above passes against a half-wired implementation that
    # biases every toy statistic in one direction.
    #
    # This second check pins the balance between them. With the penalty centred on the
    # toys' own true parameters and the tested value there too, both fits pay the same
    # penalty and it cancels, so t stays small. Drop it from the unconditional fit alone
    # and that fit escapes to a lower NLL, inflating t = cond - uncond. Measured over five
    # seeds and BOTH numba versions available here (0.60 and 0.67): correct clusters at
    # 0.466-0.506, half-wired at 1.021-1.118. The 0.75 threshold sits in that gap with
    # roughly 50% margin either side.
    #
    # 400 toys, not 60. At 60 the median is noisy enough that the two populations overlap
    # across numba versions -- measured correct 0.357 / mutated 0.778 on 0.60, which the
    # threshold that worked on 0.67 would have let through. A statistical assertion whose
    # discriminating power depends on the compiler version is not a test, and the fix is
    # more samples rather than a threshold tuned to one machine.
    _seed_numba_rng(11)
    centred = np.asarray(
        generate_and_fit_toys_grid_1d(
            1.0, 0, np.array([1.0, 1.0]), 2, full, cond, 400, s2, s2, False,
            _signal_rate, _nuisance_penalty_at_truth),
        dtype=float)
    assert np.median(centred) < 0.75, (
        f"median toy statistic is {np.median(centred):.3f}; with the penalty centred on "
        "the toys' true parameters it should be well under 1, and an inflated value means "
        "the unconditional and conditional toy fits are not both paying it"
    )


@pytest.mark.skipif(
    not bool(_numba.config.DISABLE_JIT),
    reason="only meaningful when the JIT is genuinely disabled; monkeypatching the config "
           "cannot un-compile already-jitted fitters, so this runs under "
           "NUMBA_DISABLE_JIT=1 -- which scripts/run_coverage.sh and the CI Coverage job "
           "both set.",
)
def test_grid_guard_stands_down_when_the_jit_is_disabled(tmp_path):
    """
    Under NUMBA_DISABLE_JIT=1 the @njit decorator returns the plain Python function, so the
    CPUDispatcher check would reject a correct penalty -- and nothing is compiled in that
    mode anyway, so the requirement does not apply.

    Not hypothetical: `scripts/run_coverage.sh` sets that flag for most of its run, and
    this guard failed two tests there before the exemption existed. The CI Coverage job
    runs the same script, so without this the feature would have broken CI.
    """
    def plain_python_penalty(params):
        return ((params[1] - 1.15) / 0.05) ** 2

    # Must NOT raise: with the JIT off, a plain callable is exactly what @njit produces.
    results = _fc_run(tmp_path, plain_python_penalty, strategy="grid", n_toys=3)
    assert np.all(np.isfinite(results["1d_t_data_p1"]))
