"""
Tests for `pyfc.priors`, the builders for correlated and composite `extra_nll` priors.

WHY THIS FILE EXISTS: `extra_nll` could always express a correlated prior -- it receives
the full parameter vector and returns a scalar, so a pairwise Gaussian is just a quadratic
form with off-diagonal terms. What it could not do was stop you getting one of three
things wrong, each of which is silent:

  1. UNITS. `-2 ln L` means `d^T Sigma^-1 d`, no factor of 0.5. Wrong by sqrt(2), no symptom.
  2. STALENESS. Numba freezes the matrix at compile time, so editing a covariance after
     building the prior changes nothing and reports nothing.
  3. THE MATRIX ITSELF. A covariance that is not positive definite makes the penalty
     unbounded below, so the "constraint" pushes the fit away and the run still completes.

The oracle these tests are built on is exact and does triple duty. Profiling a Gaussian
returns the MARGINAL, so with a likelihood that says nothing about the parameters:

    t(x_i) = (x_i - c_i)^2 / Sigma_ii        (Sigma = COVARIANCE, not its inverse)

giving t == 1 at c_i + sqrt(Sigma_ii) whatever the correlation. That single assertion
fails if the off-diagonal terms are dropped (they change the marginal), if the units carry
a 0.5 (t comes out 0.5), and if the covariance is inverted one time too many or too few.
"""

import pickle

import numba as _numba
import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import (
    SCIPY_AVAILABLE,
    conditional_fit_1d_scipy,
    unconditional_fit_scipy,
)
from pyfc.orchestrator import compute_fc_intervals
from pyfc.priors import (
    MATMUL_THRESHOLD,
    combine_priors,
    gaussian_block,
    gaussian_block_from_correlation,
)

pytestmark = pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required")

N_OBS = np.array([50.0])
S2 = np.array([0.0])


@njit(fastmath=True, nogil=True)
def _flat_rate(params, S_sumw2, B_sumw2):
    """Rate independent of every parameter, so ONLY the prior constrains the fit.

    That isolation is what makes the marginal oracle exact rather than approximate: the
    data contributes nothing to the curvature, so the profile likelihood is the prior's.
    """
    return np.array([50.0]), S_sumw2


def _t_stat(prior, n_params, bounds, idx, value):
    """The profile-likelihood statistic PyFC itself forms: cond - uncond."""
    uncond, _ = unconditional_fit_scipy(
        N_OBS, n_params, bounds, _flat_rate, S_sumw2=S2, B_sumw2=S2, extra_nll=prior
    )
    cond, _ = conditional_fit_1d_scipy(
        value, idx, n_params, N_OBS, bounds, _flat_rate, S_sumw2=S2, B_sumw2=S2, extra_nll=prior
    )
    return cond - uncond


def _underlying(prior):
    """The Python function behind a built prior, whichever JIT mode we are in.

    With the JIT on, a builder returns a CPUDispatcher whose `py_func` is the function
    that was compiled. Under NUMBA_DISABLE_JIT=1 the @njit decorator returns that plain
    function directly and there is no `py_func` attribute at all.
    """
    return getattr(prior, "py_func", prior)


def _correlated_cov(sig_a, sig_b, rho):
    off = rho * sig_a * sig_b
    return np.array([[sig_a**2, off], [off, sig_b**2]])


# --- 1. The marginal oracle -----------------------------------------------------------

# rho is deliberately large. At rho = 0 the marginal and conditional widths coincide, so
# every test below would pass against a build that ignored the off-diagonal entirely.
RHO = 0.9
SIG_A, SIG_B = 0.10, 0.20
CENTRES_2 = [1.0, 2.0]
COV_2 = _correlated_cov(SIG_A, SIG_B, RHO)
BOUNDS_2 = [(0.0, 3.0), (0.5, 4.0)]


@pytest.mark.parametrize("i", [0, 1])
def test_pairwise_prior_reproduces_the_marginal_width(i):
    """
    t == 1 at c_i + sqrt(Sigma_ii), exactly, for a strongly correlated pair.

    This is the whole feature in one assertion. The marginal 1-sigma point is
    sqrt(Sigma_ii) -- a covariance element -- and NOT 1/sqrt((Sigma^-1)_ii), which at
    rho=0.9 is 2.3x smaller. A build that used the inverse's diagonal, dropped the
    off-diagonal, or carried a 0.5 all miss this number in different directions.
    """
    prior = gaussian_block([0, 1], CENTRES_2, COV_2)
    x = CENTRES_2[i] + np.sqrt(COV_2[i, i])
    t = _t_stat(prior, 2, BOUNDS_2, i, x)
    assert t == pytest.approx(1.0, abs=2e-3), (
        f"t={t} at the marginal 1-sigma point of parameter {i}. Expected exactly 1.0. "
        f"If this reads ~{1.0 / (1 - RHO**2):.2f} the off-diagonal terms are being "
        "dropped; if it reads ~0.5 the penalty carries the -lnL convention's 0.5."
    )


@pytest.mark.parametrize("z", [1.0, 2.0, 3.0])
def test_the_statistic_is_quadratic_in_the_marginal_width(z):
    """t == z^2 at c + z*sigma_marginal. Pins the shape, not just one point."""
    prior = gaussian_block([0, 1], CENTRES_2, COV_2)
    x = CENTRES_2[0] + z * np.sqrt(COV_2[0, 0])
    assert _t_stat(prior, 2, BOUNDS_2, 0, x) == pytest.approx(z**2, abs=5e-3)


def test_off_diagonal_terms_change_the_answer():
    """
    The correlated block and a diagonal-of-the-inverse build must DISAGREE.

    Without this, every oracle test above could be satisfied by a build that quietly
    ignored the off-diagonal, since it would still be *a* Gaussian. Here the two are
    computed on the same point and required to differ by the known factor 1/(1-rho^2).
    """
    prior = gaussian_block([0, 1], CENTRES_2, COV_2)
    inv = np.linalg.inv(COV_2)
    diag_only = np.diag(np.diag(inv))
    naive = gaussian_block([0, 1], CENTRES_2, np.linalg.inv(diag_only))

    x = CENTRES_2[0] + np.sqrt(COV_2[0, 0])
    t_full = _t_stat(prior, 2, BOUNDS_2, 0, x)
    t_naive = _t_stat(naive, 2, BOUNDS_2, 0, x)

    assert t_full == pytest.approx(1.0, abs=2e-3)
    assert t_naive == pytest.approx(1.0 / (1 - RHO**2), rel=5e-3), (
        f"the diagonal-of-inverse build gave {t_naive}; at rho={RHO} it must give "
        f"{1 / (1 - RHO**2):.4f}. If it gives 1.0 the two builds are not actually different "
        "and this test is not discriminating anything."
    )


def test_units_carry_no_factor_of_one_half():
    """A width-s Gaussian gives t=1 at c+s. The -lnL convention would give 0.5."""
    prior = gaussian_block([0], [1.2], [[0.05**2]])
    t = _t_stat(prior, 1, [(0.5, 2.0)], 0, 1.2 + 0.05)
    assert t == pytest.approx(1.0, abs=2e-3), (
        f"t={t} one sigma out. PyFC's NLL is -2lnL; 0.5 here means every constraint "
        "in the analysis is weaker by sqrt(2)."
    )


def test_dense_thirteen_parameter_block_matches_the_oracle():
    """Scale: a full 13x13 covariance with no zero entries, four parameters checked."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(13, 13))
    cov = (a @ a.T) / 40 + np.eye(13) * 0.02
    centres = np.ones(13)
    prior = gaussian_block(np.arange(13), centres, cov)
    bounds = [(-3.0, 5.0)] * 13

    for i in (0, 3, 7, 12):
        x = centres[i] + np.sqrt(cov[i, i])
        t = _t_stat(prior, 13, bounds, i, x)
        assert t == pytest.approx(1.0, abs=1e-3), f"parameter {i} gave t={t}"


# --- 2. The two formulations --------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5, 13])
def test_explicit_and_matmul_branches_agree(n):
    """
    Both branches must be the same function, or the threshold is a correctness switch
    rather than a speed one. Checked on random points, not just the centre, since at the
    centre every formulation returns zero.
    """
    rng = np.random.default_rng(n)
    a = rng.normal(size=(n, n))
    cov = (a @ a.T) / 10 + np.eye(n) * 0.05
    centres = rng.normal(size=n)
    idx = np.arange(n)

    explicit = gaussian_block(idx, centres, cov, threshold=n + 1)   # n < threshold -> loop
    matmul = gaussian_block(idx, centres, cov, threshold=0)         # n >= threshold -> matmul

    for _ in range(20):
        p = rng.normal(size=n) * 0.5 + centres
        assert explicit(p) == pytest.approx(matmul(p), rel=1e-12, abs=1e-12)


def test_threshold_actually_selects_different_code():
    """
    The two branches must be structurally different, not just numerically equal.

    Without this, `test_explicit_and_matmul_branches_agree` would pass trivially if the
    threshold were ignored and both calls returned the same implementation -- which is
    precisely the mutation it is meant to catch.
    """
    cov = np.eye(4) * 0.01
    loop = gaussian_block(np.arange(4), np.zeros(4), cov, threshold=99)
    mm = gaussian_block(np.arange(4), np.zeros(4), cov, threshold=0)
    assert "range" in _underlying(loop).__code__.co_names
    assert "range" not in _underlying(mm).__code__.co_names


def test_the_frozen_inverse_is_exactly_symmetric():
    """
    `np.linalg.inv` of a symmetric matrix is symmetric only up to rounding, and the
    explicit-loop branch reads the UPPER TRIANGLE ONLY -- so an asymmetric inverse would
    make the two formulations compute genuinely different functions.

    Pinned here on the helper rather than through a quadratic form on purpose. The
    downstream effect is ~1e-10 relative even at condition number 1e9, which is below
    PyFC's own ~2e-15-and-growing reproducibility floor under fastmath; a test asserting
    it through the fitted result would be pinning noise and would reorder with the
    compiler. Exact symmetry of the frozen matrix is deterministic and is the actual
    property the code establishes.
    """
    from pyfc.priors import _validate_block

    rng = np.random.default_rng(0)
    n = 6
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    cov = q @ np.diag(np.geomspace(1.0, 1e-9, n)) @ q.T
    cov = 0.5 * (cov + cov.T)
    assert np.linalg.cond(cov) > 1e8, "the fixture stopped being ill-conditioned"
    assert not np.array_equal(np.linalg.inv(cov), np.linalg.inv(cov).T), (
        "raw inv() came back exactly symmetric, so this fixture no longer exercises "
        "the symmetrisation at all"
    )

    _, _, inv = _validate_block(np.arange(n), np.zeros(n), cov)
    assert np.array_equal(inv, inv.T), "the frozen inverse is not exactly symmetric"


def test_default_threshold_is_the_measured_crossover():
    """A regression pin: the default came from a benchmark, not from taste."""
    assert 30 <= MATMUL_THRESHOLD <= 40


# --- 3. The freezing trap ------------------------------------------------------------

def test_the_builder_snapshots_the_covariance():
    """
    Mutating the caller's covariance after building must NOT change the prior.

    Numba freezes a closed-over array by value at compile time, so a jitted prior written
    by hand ignores later edits -- measured for `M[i,j]=x`, `M*=x` and `M[:]=new` alike.
    Doing the inversion at build time makes that behaviour explicit and total rather than
    a surprise that depends on when compilation happened to be triggered.
    """
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    prior = gaussian_block([0, 1], [1.0, 1.0], cov)
    x = np.array([1.2, 1.0])
    before = prior(x)

    cov[0, 0] = 1e-6              # a 200x tighter constraint, if it were picked up
    assert prior(x) == pytest.approx(before, rel=0, abs=0), (
        "the prior tracked a later mutation of the caller's array; the snapshot is the "
        "documented contract and the only behaviour numba can deliver consistently."
    )

    rebuilt = gaussian_block([0, 1], [1.0, 1.0], cov)
    assert rebuilt(x) > before * 100, (
        "rebuilding did not pick up the new covariance, so the builder is caching "
        "something it should not."
    )


# --- 4. Validation -------------------------------------------------------------------

@pytest.mark.parametrize(
    "indices, centres, cov, fragment",
    [
        ([0, 1], [1.0, 1.0], np.zeros((2, 3)), "square"),
        ([0, 1], [1.0, 1.0], np.eye(3), "2 parameters"),
        ([0, 1], [1.0, 1.0], np.array([[1.0, 0.5], [0.2, 1.0]]), "symmetric"),
        ([0, 1], [1.0, 1.0], np.array([[1.0, 1.0], [1.0, 1.0]]), "positive definite"),
        ([0, 0], [1.0, 1.0], np.eye(2), "duplicates"),
        ([-1, 1], [1.0, 1.0], np.eye(2), "non-negative"),
        ([0, 1], [1.0, 1.0, 1.0], np.eye(2), "must match"),
        ([], [], np.zeros((0, 0)), "empty"),
    ],
)
def test_bad_blocks_are_rejected_with_an_actionable_message(indices, centres, cov, fragment):
    with pytest.raises(ValueError, match=fragment):
        gaussian_block(indices, centres, cov)


@pytest.mark.parametrize(
    "sig_a, sig_b",
    [(0.2, 0.3), (0.1, 0.2), (0.05, 0.5), (1.0, 1.0), (0.046, 0.11), (2.0, 3.0)],
)
def test_a_singular_covariance_is_rejected_rather_than_inverted(sig_a, sig_b):
    """
    The most dangerous input, because `np.linalg.inv` may not raise on it.

    A non-positive-definite covariance makes `d^T Sigma^-1 d` unbounded below along some
    direction, so the term pushes the fit AWAY. Nothing downstream notices: the run
    completes and the interval looks ordinary.

    Parametrised over several widths on purpose. `np.linalg.cholesky` alone is NOT a
    reliable test here: at sigma=(0.2, 0.3) with rho exactly 1, `0.04*0.09` and `0.06**2`
    differ in the last bit, the determinant comes out at +1e-18, and cholesky accepts the
    matrix. Measured, it accepted 1 of these 8-ish cases -- so a single fixture passes or
    fails on rounding luck, which is how this slipped through in the first place.
    """
    with pytest.raises(ValueError, match="positive definite"):
        gaussian_block([0, 1], [1.0, 1.0], _correlated_cov(sig_a, sig_b, 1.0))


@pytest.mark.parametrize("condition", [1e6, 1e9, 1e12])
def test_an_ill_conditioned_but_usable_covariance_is_accepted(condition):
    """
    The rank tolerance must not become a condition-number policy.

    Real covariances from a tightly-constrained fit are legitimately ill-conditioned, and
    rejecting them would turn a guard against a genuine error into an obstacle. Only
    rank-deficiency is refused.
    """
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    cov = q @ np.diag(np.geomspace(1.0, 1.0 / condition, 6)) @ q.T
    cov = 0.5 * (cov + cov.T)
    prior = gaussian_block(np.arange(6), np.zeros(6), cov)
    assert np.isfinite(prior(np.full(6, 0.01)))


# --- 5. from_correlation --------------------------------------------------------------

def test_from_correlation_matches_the_explicit_covariance():
    sig = np.array([0.046, 0.11, 0.35])
    corr = np.array([[1.0, 0.4, -0.2], [0.4, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    centres = np.array([1.0, 1.0, 1.0])

    a = gaussian_block_from_correlation([0, 1, 2], centres, sig, corr)
    b = gaussian_block([0, 1, 2], centres, np.outer(sig, sig) * corr)

    rng = np.random.default_rng(3)
    for _ in range(10):
        p = centres + rng.normal(scale=0.1, size=3)
        assert a(p) == pytest.approx(b(p), rel=1e-12)


def test_from_correlation_rejects_a_covariance_passed_by_mistake():
    """Unit diagonal is the one cheap way to tell the two matrices apart."""
    sig = np.array([0.2, 0.3])
    cov = _correlated_cov(0.2, 0.3, 0.5)
    with pytest.raises(ValueError, match="unit diagonal"):
        gaussian_block_from_correlation([0, 1], [1.0, 1.0], sig, cov)


@pytest.mark.parametrize("sigmas", [[0.2, 0.0], [0.2, -0.3]])
def test_from_correlation_rejects_a_non_positive_sigma(sigmas):
    """
    A zero or negative sigma is not a tighter constraint, it is a broken one.

    Zero would divide by zero building the covariance and yield a singular matrix; a
    negative one squares away silently and gives the same covariance as its absolute
    value, so the caller's sign error would never surface.
    """
    with pytest.raises(ValueError, match="positive"):
        gaussian_block_from_correlation([0, 1], [1.0, 1.0], sigmas, [[1.0, 0.3], [0.3, 1.0]])


def test_from_correlation_rejects_a_correlation_of_the_wrong_shape():
    """A 3x3 correlation with two sigmas is a mismatched pair of sources, not a typo."""
    with pytest.raises(ValueError, match="to match"):
        gaussian_block_from_correlation([0, 1], [1.0, 1.0], [0.2, 0.3], np.eye(3))


def test_from_correlation_rejects_an_out_of_range_entry():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        gaussian_block_from_correlation(
            [0, 1], [1.0, 1.0], [0.2, 0.3], [[1.0, 1.4], [1.4, 1.0]]
        )


# --- 6. Composition -------------------------------------------------------------------

def test_combine_is_additive_over_three_blocks():
    cov = _correlated_cov(0.2, 0.3, 0.3)
    blocks = [gaussian_block([2 * k, 2 * k + 1], [1.0, 1.0], cov) for k in range(3)]
    total = combine_priors(*blocks)

    rng = np.random.default_rng(11)
    for _ in range(10):
        p = np.ones(6) + rng.normal(scale=0.1, size=6)
        assert total(p) == pytest.approx(sum(b(p) for b in blocks), rel=1e-12)


def test_combine_of_one_prior_is_that_prior():
    prior = gaussian_block([0], [1.0], [[0.01]])
    p = np.array([1.05])
    assert combine_priors(prior)(p) == pytest.approx(prior(p), rel=0, abs=0)


def test_combine_rejects_an_empty_call():
    with pytest.raises(ValueError, match="at least one"):
        combine_priors()


@pytest.mark.skipif(
    bool(_numba.config.DISABLE_JIT),
    reason="with NUMBA_DISABLE_JIT=1 every @njit returns a plain function, so a "
           "CPUDispatcher check describes nothing. scripts/run_coverage.sh sets that flag.",
)
def test_combined_jitted_priors_stay_jitted_for_the_grid_strategy():
    """
    Composition must not quietly cost you `strategy="grid"`.

    The orchestrator rejects a non-jitted `extra_nll` under `"grid"`, so a `combine_priors` that
    returned a plain Python wrapper would turn a working configuration into a TypeError
    the moment a second constraint was added.
    """
    from numba.core.registry import CPUDispatcher

    a = gaussian_block([0], [1.0], [[0.01]])
    b = gaussian_block([1], [1.0], [[0.04]])
    assert isinstance(combine_priors(a, b), CPUDispatcher)


@pytest.mark.skipif(
    bool(_numba.config.DISABLE_JIT),
    reason="the jitted/plain distinction does not exist under NUMBA_DISABLE_JIT=1",
)
def test_combining_with_a_plain_callable_degrades_to_a_plain_callable():
    """Mixing in a Python function is allowed; it just gives up the grid strategy."""
    from numba.core.registry import CPUDispatcher

    jitted = gaussian_block([0], [1.0], [[0.01]])

    def plain(params):
        return float(params[1] ** 2)

    mixed = combine_priors(jitted, plain)
    assert not isinstance(mixed, CPUDispatcher)
    p = np.array([1.1, 2.0])
    assert mixed(p) == pytest.approx(jitted(p) + plain(p))


# --- 7. Integration with the construction ---------------------------------------------

def _fc_run(tmp_path, prior, strategy, n_toys=20):
    np.random.seed(5)
    data = np.array([22.0, 14.0])
    s2 = np.zeros(2)
    grids = [np.linspace(0.7, 1.3, 5), np.linspace(0.7, 1.3, 3)]

    @njit(fastmath=True, nogil=True)
    def rate(params, S_sumw2, B_sumw2):
        return np.array([12.0 * params[0] + 6.0 * params[1],
                         5.0 * params[0] + 9.0 * params[1]]), S_sumw2

    results, _ = compute_fc_intervals(
        data=data, grids=grids, compute_rates_func=rate, cl=[0.90], n_toys=n_toys,
        strategy=strategy, num_cores=1, verbose=0, sparsify_grid=False, warm_start=False,
        likelihood_type="binned", S_sumw2=s2, B_sumw2=s2,
        use_finite_mc_correction_binned=False, output_file=None,
        save_directory=str(tmp_path), compute_1D_intervals=True,
        compute_2D_intervals=False, extra_nll=prior,
    )
    return results


@pytest.mark.parametrize("strategy", ["scipy", "grid"])
def test_a_correlated_block_drives_the_full_construction(tmp_path, strategy):
    """
    End to end, including `strategy="grid"` -- the path that runs inside @njit code and
    rejects a plain Python callable. A builder that returned something un-jittable would
    fail here and nowhere else.
    """
    prior = gaussian_block([0, 1], [1.0, 1.0], _correlated_cov(0.15, 0.25, 0.6))
    results = _fc_run(tmp_path, prior, strategy)
    t_data = np.asarray(results["1d_t_data_p1"], dtype=float)
    assert np.all(np.isfinite(t_data))
    assert np.any(t_data > 0), "every t is zero, so the scan did not discriminate anything"


def test_a_correlated_prior_tightens_the_interval(tmp_path):
    """
    The prior must do statistical work, not merely run.

    Asserting only finiteness above would pass against a builder that returned a constant.
    A correlated constraint on both parameters removes freedom the profile fit would
    otherwise use, so the observed statistic at off-best-fit points must RISE.
    """
    tight = gaussian_block([0, 1], [1.0, 1.0], _correlated_cov(0.03, 0.03, 0.5))
    loose = gaussian_block([0, 1], [1.0, 1.0], _correlated_cov(3.0, 3.0, 0.5))

    t_tight = np.asarray(_fc_run(tmp_path, tight, "scipy")["1d_t_data_p1"], dtype=float)
    t_loose = np.asarray(_fc_run(tmp_path, loose, "scipy")["1d_t_data_p1"], dtype=float)

    assert t_tight.sum() > t_loose.sum() * 1.5, (
        f"tight prior gave sum(t)={t_tight.sum():.3f} against {t_loose.sum():.3f} for a "
        "prior 100x wider; the constraint is not reaching the fits."
    )


@pytest.mark.skipif(
    bool(_numba.config.DISABLE_JIT),
    reason="picklability here is numba's -- a CPUDispatcher implements __reduce__, while the "
           "plain closure @njit returns under NUMBA_DISABLE_JIT=1 cannot be pickled by "
           "definition. The mode is for coverage measurement, not for running analyses, so "
           "there is nothing to preserve in it.",
)
def test_a_built_prior_survives_pickling():
    """
    Unbinned toys go to a ProcessPoolExecutor, which pickles its arguments.

    A builder returning a closure that cannot be pickled would still work -- PyFC falls
    back to threads -- but silently lose the process pool, which is a performance cliff
    nobody would connect to the prior.
    """
    prior = gaussian_block([0, 1], [1.0, 1.0], _correlated_cov(0.1, 0.2, 0.4))
    revived = pickle.loads(pickle.dumps(prior))
    p = np.array([1.1, 2.1])
    assert revived(p) == pytest.approx(prior(p), rel=0, abs=0)
