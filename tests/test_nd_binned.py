"""
Tests for N-dimensional binned data support (arbitrary-shape N_obs/mu/sigma2).
"""
import numpy as np
from numba import njit

from pyfc.binned import calc_nll, unconditional_fit_grid
from pyfc.orchestrator import compute_fc_intervals


@njit(fastmath=True, nogil=True)
def _compute_rates_nd(params, S_sumw2, B_sumw2):
    mu = params[0] * np.ones_like(S_sumw2) + params[1]
    return mu, S_sumw2


def test_calc_nll_2d_matches_flattened_1d_standard_poisson():
    """
    A genuinely 2D (E, cos_theta)-shaped histogram must give a byte-identical
    NLL to the same data flattened to 1D, for the standard Poisson likelihood.
    """
    params = np.array([2.0, 1.0])

    N_obs_2d = np.array([[5.0, 3.0, 9.0], [8.0, 2.0, 4.0]])
    S_sumw2_2d = np.zeros((2, 3))
    B_sumw2_2d = np.zeros((2, 3))

    nll_2d = calc_nll(params, N_obs_2d, S_sumw2_2d, B_sumw2_2d, False, _compute_rates_nd)
    nll_1d = calc_nll(
        params,
        N_obs_2d.reshape(-1),
        S_sumw2_2d.reshape(-1),
        B_sumw2_2d.reshape(-1),
        False,
        _compute_rates_nd,
    )

    assert nll_2d == nll_1d


def test_calc_nll_2d_matches_flattened_1d_finite_mc():
    """
    Same flatten-invariance property, but exercising the finite-MC
    (Poisson-Gamma mixture) branch, which reads sigma2 per-bin.
    """
    params = np.array([2.0, 1.0])

    N_obs_2d = np.array([[5.0, 3.0, 9.0], [8.0, 2.0, 4.0]])
    S_sumw2_2d = np.full((2, 3), 0.5)
    B_sumw2_2d = np.zeros((2, 3))

    nll_2d = calc_nll(params, N_obs_2d, S_sumw2_2d, B_sumw2_2d, True, _compute_rates_nd)
    nll_1d = calc_nll(
        params,
        N_obs_2d.reshape(-1),
        S_sumw2_2d.reshape(-1),
        B_sumw2_2d.reshape(-1),
        True,
        _compute_rates_nd,
    )

    assert nll_2d == nll_1d


def test_calc_nll_raises_on_mu_shape_mismatch():
    """
    compute_rates_func returning a `mu` with a different bin count than
    N_obs must raise a clear ValueError instead of an obscure indexing crash.
    """
    params = np.array([2.0, 1.0])
    N_obs = np.array([1.0, 2.0, 3.0])
    S_sumw2 = np.zeros((2, 2))  # deliberately mismatched bin count (4 vs 3)
    B_sumw2 = np.zeros((2, 2))

    try:
        calc_nll(params, N_obs, S_sumw2, B_sumw2, False, _compute_rates_nd)
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_unconditional_fit_grid_2d_matches_flattened_1d():
    """
    The grid-search optimizer itself (not just calc_nll) must be agnostic
    to whether N_obs is passed as a 2D histogram or its 1D flattening.
    """
    N_obs_2d = np.array([[5.0, 3.0], [8.0, 2.0]])
    S_sumw2_2d = np.zeros((2, 2))
    B_sumw2_2d = np.zeros((2, 2))

    grid_a = np.linspace(0.0, 5.0, 6)
    grid_b = np.linspace(0.0, 5.0, 6)
    full_grid_points = np.array([[a, b] for a in grid_a for b in grid_b])

    min_nll_2d, best_2d = unconditional_fit_grid(
        N_obs_2d, full_grid_points, S_sumw2_2d, B_sumw2_2d, False, _compute_rates_nd
    )
    min_nll_1d, best_1d = unconditional_fit_grid(
        N_obs_2d.reshape(-1), full_grid_points, S_sumw2_2d.reshape(-1), B_sumw2_2d.reshape(-1),
        False, _compute_rates_nd
    )

    assert min_nll_2d == min_nll_1d
    assert np.array_equal(best_2d, best_1d)


def test_compute_fc_intervals_accepts_genuine_2d_histogram(tmp_path):
    """
    Smoke test: a genuine 2D (e.g. (E_bins, cos_theta_bins)) histogram must
    flow correctly through a full compute_fc_intervals call, and produce the
    same unconditional NLL as the same physical model evaluated on the
    1D-flattened equivalent data (small grid/n_toys per the performance
    guidance for this kind of regression check).
    """
    S_template_2d = np.array([[1.0, 2.0], [3.0, 1.5]])
    B_template_2d = np.array([[2.0, 1.0], [1.0, 2.5]])

    @njit(fastmath=True, nogil=True)
    def compute_rates_2d(params, S_sumw2, B_sumw2):
        mu = params[0] * S_template_2d + params[1] * B_template_2d
        return mu, S_sumw2

    np.random.seed(1)
    data_2d = np.random.poisson(1.0 * S_template_2d + 1.0 * B_template_2d).astype(float)
    grids = [np.linspace(0.2, 2.0, 4), np.linspace(0.2, 2.0, 4)]

    results_2d, _ = compute_fc_intervals(
        data=data_2d, grids=grids, compute_rates_func=compute_rates_2d,
        cl=[0.68], n_toys=10, strategy="scipy", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        compute_1D_intervals=True, compute_2D_intervals=False,
        save_directory=str(tmp_path / "fc_2d"),
    )

    S_template_1d = S_template_2d.reshape(-1)
    B_template_1d = B_template_2d.reshape(-1)

    @njit(fastmath=True, nogil=True)
    def compute_rates_1d(params, S_sumw2, B_sumw2):
        mu = params[0] * S_template_1d + params[1] * B_template_1d
        return mu, S_sumw2

    results_1d, _ = compute_fc_intervals(
        data=data_2d.reshape(-1), grids=grids, compute_rates_func=compute_rates_1d,
        cl=[0.68], n_toys=10, strategy="scipy", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        compute_1D_intervals=True, compute_2D_intervals=False,
        save_directory=str(tmp_path / "fc_1d"),
    )

    assert np.isclose(results_2d["data_uncond_nll"], results_1d["data_uncond_nll"])
