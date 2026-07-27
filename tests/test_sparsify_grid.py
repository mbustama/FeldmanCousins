"""
Regression tests for the sparsify_grid=True boundary-refinement bug.

eval_2d_point's memoization guard used to check 2d_t_critical for NaN-ness.
But the coarse-to-fine RectBivariateSpline interpolation step fills in
2d_t_critical for *every* cell (real or not) before the refinement pass
runs, so the guard always fired and the refinement pass was a silent
no-op: every non-coarse cell's 2d_t_data stayed NaN forever, and
`accepted = t_data <= t_critical` is always False against NaN, so those
cells were force-excluded from the accepted region regardless of their
true status. The guard now keys off 2d_t_data instead, which is only ever
set by an actual conditional fit and untouched by the interpolation step.
"""
import numpy as np
from numba import njit

from pyfc.orchestrator import compute_fc_intervals

S_TEMPLATE = np.array([0.1, 0.5, 2.0, 5.0])
B_TEMPLATE = np.array([15.0, 5.0, 1.0, 0.1])


@njit(fastmath=True, nogil=True)
def _compute_rates(params, S_sumw2, B_sumw2):
    mu = params[0] * S_TEMPLATE + params[1] + params[2] * B_TEMPLATE
    sigma2 = (params[0] ** 2) * S_sumw2 + (params[2] ** 2) * B_sumw2
    return mu, sigma2


def _coarse_cell_count(n_grid):
    """Mirrors orchestrator.py's own coarse-index selection for one axis."""
    step = max(1, n_grid // 5)
    coarse_idx = sorted(set(list(range(0, n_grid, step)) + [n_grid - 1]))
    return len(coarse_idx)


def test_sparsify_grid_refinement_evaluates_cells_beyond_the_coarse_subset(tmp_path):
    np.random.seed(42)
    n_data = np.random.poisson(1.0 * S_TEMPLATE + 1.0 + 1.0 * B_TEMPLATE)

    n_grid = 16
    grids = [
        np.linspace(0.5, 2.0, n_grid),
        np.linspace(0.5, 2.0, n_grid),
        np.linspace(0.5, 1.5, 5),
    ]

    results, _ = compute_fc_intervals(
        data=n_data, grids=grids, compute_rates_func=_compute_rates,
        cl=[0.90], n_toys=20, strategy="grid", num_cores=1, verbose=0,
        sparsify_grid=True, warm_start=False, likelihood_type="binned",
        S_sumw2=S_TEMPLATE.copy(), B_sumw2=B_TEMPLATE.copy(),
        compute_1D_intervals=False, compute_2D_intervals=True,
        save_directory=str(tmp_path / "fc_sparse"),
    )

    t_data = results["2d_t_data_p1p2"]
    n_evaluated = int(np.sum(~np.isnan(t_data)))
    n_coarse = _coarse_cell_count(n_grid) ** 2

    assert n_evaluated > n_coarse, (
        f"Only the {n_coarse} coarse cells were ever evaluated "
        f"({n_evaluated} total) -- the boundary-refinement pass regressed "
        "back into a silent no-op."
    )

    # Sanity: every evaluated cell's t_critical must be a real (non-NaN)
    # value, i.e. eval_2d_point ran both steps (data fit + toys) together.
    t_critical = results["2d_t_critical_p1p2"][0.90]
    assert np.all(~np.isnan(t_critical[~np.isnan(t_data)]))


def test_sparsify_grid_true_matches_false_on_actually_evaluated_cells(tmp_path):
    """
    sparsify_grid's known coverage limitation (see the docstring/README
    caveat) means True and False won't cover every cell alike on larger
    grids, but wherever sparsify_grid=True *did* run an exact conditional
    fit for a cell, it must produce the same 2d_t_data value a full
    sparsify_grid=False scan would -- this checks eval_2d_point's own
    per-cell fit logic is unaffected by the guard fix, independent of any
    Monte Carlo toy noise (t_data is a deterministic fit on the real data,
    so n_toys=1 is enough here).
    """
    np.random.seed(7)
    n_data = np.random.poisson(1.0 * S_TEMPLATE + 1.0 + 1.0 * B_TEMPLATE)

    n_grid = 12
    grids = [
        np.linspace(0.5, 2.0, n_grid),
        np.linspace(0.5, 2.0, n_grid),
        np.linspace(0.5, 1.5, 5),
    ]

    results_sparse, _ = compute_fc_intervals(
        data=n_data, grids=grids, compute_rates_func=_compute_rates,
        cl=[0.90], n_toys=1, strategy="grid", num_cores=1, verbose=0,
        sparsify_grid=True, warm_start=False, likelihood_type="binned",
        S_sumw2=S_TEMPLATE.copy(), B_sumw2=B_TEMPLATE.copy(),
        compute_1D_intervals=False, compute_2D_intervals=True,
        save_directory=str(tmp_path / "fc_sparse2"),
    )

    results_full, _ = compute_fc_intervals(
        data=n_data, grids=grids, compute_rates_func=_compute_rates,
        cl=[0.90], n_toys=1, strategy="grid", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        S_sumw2=S_TEMPLATE.copy(), B_sumw2=B_TEMPLATE.copy(),
        compute_1D_intervals=False, compute_2D_intervals=True,
        save_directory=str(tmp_path / "fc_full2"),
    )

    evaluated_mask = ~np.isnan(results_sparse["2d_t_data_p1p2"])
    assert evaluated_mask.sum() > _coarse_cell_count(n_grid) ** 2

    t_data_sparse = results_sparse["2d_t_data_p1p2"]
    t_data_full = results_full["2d_t_data_p1p2"]
    assert np.allclose(t_data_sparse[evaluated_mask], t_data_full[evaluated_mask])
