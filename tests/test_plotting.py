"""
Tests for `plotting.generate_corner_plot`. Previously had zero coverage of
its own -- it's exercised incidentally by any end-to-end
`compute_fc_intervals` call (which calls it internally to produce the
default plot), but nothing asserted anything about the plot itself, so a
crash or a wrong axes/data mapping inside it could go unnoticed as long as
`compute_fc_intervals`'s own numeric results stayed correct. `smooth_1d`/
`smooth_2d` were consequently also untested (easy to confuse with the
unrelated `tests/test_smoothing.py`, which tests the NLL barrier, not the
plotting smoothing toggles).
"""
import numpy as np
import pytest
from numba import njit

from pyfc.orchestrator import compute_fc_intervals
from pyfc.plotting import MATPLOTLIB_AVAILABLE, generate_corner_plot

pytestmark = pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib is required for these tests")


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """
    generate_corner_plot deliberately does not close its own figure (see
    its docstring/test above) -- the caller owns it. Close whatever this
    test session accumulated so repeated runs don't trip matplotlib's
    "more than 20 open figures" warning.
    """
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _make_1d2d_results(n_params=2, cl=(0.9,)):
    """
    Hand-built `results` dict matching the exact keys/structure
    generate_corner_plot reads -- note this is the in-memory structure
    (nested dicts keyed by CL), not the flattened `_{cl}`-suffixed key
    naming `_save_fc_archive` uses when exporting to `.npz`.
    """
    results = {}
    grids = [np.linspace(0.0, 1.0, 5) for _ in range(n_params)]
    for i, grid in enumerate(grids):
        results[f"grid_p{i+1}"] = grid
        results[f"1d_test_p{i+1}"] = grid
        t_data = np.abs(np.sin(grid * 3))
        results[f"1d_t_data_p{i+1}"] = t_data
        results[f"1d_t_critical_p{i+1}"] = {c: np.full_like(grid, 0.5) for c in cl}
        results[f"1d_accepted_p{i+1}"] = {c: t_data <= 0.5 for c in cl}

    if n_params > 1:
        for i, j in [(a, b) for a in range(n_params) for b in range(n_params) if a < b]:
            pair_name = f"p{i+1}p{j+1}"
            gA, gB = grids[i], grids[j]
            data2d = np.abs(np.outer(np.sin(gA * 3), np.cos(gB * 3)))
            results[f"2d_t_data_{pair_name}"] = data2d
            results[f"2d_t_critical_{pair_name}"] = {c: np.full_like(data2d, 0.5) for c in cl}
            results[f"2d_accepted_{pair_name}"] = {c: data2d <= 0.5 for c in cl}

    return results, grids


def _make_config(n_params, save_directory, cl=(0.9,), smooth_1d=True, smooth_2d=True):
    return {
        "n_params": n_params,
        "param_names": [f"param{i+1}" for i in range(n_params)],
        "cl": list(cl),
        "smooth_1d": smooth_1d,
        "smooth_2d": smooth_2d,
        "save_directory": str(save_directory),
        "compute_1D_intervals": True,
        "compute_2D_intervals": n_params > 1,
    }


def test_generate_corner_plot_returns_correctly_shaped_figure(tmp_path):
    results, _ = _make_1d2d_results(n_params=3)
    config = _make_config(3, tmp_path)

    fig = generate_corner_plot(results, config)

    assert fig is not None
    axs = fig.get_axes()
    # Lower-triangle layout: n_params^2 Axes are created (upper triangle hidden, not removed).
    assert len(axs) == 3 * 3


def test_generate_corner_plot_writes_pdf_to_save_directory(tmp_path):
    results, _ = _make_1d2d_results(n_params=2)
    config = _make_config(2, tmp_path)

    generate_corner_plot(results, config)

    assert (tmp_path / "fc_corner_plot.pdf").exists()


def test_generate_corner_plot_single_param_handles_scalar_axes(tmp_path):
    """
    matplotlib's plt.subplots(1, 1, ...) returns a single Axes object, not a
    2D array -- generate_corner_plot has a special case for this (n_params=1).
    """
    results, _ = _make_1d2d_results(n_params=1)
    config = _make_config(1, tmp_path)

    fig = generate_corner_plot(results, config)

    assert fig is not None
    assert len(fig.get_axes()) == 1


@pytest.mark.parametrize("smooth_1d,smooth_2d", [(True, True), (False, False), (True, False), (False, True)])
def test_generate_corner_plot_smoothing_toggles_do_not_crash(tmp_path, smooth_1d, smooth_2d):
    results, _ = _make_1d2d_results(n_params=2)
    config = _make_config(2, tmp_path, smooth_1d=smooth_1d, smooth_2d=smooth_2d)

    fig = generate_corner_plot(results, config)

    assert fig is not None


def test_generate_corner_plot_caller_owns_the_figure_not_closed_internally(tmp_path):
    """
    Regression test for the docstring bug: generate_corner_plot's docstring
    used to claim it returns None and closes the matplotlib figure to free
    memory. It actually returns the live Figure and never closes it -- the
    figure must still be open (not closed) right after the call returns.
    """
    import matplotlib.pyplot as plt

    results, _ = _make_1d2d_results(n_params=2)
    config = _make_config(2, tmp_path)

    fig = generate_corner_plot(results, config)

    assert fig is not None
    assert plt.fignum_exists(fig.number)
    plt.close(fig)


def test_compute_fc_intervals_end_to_end_returns_live_figure(tmp_path):
    """Realistic integration coverage: the plot compute_fc_intervals itself produces."""
    S_template = np.array([0.1, 0.5, 2.0, 5.0])
    B_template = np.array([15.0, 5.0, 1.0, 0.1])
    s2 = np.zeros_like(S_template)

    @njit(fastmath=True, nogil=True)
    def rate_func(params, S_s2, B_s2):
        mu = params[0] * S_template + params[1] * B_template
        return mu, S_s2

    N_data = np.array([3, 6, 4, 1])
    grids = [np.linspace(0.5, 2.0, 3), np.linspace(0.5, 2.0, 3)]

    results, fig = compute_fc_intervals(
        data=N_data, grids=grids,
        compute_rates_func=rate_func,
        cl=[0.90], n_toys=5, strategy="scipy", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="binned", S_sumw2=s2, B_sumw2=s2,
        output_file=None, save_directory=str(tmp_path),
        compute_1D_intervals=True, compute_2D_intervals=True,
    )

    assert fig is not None
    assert len(fig.get_axes()) == 2 * 2
    assert (tmp_path / "fc_corner_plot.pdf").exists()
