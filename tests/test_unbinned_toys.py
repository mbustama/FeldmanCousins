"""
Tests for the unbinned toy-generation/fitting pipeline in
`toys.generate_and_fit_toys_python`'s "unbinned" branch (`generate_toy_func`,
`S_mc_pool`/`B_mc_pool`, `_worker_unbinned_toy`, dispatched through a real
`ProcessPoolExecutor`).

Previously untested: `tests/test_pdf_components.py` only covers the NLL
formula and grid-search fitting, and `tests/test_multiprocessing.py` only
exercises a mock worker function, not PyFC's actual unbinned toy pipeline.
Only the NLL formula's correctness was covered, not that the empirical
critical-value machinery (interval construction) actually works for
unbinned models.

Uses ProcessPoolExecutor under the hood (see `toys.py`), so every
callable passed in below must be a top-level, picklable function -- not a
closure/lambda defined inside a test function -- matching this file's own
module-level convention and the note in `test_multiprocessing.py`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyfc.orchestrator import compute_fc_intervals


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """compute_fc_intervals's default plotting leaves its figure open (caller-owned)."""
    yield
    plt.close("all")


def _s_pdf(x):
    return np.exp(-0.5 * (x - 5.0) ** 2)


def _b_pdf(x):
    return np.full_like(x, 0.1)


def _compute_rates_unbinned(params, probs):
    s_probs, b_probs = probs[0], probs[1]
    expected_total = params[0] * 10.0 + params[1] * 5.0
    if len(s_probs) == 0 and len(b_probs) == 0:
        return expected_total, np.array([])
    p_events = params[0] * 10.0 * s_probs + params[1] * 5.0 * b_probs
    return expected_total, p_events


def _generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool):
    n_sig = np.random.poisson(true_params[0] * 10.0)
    n_bkg = np.random.poisson(true_params[1] * 5.0)
    parts = []
    if n_sig > 0:
        parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
    if n_bkg > 0:
        parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
    return np.concatenate(parts) if parts else np.array([])


def test_compute_fc_intervals_unbinned_end_to_end_produces_finite_intervals(tmp_path):
    """
    Real run through the unbinned toy-generation/fitting pipeline: not just
    that it doesn't crash, but that it produces finite, sane test statistics
    and a properly shaped accepted-region mask (the actual thing FC interval
    construction needs the toy machinery for).
    """
    np.random.seed(11)
    S_mc_pool = np.random.normal(loc=5.0, scale=1.0, size=2000)
    B_mc_pool = np.random.exponential(scale=3.0, size=2000)

    true_params = np.array([1.0, 1.0])
    observed = _generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool)

    grids = [np.linspace(0.3, 2.0, 3), np.linspace(0.3, 2.0, 3)]

    results, _ = compute_fc_intervals(
        data=observed, grids=grids,
        compute_rates_func=_compute_rates_unbinned,
        generate_toy_func=_generate_unbinned_toy,
        pdf_components=[_s_pdf, _b_pdf],
        cl=[0.90], n_toys=8, strategy="scipy", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="unbinned", S_mc_pool=S_mc_pool, B_mc_pool=B_mc_pool,
        output_file=None, save_directory=str(tmp_path),
        compute_1D_intervals=True, compute_2D_intervals=True,
    )

    for p_idx in range(2):
        t_data = results[f"1d_t_data_p{p_idx+1}"]
        assert np.all(np.isfinite(t_data))
        assert np.all(t_data >= 0.0)
        accepted = results[f"1d_accepted_p{p_idx+1}"][0.90]
        assert accepted.dtype == bool
        assert accepted.shape == t_data.shape

    t_data_2d = results["2d_t_data_p1p2"]
    assert np.all(np.isfinite(t_data_2d))
    assert np.all(t_data_2d >= 0.0)
    accepted_2d = results["2d_accepted_p1p2"][0.90]
    assert accepted_2d.shape == t_data_2d.shape


def test_generate_unbinned_toy_produces_events_within_mc_pool_support():
    """The toy bootstrapper only ever resamples from the provided MC pools."""
    np.random.seed(3)
    S_mc_pool = np.random.normal(loc=5.0, scale=1.0, size=500)
    B_mc_pool = np.random.exponential(scale=3.0, size=500)

    toy = _generate_unbinned_toy(np.array([2.0, 2.0]), S_mc_pool, B_mc_pool)

    assert len(toy) > 0
    combined_pool = np.concatenate([S_mc_pool, B_mc_pool])
    assert np.all(np.isin(toy, combined_pool))


def test_generate_unbinned_toy_can_return_empty_array_for_zero_expected_events():
    toy = _generate_unbinned_toy(np.array([0.0, 0.0]), np.array([1.0]), np.array([1.0]))
    assert isinstance(toy, np.ndarray)
    assert len(toy) == 0
