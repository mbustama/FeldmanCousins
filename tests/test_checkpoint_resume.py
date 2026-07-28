"""
Tests for the real `warm_start`/checkpoint-resume machinery in
`orchestrator.py` (the "Resume Protocol Validation" block, ~orchestrator.py:575).

`tests/test_io.py` only round-trips a hand-built `np.savez`/`np.load` pair
and never exercises PyFC's own resume logic at all -- it would pass
identically even if the resume block were deleted entirely. These tests
simulate a genuine mid-run interruption (patching a fit function to raise
partway through the 1D scan) and verify a second `compute_fc_intervals`
call actually restores the completed work instead of silently recomputing
it, which is the behavior `examples/07_pyfc_checkpointing_tutorial.ipynb`
demonstrates interactively via a real SIGTERM kill.
"""
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import conditional_fit_1d_scipy
from pyfc.orchestrator import compute_fc_intervals

S_template = np.array([0.1, 0.5, 2.0, 5.0])
B_template = np.array([15.0, 5.0, 1.0, 0.1])


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """compute_fc_intervals's default 1D plotting leaves its figure open (caller-owned)."""
    yield
    plt.close("all")


@njit(fastmath=True, nogil=True)
def _rate_func(params, S_s2, B_s2):
    mu = params[0] * S_template + params[1] + params[2] * B_template
    return mu, S_s2


def _run(save_directory, N_data, grids, warm_start):
    return compute_fc_intervals(
        data=N_data, grids=grids,
        compute_rates_func=_rate_func,
        cl=[0.90], n_toys=3, strategy="scipy", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=warm_start,
        likelihood_type="binned", S_sumw2=np.zeros_like(S_template), B_sumw2=np.zeros_like(B_template),
        output_file=None, save_directory=str(save_directory),
        compute_1D_intervals=True, compute_2D_intervals=False,
    )


def test_interrupted_run_leaves_partial_checkpoint(tmp_path):
    """A crash partway through the 1D scan must leave param 1 complete, param 2 untouched."""
    np.random.seed(5)
    N_data = np.random.poisson(1.0 * S_template + 1.0 + 1.0 * B_template)
    grids = [np.linspace(0.5, 2.0, 4), np.linspace(0.5, 1.5, 4), np.linspace(0.5, 1.5, 4)]

    def crash_on_second_param(*args, **kwargs):
        fix_idx = args[1]
        if fix_idx == 1:
            raise RuntimeError("simulated interruption")
        return conditional_fit_1d_scipy(*args, **kwargs)

    with patch("pyfc.orchestrator.conditional_fit_1d_scipy", side_effect=crash_on_second_param):
        try:
            _run(tmp_path, N_data, grids, warm_start=True)
            assert False, "expected the simulated interruption to propagate"
        except RuntimeError:
            pass

    ckpt = np.load(tmp_path / "checkpoint_fc.npz")
    assert np.all(np.isfinite(ckpt["1d_t_data_p1"]))
    assert np.all(np.isnan(ckpt["1d_t_data_p2"]))
    assert np.all(np.isnan(ckpt["1d_t_data_p3"]))


def test_resumed_run_does_not_recompute_already_checkpointed_parameter(tmp_path):
    """
    After an interrupted run, a fresh compute_fc_intervals(warm_start=True)
    call over the SAME grids must restore param 1's results from disk
    rather than recomputing them -- proven by asserting the fit function is
    never called again with fix_idx=0, not just that the final numbers look
    plausible (which recomputation would also produce).
    """
    np.random.seed(5)
    N_data = np.random.poisson(1.0 * S_template + 1.0 + 1.0 * B_template)
    grids = [np.linspace(0.5, 2.0, 4), np.linspace(0.5, 1.5, 4), np.linspace(0.5, 1.5, 4)]

    def crash_on_second_param(*args, **kwargs):
        if args[1] == 1:
            raise RuntimeError("simulated interruption")
        return conditional_fit_1d_scipy(*args, **kwargs)

    with patch("pyfc.orchestrator.conditional_fit_1d_scipy", side_effect=crash_on_second_param):
        try:
            _run(tmp_path, N_data, grids, warm_start=True)
        except RuntimeError:
            pass

    fix_idx_calls = []

    def track_fix_idx(*args, **kwargs):
        fix_idx_calls.append(args[1])
        return conditional_fit_1d_scipy(*args, **kwargs)

    with patch("pyfc.orchestrator.conditional_fit_1d_scipy", side_effect=track_fix_idx):
        results, _ = _run(tmp_path, N_data, grids, warm_start=True)

    assert 0 not in fix_idx_calls, "resume must not recompute the already-checkpointed parameter 1"
    assert 1 in fix_idx_calls
    assert 2 in fix_idx_calls
    for p_idx in range(3):
        assert np.all(np.isfinite(results[f"1d_t_data_p{p_idx+1}"]))


def test_grid_mismatch_ignores_checkpoint_and_starts_fresh(tmp_path):
    """Changing the grid geometry between runs must not silently reuse stale checkpoint data."""
    np.random.seed(5)
    N_data = np.random.poisson(1.0 * S_template + 1.0 + 1.0 * B_template)
    grids_v1 = [np.linspace(0.5, 2.0, 4), np.linspace(0.5, 1.5, 4), np.linspace(0.5, 1.5, 4)]

    def crash_on_second_param(*args, **kwargs):
        if args[1] == 1:
            raise RuntimeError("simulated interruption")
        return conditional_fit_1d_scipy(*args, **kwargs)

    with patch("pyfc.orchestrator.conditional_fit_1d_scipy", side_effect=crash_on_second_param):
        try:
            _run(tmp_path, N_data, grids_v1, warm_start=True)
        except RuntimeError:
            pass

    grids_v2 = [np.linspace(0.5, 2.0, 5), np.linspace(0.5, 1.5, 4), np.linspace(0.5, 1.5, 4)]
    results, _ = _run(tmp_path, N_data, grids_v2, warm_start=True)

    assert len(results["1d_t_data_p1"]) == 5
    assert np.all(np.isfinite(results["1d_t_data_p1"]))
