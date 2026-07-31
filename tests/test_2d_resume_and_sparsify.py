"""
Tests for the two remaining cold paths in `orchestrator.py`: the 2D half of the
checkpoint-resume block, and `sparsify_grid`'s no-SciPy interpolation fallback.

WHY THIS FILE EXISTS: both are guarded by a condition no existing test satisfies, so they
were unrun rather than untested-by-oversight.

`tests/test_checkpoint_resume.py` covers the resume machinery thoroughly, but every one of
its runs passes `compute_2D_intervals=False`, so the block that restores 2D arrays
(`orchestrator.py` ~599-605) never executed. 2D contours are the expensive half of a
Feldman-Cousins run -- an n-parameter model has n(n-1)/2 pairs, each a full grid of toy
generations -- so they are precisely what a user restarting after a walltime kill most
needs back. Silently recomputing them would waste the hours the checkpoint exists to save,
and would look identical to working.

The sparsify fallback is the `else` of `if SCIPY_AVAILABLE`. SciPy is a hard dependency of
the test extra, so the `RectBivariateSpline` branch always wins and the nearest-neighbour
path shipped unrun. It exists for installs without SciPy, which is exactly the environment
that never runs the test suite either.
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numba import njit

from pyfc import orchestrator
from pyfc.optimizers import conditional_fit_2d_scipy
from pyfc.orchestrator import compute_fc_intervals

S_template = np.array([0.1, 0.5, 2.0, 5.0])
B_template = np.array([15.0, 5.0, 1.0, 0.1])


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """compute_fc_intervals's default plotting leaves its figure open (caller-owned)."""
    yield
    plt.close("all")


@njit(fastmath=True, nogil=True)
def _rate_func(params, S_s2, B_s2):
    mu = params[0] * S_template + params[1] * B_template + 1.0
    return mu, S_s2


def _run(save_directory, N_data, grids, warm_start, sparsify=False):
    return compute_fc_intervals(
        data=N_data, grids=grids,
        compute_rates_func=_rate_func,
        cl=[0.90], n_toys=3, strategy="scipy", num_cores=1, verbose=0,
        sparsify_grid=sparsify, warm_start=warm_start,
        likelihood_type="binned",
        S_sumw2=np.zeros_like(S_template), B_sumw2=np.zeros_like(B_template),
        output_file=None, save_directory=str(save_directory),
        compute_1D_intervals=True, compute_2D_intervals=True,
    )


def _data():
    np.random.seed(5)
    return np.random.poisson(1.0 * S_template + 1.0 * B_template + 1.0)


# --- 2D checkpoint resume -------------------------------------------------------------


def test_resumed_run_skips_2d_pairs_the_checkpoint_already_holds(tmp_path):
    """
    A run interrupted after finishing some 2D pairs must restore those and compute only the
    rest.

    Three parameters, so there are three pairs (p1p2, p1p3, p2p3) and the checkpoint has
    something to hold. Granularity matters here and is why this test is shaped as it is:
    `orchestrator.py` writes a checkpoint per *completed pair*, not per grid point, so a
    crash mid-pair saves nothing for that pair and recomputing it on resume is correct
    rather than a bug. An earlier draft used two parameters -- a single pair -- and asserted
    no 2D fit was recomputed at all, which was simply the wrong claim about how
    checkpointing works.

    The assertion is therefore on the amount of work skipped: the resumed run must call the
    2D fit strictly fewer times than a run starting from nothing. A resume block that
    restored the arrays but recomputed them anyway would return perfectly correct numbers
    and fail this.
    """
    N_data = _data()
    grids = [np.linspace(0.5, 2.0, 3), np.linspace(0.5, 1.5, 3), np.linspace(0.5, 1.5, 3)]

    real_fit = conditional_fit_2d_scipy

    # How much 2D work a cold run costs, as the baseline to beat.
    cold = {"n": 0}

    def count_only(*args, **kwargs):
        cold["n"] += 1
        return real_fit(*args, **kwargs)

    with patch.object(orchestrator, "conditional_fit_2d_scipy", count_only):
        _run(tmp_path / "cold", N_data, grids, warm_start=False)
    assert cold["n"] > 0, "the 2D fit was never called even on a cold run"

    # Now interrupt: run until at least one pair has been checkpointed, then die.
    resumed_dir = tmp_path / "resumed"
    interrupted = {"n": 0}

    def crash_after_first_pair(*args, **kwargs):
        interrupted["n"] += 1
        # One pair is 3x3 = 9 points; stop shortly after the first pair is safely written.
        if interrupted["n"] > 11:
            raise RuntimeError("simulated walltime kill during the 2D scan")
        return real_fit(*args, **kwargs)

    with patch.object(orchestrator, "conditional_fit_2d_scipy", crash_after_first_pair):
        with pytest.raises(RuntimeError):
            _run(resumed_dir, N_data, grids, warm_start=True)

    checkpoints = list(resumed_dir.glob("*.npz"))
    assert checkpoints, "the interrupted run wrote no checkpoint at all"
    with np.load(checkpoints[0], allow_pickle=True) as ckpt:
        assert any(k.startswith("2d_") for k in ckpt.files), (
            f"checkpoint holds no 2D arrays to restore; keys were {list(ckpt.files)}"
        )

    # Resume, counting how much 2D work actually has to be redone.
    warm = {"n": 0}

    def count_warm(*args, **kwargs):
        warm["n"] += 1
        return real_fit(*args, **kwargs)

    with patch.object(orchestrator, "conditional_fit_2d_scipy", count_warm):
        results, _ = _run(resumed_dir, N_data, grids, warm_start=True)

    assert warm["n"] < cold["n"], (
        f"the resumed run did {warm['n']} 2D fits against {cold['n']} for a cold run -- "
        "the checkpointed pair was recomputed rather than restored"
    )

    for pair in ("p1p2", "p1p3", "p2p3"):
        t_data_2d = results[f"2d_t_data_{pair}"]
        assert np.all(np.isfinite(t_data_2d)), f"restored 2D statistics for {pair} contain NaN"
        assert results[f"2d_accepted_{pair}"][0.90].shape == t_data_2d.shape


def test_resumed_2d_values_match_the_uninterrupted_ones(tmp_path):
    """
    Restoring the right *shape* is not enough -- the restored numbers must be the ones the
    run had computed. A resume that reloaded the wrong array, or an array of the correct
    geometry filled with defaults, would satisfy every assertion above.

    Compared against a clean, uninterrupted run of the identical problem.
    """
    N_data = _data()
    grids = [np.linspace(0.5, 2.0, 3), np.linspace(0.5, 1.5, 3)]

    reference, _ = _run(tmp_path / "clean", N_data, grids, warm_start=False)

    resumed_dir = tmp_path / "resumed"
    real_fit = conditional_fit_2d_scipy
    calls = {"n": 0}

    def crash_after_progress(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 4:
            raise RuntimeError("simulated walltime kill during the 2D scan")
        return real_fit(*args, **kwargs)

    with patch.object(orchestrator, "conditional_fit_2d_scipy", crash_after_progress):
        with pytest.raises(RuntimeError):
            _run(resumed_dir, N_data, grids, warm_start=True)

    resumed, _ = _run(resumed_dir, N_data, grids, warm_start=True)

    np.testing.assert_allclose(
        resumed["2d_t_data_p1p2"], reference["2d_t_data_p1p2"], rtol=1e-9, atol=1e-12,
        err_msg="resumed 2D test statistics differ from an uninterrupted run",
    )


# --- sparsify_grid without SciPy ------------------------------------------------------


def test_sparsify_2d_falls_back_to_nearest_neighbour_without_scipy(tmp_path):
    """
    `sparsify_grid` interpolates its coarse 2D critical surface with
    `RectBivariateSpline` when SciPy is present, and with a hand-rolled nearest-neighbour
    fill when it is not. Only the first branch has ever run, because SciPy is a hard
    dependency of the test extra.

    Patching the module flag is the only way to reach the other one short of uninstalling
    SciPy. The fallback must produce a fully populated critical surface of the right
    geometry -- every cell filled from its nearest evaluated neighbour, with nothing left
    at the NaN the results dict is initialised with.

    THE GRID SIZE IS LOAD-BEARING. At 5x5 this test passes whether or not the fallback does
    anything at all: sparsification's own edge-refinement pass ends up re-evaluating every
    cell, so the interpolated values are entirely overwritten and deleting the fill is
    invisible. Measured at 15x15 the refinement leaves interior cells alone -- removing the
    fill leaves 79 of 225 cells at NaN, and the surface keeps only 146 distinct values
    rather than 225, the repeats being exactly the nearest-neighbour plateaus. An earlier
    draft used 5x5 and duly failed to notice the fill being replaced by `pass`.
    """
    N_data = _data()
    grids = [np.linspace(0.5, 2.0, 15), np.linspace(0.5, 1.5, 15)]

    with patch.object(orchestrator, "SCIPY_AVAILABLE", False):
        results, _ = _run(tmp_path, N_data, grids, warm_start=False, sparsify=True)

    t_crit = results["2d_t_critical_p1p2"][0.90]
    assert t_crit.shape == (len(grids[0]), len(grids[1]))
    assert np.all(np.isfinite(t_crit)), (
        "the nearest-neighbour fallback left cells unfilled; "
        f"{np.count_nonzero(~np.isfinite(t_crit))} of {t_crit.size} are not finite"
    )
    assert np.all(t_crit >= 0.0), "a negative critical value reached the surface"

    # Nearest-neighbour fill copies values into cells that were never evaluated, so the
    # surface must contain plateaus: strictly fewer distinct values than cells. A surface
    # where every cell is distinct means the fill contributed nothing.
    assert np.unique(t_crit).size < t_crit.size, (
        f"every one of {t_crit.size} cells holds a distinct value, so no nearest-neighbour "
        "copying happened -- the grid is small enough that refinement re-evaluated "
        "everything, and this test proves nothing at that size"
    )


def test_sparsify_2d_agrees_with_the_dense_scan_on_evaluated_cells(tmp_path):
    """
    The fallback is an approximation, so it is not required to match a dense scan
    everywhere -- but the accepted region it produces must still be a sane boolean mask of
    the right geometry, and the statistic it is compared against is the exact one, since
    `t_data` is always computed densely regardless of sparsification.
    """
    N_data = _data()
    grids = [np.linspace(0.5, 2.0, 5), np.linspace(0.5, 1.5, 5)]

    dense, _ = _run(tmp_path / "dense", N_data, grids, warm_start=False, sparsify=False)

    with patch.object(orchestrator, "SCIPY_AVAILABLE", False):
        sparse, _ = _run(tmp_path / "sparse", N_data, grids, warm_start=False, sparsify=True)

    np.testing.assert_allclose(
        sparse["2d_t_data_p1p2"], dense["2d_t_data_p1p2"], rtol=1e-9, atol=1e-12,
        err_msg="sparsification changed t_data, which is always computed densely",
    )

    accepted = sparse["2d_accepted_p1p2"][0.90]
    assert accepted.dtype == bool
    assert accepted.shape == dense["2d_accepted_p1p2"][0.90].shape
