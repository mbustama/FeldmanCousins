"""
Tests for reporting disconnected 1D accepted intervals as separate
[lo, hi] pairs instead of merging them into one too-wide interval.

Previously, `_save_fc_json` computed `interval_bounds` as a flat
`[min(accepted), max(accepted)]` across all accepted points, which -- if
the true accepted set is disconnected (a real possibility under the
Feldman-Cousins unified construction near certain physical boundaries) --
silently reports one interval spanning from the region's minimum to
maximum, including whatever rejected gap sits in between.
"""
import json
import os
import tempfile

import numpy as np
import pytest
from numba import njit

from pyfc.orchestrator import _find_contiguous_intervals, _save_fc_json, compute_fc_intervals


# --- Unit tests for the contiguous-run helper itself ---

def test_find_contiguous_intervals_two_separate_runs():
    """The brief's own worked example: two separate True runs."""
    test_points = list(range(10))
    accepted = [False, False, True, True, True, False, False, True, True, False]

    intervals = _find_contiguous_intervals(test_points, accepted)

    assert intervals == [[2, 4], [7, 8]]


def test_find_contiguous_intervals_single_contiguous_run():
    """A single contiguous run must still come back as a list-of-one-pair."""
    test_points = np.linspace(0.0, 1.0, 5)
    accepted = [False, True, True, True, False]

    intervals = _find_contiguous_intervals(test_points, accepted)

    assert intervals == [[test_points[1], test_points[3]]]


def test_find_contiguous_intervals_all_accepted():
    test_points = [0.0, 1.0, 2.0]
    accepted = [True, True, True]

    assert _find_contiguous_intervals(test_points, accepted) == [[0.0, 2.0]]


def test_find_contiguous_intervals_none_accepted():
    test_points = [0.0, 1.0, 2.0]
    accepted = [False, False, False]

    assert _find_contiguous_intervals(test_points, accepted) == []


def test_find_contiguous_intervals_single_point_run():
    test_points = [0.0, 1.0, 2.0]
    accepted = [False, False, True]

    assert _find_contiguous_intervals(test_points, accepted) == [[2.0, 2.0]]


def test_find_contiguous_intervals_run_touching_both_edges():
    test_points = [0.0, 1.0, 2.0, 3.0]
    accepted = [True, False, False, True]

    assert _find_contiguous_intervals(test_points, accepted) == [[0.0, 0.0], [3.0, 3.0]]


# --- _save_fc_json schema test with a hand-crafted results dict ---

def test_save_fc_json_reports_disconnected_intervals_not_merged():
    """
    Directly proves the JSON-writing logic itself is correct for a
    hand-crafted, deliberately disconnected accepted mask (independent of
    whether a real physics model happens to produce one).
    """
    test_points = np.linspace(0.0, 9.0, 10)
    accepted = np.array([False, False, True, True, True, False, False, True, True, False])

    results = {
        "best_fit": np.array([4.0]),
        "data_uncond_nll": 0.0,
        "1d_test_p1": test_points,
        "1d_t_data_p1": np.zeros(10),
        "1d_prof_params_p1": np.zeros((10, 1)),
        "1d_t_critical_p1": {0.9: np.ones(10)},
        "1d_accepted_p1": {0.9: accepted},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "fc_results.json")
        _save_fc_json(results, out_path, [0.9], True, False, 1)

        with open(out_path) as f:
            saved = json.load(f)

    bounds = saved["1d_intervals"]["param1"]["thresholds"]["0.9"]["interval_bounds"]
    assert bounds == [[2.0, 4.0], [7.0, 8.0]]

    # Never merged into a single [lo, hi] spanning the rejected gap.
    assert bounds != [2.0, 8.0]


def test_save_fc_json_single_interval_still_wrapped_in_list_of_pairs():
    """
    Schema is always a list-of-pairs, even with exactly one contiguous
    run -- no special-casing "flatten if there's only one interval", which
    would reintroduce ambiguity for downstream parsers.
    """
    test_points = np.linspace(0.0, 4.0, 5)
    accepted = np.array([False, True, True, True, False])

    results = {
        "best_fit": np.array([2.0]),
        "data_uncond_nll": 0.0,
        "1d_test_p1": test_points,
        "1d_t_data_p1": np.zeros(5),
        "1d_prof_params_p1": np.zeros((5, 1)),
        "1d_t_critical_p1": {0.9: np.ones(5)},
        "1d_accepted_p1": {0.9: accepted},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "fc_results.json")
        _save_fc_json(results, out_path, [0.9], True, False, 1)
        with open(out_path) as f:
            saved = json.load(f)

    bounds = saved["1d_intervals"]["param1"]["thresholds"]["0.9"]["interval_bounds"]
    assert bounds == [[1.0, 3.0]]
    assert isinstance(bounds[0], list)


# --- End-to-end: a genuine physical model with a disconnected accepted region ---

@njit(fastmath=True, nogil=True)
def _oscillating_rate_func(params, S_s2, B_s2):
    """
    mu(x) = 5 + 4*sin(2*pi*x) oscillates through the observed count (5)
    multiple times over x in [0, 1], with a single deep dip (mu=1) around
    x=0.75. This produces a t_data profile with two separated troughs
    (near the mu=n crossings) and a peak in between -- a genuinely
    disconnected accepted region, not a contrived mock.
    """
    mu = np.array([5.0 + 4.0 * np.sin(2.0 * np.pi * params[0])])
    return mu, np.array([0.0])


def test_compute_fc_intervals_reports_disconnected_region_end_to_end(tmp_path):
    N_obs = np.array([5.0])
    grids = [np.linspace(0.0, 1.0, 41)]

    results, _ = compute_fc_intervals(
        data=N_obs, grids=grids, compute_rates_func=_oscillating_rate_func,
        cl=[0.90], n_toys=300, strategy="grid", num_cores=1, verbose=0,
        sparsify_grid=False, warm_start=False, likelihood_type="binned",
        S_sumw2=np.zeros(1), B_sumw2=np.zeros(1),
        compute_1D_intervals=True, compute_2D_intervals=False,
        output_file="fc_results",
        save_directory=str(tmp_path / "disconnected_demo"),
    )

    accepted = results["1d_accepted_p1"][0.90]
    test_points = results["1d_test_p1"]
    intervals = _find_contiguous_intervals(test_points, accepted)

    assert len(intervals) >= 2, (
        "Expected a genuinely disconnected accepted region (>=2 separate "
        f"runs) from the oscillating rate function; got {intervals}"
    )

    with open(tmp_path / "disconnected_demo" / "fc_results.json") as f:
        saved = json.load(f)
    json_bounds = saved["1d_intervals"]["param1"]["thresholds"]["0.9"]["interval_bounds"]

    assert json_bounds == intervals
    assert len(json_bounds) >= 2
    for lo, hi in json_bounds:
        assert lo <= hi
