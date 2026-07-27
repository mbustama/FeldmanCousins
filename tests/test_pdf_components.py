"""
Tests for the generalized `pdf_components` (list of callables) mechanism
that replaces the old hardcoded S_model/B_model pair for unbinned fits.
"""
import math

import numpy as np

from pyfc.unbinned import calc_nll_unbinned, unconditional_fit_grid_unbinned


def _signal_pdf(x):
    return np.exp(-0.5 * x**2)


def _background_pdf(x):
    return np.full_like(x, 0.1)


def test_two_component_matches_manual_signal_background_computation():
    """
    2-component pdf_components=[signal_pdf, background_pdf] must reproduce
    exactly what the old S_model/B_model two-argument compute_rates_func did.
    """
    events = np.array([0.1, 0.5, -0.3, 1.2])

    def compute_rates_old_style(params, s_probs, b_probs):
        p_events = params[0] * s_probs + params[1] * b_probs
        return params[0] + params[1], p_events

    def compute_rates_new_style(params, probs):
        return compute_rates_old_style(params, probs[0], probs[1])

    s_probs = _signal_pdf(events)
    b_probs = _background_pdf(events)
    params = np.array([5.0, 2.0])

    expected_total, p_events = compute_rates_old_style(params, s_probs, b_probs)
    nll_expected = expected_total - np.sum(np.log(p_events))

    probs = [s_probs, b_probs]
    nll_actual = calc_nll_unbinned(params, len(events), probs, compute_rates_new_style)

    assert np.isclose(nll_actual, nll_expected)


def test_three_component_against_hand_computed_reference():
    """
    3+ components is a capability that has no old-API equivalent to diff
    against, so this checks calc_nll_unbinned against an independently
    hand-computed reference NLL.

    Setup: three constant-valued PDFs (1.0, 2.0, 3.0) over 3 events, so
    every event has the same mixture density p = a1*1 + a2*2 + a3*3.
    With params = [1.0, 2.0, 3.0]:
      p_events = 1*1 + 2*2 + 3*3 = 14.0  (same for all 3 events)
      expected_total = 1.0 + 2.0 + 3.0 = 6.0
      nll = expected_total - sum(log(p_events)) = 6.0 - 3*log(14.0)
    """
    events = np.array([0.0, 1.0, 2.0])
    probs = [np.full_like(events, 1.0), np.full_like(events, 2.0), np.full_like(events, 3.0)]
    params = np.array([1.0, 2.0, 3.0])

    def compute_rates_3comp(params, probs):
        total = 0.0
        p_events = np.zeros_like(probs[0])
        for i, p in enumerate(probs):
            total += params[i]
            p_events = p_events + params[i] * p
        return total, p_events

    nll_actual = calc_nll_unbinned(params, len(events), probs, compute_rates_3comp)
    nll_reference = 6.0 - 3 * math.log(14.0)

    assert np.isclose(nll_actual, nll_reference)


def test_unconditional_fit_grid_unbinned_with_pdf_components_list():
    """
    The grid-search unbinned optimizer must accept pdf_components as a
    plain list of callables and pre-evaluate each exactly once.
    """
    events = np.array([0.1, 0.5, -0.3, 1.2])

    def compute_rates(params, probs):
        s_probs, b_probs = probs[0], probs[1]
        p_events = params[0] * s_probs + params[1] * b_probs
        return params[0] + params[1], p_events

    full_grid_points = np.array([[1.0, 1.0], [5.0, 2.0], [3.0, 3.0]])
    min_nll, best_params = unconditional_fit_grid_unbinned(
        events, [_signal_pdf, _background_pdf], full_grid_points, compute_rates
    )

    # Brute force check: manually recompute NLL at each grid point.
    s_probs = _signal_pdf(events)
    b_probs = _background_pdf(events)
    manual_nlls = []
    for p in full_grid_points:
        expected_total, p_events = compute_rates(p, [s_probs, b_probs])
        manual_nlls.append(expected_total - np.sum(np.log(p_events)))
    expected_min_nll = min(manual_nlls)
    expected_best = full_grid_points[int(np.argmin(manual_nlls))]

    assert np.isclose(min_nll, expected_min_nll)
    assert np.array_equal(best_params, expected_best)
