"""
Tests for continuous optimization routines.
"""
import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import SCIPY_AVAILABLE, unconditional_fit_scipy


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this test")
def test_scipy_boundary_clamping():
    """
    Verify that microscopic floating-point overshoots in the seed 
    are properly clamped by np.clip before entering SciPy's optimizer.
    """
    # Mock bounds: [0.0, 1.0]
    bounds_list = [(0.0, 1.0)]
    
    # Intentionally provide a seed that slightly violates the upper bound
    bad_seed = np.array([1.0000000000000002])
    
    # Provide Numba-compatible arrays for binned templates (not lambdas)
    dummy_data = np.array([10.0])
    dummy_S = np.array([1.0])
    dummy_B = np.array([1.0])
    dummy_S_sig = np.array([0.1])
    dummy_B_sig = np.array([0.1])

    # Provide a properly JIT-compiled compute function so Numba doesn't fail
    @njit
    def dummy_compute(params, S_template, B_template, S_sigma2, B_sigma2):
        mu = params[0] * S_template
        sigma2 = params[0] * S_template * 0.1
        return mu, sigma2

    try:
        # If clipping is working, this will not raise a SciPy ValueError
        min_nll, best_params = unconditional_fit_scipy(
            data=dummy_data,
            S_model=dummy_S,
            B_model=dummy_B,
            n_params=1,
            bounds_list=bounds_list,
            compute_rates_func=dummy_compute,
            seed=bad_seed,
            likelihood_type="binned",
            S_sigma2=dummy_S_sig,
            B_sigma2=dummy_B_sig,
            use_finite_mc=False
        )
        assert True
    except ValueError as e:
        if "violates bound constraints" in str(e):
            pytest.fail("x0 boundary clamping failed: SciPy rejected the seed.")
        else:
            raise e