"""
Tests for continuous optimization routines.
"""
import numpy as np
import pytest
from pyfc.optimizers import unconditional_fit_scipy, SCIPY_AVAILABLE

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
    
    # Dummy mock functions to satisfy the optimizer signature
    dummy_data = np.array([10])
    dummy_S = lambda x: x
    dummy_B = lambda x: x
    dummy_compute = lambda p, *args: (p[0], 0.1)

    try:
        # If clipping is working, this will not raise a ValueError
        min_nll, best_params = unconditional_fit_scipy(
            data=dummy_data,
            S_model=dummy_S,
            B_model=dummy_B,
            n_params=1,
            bounds_list=bounds_list,
            compute_rates_func=dummy_compute,
            seed=bad_seed,
            likelihood_type="binned",
            use_finite_mc=False
        )
        assert True
    except ValueError as e:
        if "violates bound constraints" in str(e):
            pytest.fail("x0 boundary clamping failed: SciPy rejected the seed.")
        else:
            raise e