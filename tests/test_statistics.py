"""
Tests for statistical correctness and likelihood behavior.
"""
import numpy as np
import pytest
from numba import njit

from pyfc.optimizers import SCIPY_AVAILABLE, unconditional_fit_scipy


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy is required for this test")
def test_asimov_treatment_convergence():
    """
    Verify that the optimizer converges exactly to the true parameters 
    under an Asimov statistical treatment, ensuring the likelihood 
    geometry is constructed correctly.
    """
    # We construct a scenario where the true parameter should be exactly 5.0
    true_param = 5.0
    bounds_list = [(0.0, 10.0)]
    
    # S and B templates
    dummy_S = np.array([1.0, 0.0])
    dummy_B = np.array([0.0, 20.0])
    
    # Asimov data exactly matches: (5.0 * S) + B
    asimov_data = np.array([5.0, 20.0])
    
    dummy_sig = np.zeros_like(dummy_S)

    @njit
    def asimov_compute(params, S_template, B_template, S_sigma2, B_sigma2):
        mu = params[0] * S_template + B_template
        # Variances are zeroed out for a pure Asimov check without Finite MC
        return mu, S_sigma2

    min_nll, best_params = unconditional_fit_scipy(
        data=asimov_data,
        S_model=dummy_S,
        B_model=dummy_B,
        n_params=1,
        bounds_list=bounds_list,
        compute_rates_func=asimov_compute,
        seed=np.array([1.0]), # Start far away from 5.0
        likelihood_type="binned",
        S_sigma2=dummy_sig,
        B_sigma2=dummy_sig,
        use_finite_mc=False
    )
    
    # The optimizer should cleanly find the true parameter of 5.0
    assert np.isclose(best_params[0], true_param, rtol=1e-4), \
        f"Asimov convergence failed: expected {true_param}, got {best_params[0]}"