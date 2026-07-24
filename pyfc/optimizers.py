"""
Continuous Optimization Module

This file contains the continuous optimization routines used to minimize the 
Negative Log-Likelihood (NLL) functions for both binned and unbinned models. 
It provides interfaces for gradient-based local optimization via SciPy (L-BFGS-B) 
and robust global optimization via UltraNest (Nested Sampling). These optimizers 
are essential for profiling out nuisance parameters and finding the global 
and conditional likelihood maxima required for the Profile Likelihood Ratio 
test statistic in the Feldman-Cousins construction.

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np
from .binned import calc_nll
from .unbinned import calc_nll_unbinned

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import ultranest
    ULTRANEST_AVAILABLE = True
except ImportError:
    ULTRANEST_AVAILABLE = False


# --- 1. Scipy Continuous Optimizers ---
def unconditional_fit_scipy(data, S_model, B_model, n_params, bounds_list, seed=None, 
                            likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """
    Performs an unconditional global maximum likelihood fit using SciPy's L-BFGS-B.
    
    Statistical Theory & Assumptions:
    To evaluate the denominator of the Profile Likelihood Ratio (PLR), we must find 
    the parameter combination that yields the absolute minimum Negative Log-Likelihood 
    (NLL). 
    
    Mathematical Expression:
    NLL_uncond = min_{theta} NLL(theta | data)
    where theta represents the full set of N free parameters in the model.
    
    The L-BFGS-B algorithm is a quasi-Newton method that approximates the Broyden-Fletcher-Goldfarb-Shanno 
    (BFGS) update to the Hessian matrix, scaled for bounded parameter spaces. It assumes 
    the likelihood surface is sufficiently smooth and twice-differentiable. Because it is 
    a local optimizer, it can be sensitive to the initial seed.

    Parameters:
    -----------
    data : array_like
        The observed dataset (binned counts or unbinned events).
    S_model, B_model : callable or array_like
        Functions or arrays representing the signal and background models/templates.
    n_params : int
        Total number of parameters in the model.
    bounds_list : list of tuples
        A list of (min, max) bounds for each parameter.
    seed : array_like, 1D, optional
        Initial parameter guess. If None, the midpoint of the bounds is used.
    likelihood_type : str
        Either "binned" or "unbinned".
    S_sigma2, B_sigma2 : array_like, optional
        Template variances for finite MC binned likelihoods.
    use_finite_mc : bool
        Flag to enable the Poisson-Gamma mixture likelihood for binned data.

    Returns:
    --------
    min_nll : float
        The minimal negative log-likelihood found by the optimizer.
    best_params : array_like, 1D
        The parameter combination corresponding to min_nll.
    """
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def cost(params):
            return calc_nll_unbinned(params, len_obs, s_probs, b_probs)
    else:
        def cost(params):
            return calc_nll(params, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = seed if seed is not None else [(b[0] + b[1]) / 2.0 for b in bounds_list]
    res = optimize.minimize(cost, x0=x0, bounds=bounds_list, method='L-BFGS-B')
    return res.fun, res.x

def conditional_fit_1d_scipy(test_val, fix_idx, n_params, data, S_model, B_model, bounds_list, seed=None, 
                             likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """
    Performs a conditional maximum likelihood fit (profiling 1 parameter) using SciPy.
    
    Statistical Theory:
    To evaluate the numerator of the Profile Likelihood Ratio for 1D intervals, 
    one parameter of interest is fixed to a specific 'test_val', and the NLL is 
    minimized over the remaining (N-1) nuisance parameters.
    
    Mathematical Expression:
    NLL_cond = min_{nu} NLL(theta_fixed, nu | data)
    where theta_fixed is the parameter of interest and nu are the nuisance parameters.

    Parameters:
    -----------
    test_val : float
        The value at which to fix the parameter of interest.
    fix_idx : int
        The index of the parameter being fixed.
    n_params : int
        Total number of parameters.
    data, S_model, B_model, bounds_list : various
        Dataset, models, and boundary constraints.
    seed : array_like, 1D, optional
        Initial guess for the *free* parameters.
    likelihood_type, S_sigma2, B_sigma2, use_finite_mc : various
        Likelihood formulation configurations.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the fixed test_val and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i != fix_idx]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = seed if seed is not None else [(b[0] + b[1]) / 2.0 for b in free_bounds]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_idx] = test_val
        return cost(np.array([])), p
        
    res = optimize.minimize(cost, x0=x0, bounds=free_bounds, method='L-BFGS-B')
    
    best_p = np.zeros(n_params)
    best_p[fix_idx] = test_val
    for idx, free_val in zip(free_indices, res.x):
        best_p[idx] = free_val
        
    return res.fun, best_p

def conditional_fit_2d_scipy(test_vA, test_vB, fix_A, fix_B, n_params, data, S_model, B_model, bounds_list, seed=None, 
                             likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """
    Performs a conditional maximum likelihood fit (profiling 2 parameters) using SciPy.
    
    Statistical Theory:
    Analogous to the 1D profiling function, but computes the joint conditional 
    minimum for a 2D confidence region by fixing two parameters and optimizing 
    the remaining (N-2) nuisance parameters.

    Parameters:
    -----------
    test_vA, test_vB : float
        The values at which to fix the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    n_params : int
        Total number of parameters.
    data, S_model, B_model, bounds_list : various
        Dataset, models, and boundary constraints.
    seed : array_like, 1D, optional
        Initial guess for the *free* parameters.
    likelihood_type, S_sigma2, B_sigma2, use_finite_mc : various
        Likelihood formulation configurations.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the two fixed values and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i not in (fix_A, fix_B)]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = seed if seed is not None else [(b[0] + b[1]) / 2.0 for b in free_bounds]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_A] = test_vA
        p[fix_B] = test_vB
        return cost(np.array([])), p
        
    res = optimize.minimize(cost, x0=x0, bounds=free_bounds, method='L-BFGS-B')
    
    best_p = np.zeros(n_params)
    best_p[fix_A] = test_vA
    best_p[fix_B] = test_vB
    for idx, free_val in zip(free_indices, res.x):
        best_p[idx] = free_val
        
    return res.fun, best_p


# --- 2. UltraNest Optimizers ---
def unconditional_fit_ultranest(data, S_model, B_model, n_params, bounds_list, verbose=1, 
                                likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """
    Performs an unconditional global maximum likelihood fit using UltraNest.
    
    Statistical Theory & Assumptions:
    UltraNest relies on Reactive Nested Sampling, an algorithm traditionally used 
    for Bayesian evidence computation and posterior sampling. However, because it 
    exhaustively maps the likelihood surface across the entire prior volume without 
    relying on gradients, it acts as a highly robust global optimizer, virtually 
    immune to local minima traps. 
    
    The algorithm evaluates the likelihood by converting a uniformly sampled unit 
    hypercube [0, 1]^N into the physical parameter bounds via a prior transform. 
    The returned value is the Maximum Likelihood Estimate (MLE) discovered during 
    the integration phase.

    Parameters:
    -----------
    data : array_like
        The observed dataset (binned counts or unbinned events).
    S_model, B_model : callable or array_like
        Functions or arrays representing the signal and background models.
    n_params : int
        Total number of parameters in the model.
    bounds_list : list of tuples
        A list of (min, max) bounds for each parameter.
    verbose : int
        Verbosity level controlling UltraNest output (2 for full tracking).
    likelihood_type, S_sigma2, B_sigma2, use_finite_mc : various
        Likelihood formulation configurations.

    Returns:
    --------
    min_nll : float
        The absolute minimum negative log-likelihood (-1 * max_logL).
    best_params : array_like, 1D
        The parameter combination corresponding to the maximum log-likelihood point.
    """
    def prior_transform(cube):
        return np.array([cube[i] * (bounds_list[i][1] - bounds_list[i][0]) + bounds_list[i][0] for i in range(n_params)])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def log_likelihood(p): 
            return -calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def log_likelihood(p): 
            return -calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    param_names = [f'p{i+1}' for i in range(n_params)]
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    return -result['maximum_likelihood']['logl'], result['maximum_likelihood']['point']

def conditional_fit_1d_ultranest(test_val, fix_idx, n_params, data, S_model, B_model, bounds_list, verbose=1, 
                                 likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """
    Performs a conditional maximum likelihood fit (profiling 1 parameter) using UltraNest.
    
    Statistical Theory:
    Executes Reactive Nested Sampling in the reduced (N-1) dimensional subspace 
    of nuisance parameters to robustly find the conditional maximum likelihood.
    This prevents false confidence interval exclusions that can occur if a gradient 
    optimizer falls into a local minimum when profiling.

    Parameters:
    -----------
    test_val : float
        The value at which to fix the parameter of interest.
    fix_idx : int
        The index of the parameter being fixed.
    n_params : int
        Total number of parameters in the model.
    data, S_model, B_model, bounds_list : various
        Dataset, models, and boundary constraints.
    verbose : int
        Verbosity level.
    likelihood_type, S_sigma2, B_sigma2, use_finite_mc : various
        Likelihood formulation configurations.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the fixed test_val and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i != fix_idx]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_idx] = test_val
        if likelihood_type == "unbinned":
            s_probs = S_model(data) if len(data) > 0 else np.array([])
            b_probs = B_model(data) if len(data) > 0 else np.array([])
            return calc_nll_unbinned(p, len(data), s_probs, b_probs), p
        else:
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc), p

    def prior_transform(cube):
        return np.array([cube[i] * (free_bounds[i][1] - free_bounds[i][0]) + free_bounds[i][0] for i in range(len(free_bounds))])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    param_names = [f'f{i+1}' for i in range(len(free_bounds))]
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    
    best_p = np.zeros(n_params)
    best_p[fix_idx] = test_val
    for idx, free_val in zip(free_indices, result['maximum_likelihood']['point']):
        best_p[idx] = free_val
        
    return -result['maximum_likelihood']['logl'], best_p

def conditional_fit_2d_ultranest(test_vA, test_vB, fix_A, fix_B, n_params, data, S_model, B_model, bounds_list, verbose=1, 
                                 likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """
    Performs a conditional maximum likelihood fit (profiling 2 parameters) using UltraNest.
    
    Statistical Theory:
    Executes Reactive Nested Sampling in the (N-2) dimensional nuisance parameter 
    subspace to construct a robust 2D confidence region profile.

    Parameters:
    -----------
    test_vA, test_vB : float
        The values at which to fix the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    n_params : int
        Total number of parameters in the model.
    data, S_model, B_model, bounds_list : various
        Dataset, models, and boundary constraints.
    verbose : int
        Verbosity level.
    likelihood_type, S_sigma2, B_sigma2, use_finite_mc : various
        Likelihood formulation configurations.

    Returns:
    --------
    min_nll : float
        The profiled minimum negative log-likelihood.
    best_p : array_like, 1D
        The full parameter array (size N) including the two fixed values and the optimized nuisance parameters.
    """
    free_indices = [i for i in range(n_params) if i not in (fix_A, fix_B)]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_A] = test_vA
        p[fix_B] = test_vB
        if likelihood_type == "unbinned":
            s_probs = S_model(data) if len(data) > 0 else np.array([])
            b_probs = B_model(data) if len(data) > 0 else np.array([])
            return calc_nll_unbinned(p, len(data), s_probs, b_probs), p
        else:
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc), p
            
    def prior_transform(cube):
        return np.array([cube[i] * (free_bounds[i][1] - free_bounds[i][0]) + free_bounds[i][0] for i in range(len(free_bounds))])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    param_names = [f'f{i+1}' for i in range(len(free_bounds))]
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    
    best_p = np.zeros(n_params)
    best_p[fix_A] = test_vA
    best_p[fix_B] = test_vB
    for idx, free_val in zip(free_indices, result['maximum_likelihood']['point']):
        best_p[idx] = free_val
        
    return -result['maximum_likelihood']['logl'], best_p