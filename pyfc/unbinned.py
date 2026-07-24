"""
Unbinned Likelihood and Grid Optimization Module

This file implements the Extended Unbinned Maximum Likelihood (EUML) routines 
for the PyFC package. Unlike binned analyses which rely on histograms, the unbinned 
method evaluates the probability density directly on an event-by-event basis. 
This is statistically powerful for small datasets where binning would cause 
unacceptable information loss.

The module also includes brute-force grid search optimizers for the unbinned 
likelihood, which serve as robust alternatives to gradient-based minimizers 
when the likelihood surface is highly irregular.

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np

# --- 1. Core Math (Unbinned) ---
def calc_nll_unbinned(params, len_obs, s_probs, b_probs, compute_rates_func):
    """
    Computes the Extended Unbinned Negative Log-Likelihood (NLL).
    
    Mathematical Expression:
    -ln(L) = N_expected - sum(ln(lambda(x_i)))
    
    Note: The factorial term ln(N_obs!) is dropped from the standard EUML 
    formula because it is a constant that depends only on the data, 
    and therefore strictly cancels out when computing delta-NLL test statistics.

    Parameters:
    -----------
    params : array_like
        The physical parameters.
    len_obs : int
        The total number of observed events (N_obs).
    s_probs, b_probs : np.ndarray
        The pre-evaluated signal and background PDFs for the observed events.
    compute_rates_func : callable
        User-provided function mapping parameters to expected total events and 
        unnormalized probability densities.

    Returns:
    --------
    float
        The calculated NLL value.
    """
    expected_total, p_events = compute_rates_func(params, s_probs, b_probs)
    
    # Handle the zero-observation case perfectly
    if len_obs == 0:
        return expected_total

    # Apply a heavy penalty (1e10) if the model predicts negative or zero rate 
    # for an event that actually occurred, avoiding fatal np.log(x <= 0) math domain errors.
    if np.any(p_events <= 0):
        return 1e10 
        
    return expected_total - np.sum(np.log(p_events))


# --- 2. Grid Search Optimizers (Unbinned) ---
def unconditional_fit_grid_unbinned(obs_events, S_pdf, B_pdf, full_grid_points, compute_rates_func):
    """
    Performs a brute-force global scan over the parameter space to find the unconditional MLE.
    
    Because evaluating the PDF function S_pdf(obs_events) for every parameter combination 
    can be incredibly slow in Python, this function evaluates the PDFs exactly ONCE for 
    the observed dataset, and reuses those static arrays across the entire grid loop.
    
    Parameters:
    -----------
    obs_events : array_like
        The observed data events.
    S_pdf, B_pdf : callable
        Probability density functions for signal and background.
    full_grid_points : array_like, 2D
        Matrix containing all parameter combinations to scan.
    compute_rates_func : callable
        User-provided expected rates function.
        
    Returns:
    --------
    min_nll : float
        The absolute minimum negative log-likelihood found.
    best_params : array_like, 1D
        The parameter combination corresponding to min_nll.
    """
    len_obs = len(obs_events)
    # Pre-compute structural PDFs to avoid massive redundant computation inside the loop
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_params = full_grid_points[0].copy()
    
    for row in range(len(full_grid_points)):
        p = full_grid_points[row]
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs, compute_rates_func)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

def conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, obs_events, S_pdf, B_pdf, cond_grid_points, compute_rates_func):
    """
    Performs a brute-force 1D conditional profile scan for the unbinned likelihood.
    Fixes one parameter of interest and scans the remaining grid.
    
    Parameters:
    -----------
    test_val : float
        The fixed value for the parameter of interest.
    fix_idx : int
        The index of the parameter to fix.
    n_params : int
        Total number of parameters.
    obs_events : array_like
        The observed data events.
    S_pdf, B_pdf : callable
        Probability density functions.
    cond_grid_points : array_like, 2D
        The sub-grid of nuisance parameters to profile over.
    compute_rates_func : callable
        User-provided expected rates function.
        
    Returns:
    --------
    min_nll : float
        The minimum negative log-likelihood given the fixed parameter.
    best_params : array_like, 1D
        The parameter combination that produced min_nll.
    """
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_params = np.zeros(n_params)
    
    # Pre-allocate array and lock the parameter of interest
    p = np.zeros(n_params)
    p[fix_idx] = test_val
    
    for row in range(len(cond_grid_points)):
        free_p = cond_grid_points[row]
        free_i = 0
        
        # Inject the free nuisance parameters into the full parameter array
        for i in range(n_params):
            if i != fix_idx:
                p[i] = free_p[free_i]
                free_i += 1
                
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs, compute_rates_func)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

def conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, obs_events, S_pdf, B_pdf, cond_grid_points, compute_rates_func):
    """
    Performs a brute-force 2D conditional profile scan for the unbinned likelihood.
    Fixes two parameters of interest to map joint contours.
    
    Parameters:
    -----------
    test_vA, test_vB : float
        The fixed values for the two parameters of interest.
    fix_A, fix_B : int
        The indices of the fixed parameters.
    n_params : int
        Total number of parameters.
    obs_events : array_like
        The observed data events.
    S_pdf, B_pdf : callable
        Probability density functions.
    cond_grid_points : array_like, 2D
        The sub-grid of nuisance parameters to profile over.
    compute_rates_func : callable
        User-provided expected rates function.
        
    Returns:
    --------
    min_nll : float
        The minimum negative log-likelihood given the two fixed parameters.
    best_params : array_like, 1D
        The parameter combination that produced min_nll.
    """
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_params = np.zeros(n_params)
    
    p = np.zeros(n_params)
    p[fix_A] = test_vA
    p[fix_B] = test_vB
    
    for row in range(len(cond_grid_points)):
        free_p = cond_grid_points[row]
        free_i = 0
        for i in range(n_params):
            if i != fix_A and i != fix_B:
                p[i] = free_p[free_i]
                free_i += 1
                
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs, compute_rates_func)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params


# --- 3. Toy Generators (Unbinned) ---
def generate_and_fit_toys_grid_unbinned_1d(test_val, fix_idx, true_params, n_params, S_pdf, B_pdf, 
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool, 
                                           compute_rates_func, generate_toy_func):
    """
    Sequentially generates and evaluates 1D unbinned toys entirely using grid search.
    
    Used when gradient optimizers fail and external parallelization is disabled. 
    It evaluates the Profile Likelihood Ratio (cond_nll - uncond_nll) for each 
    simulated unbinned dataset.
    
    Parameters:
    -----------
    test_val : float
        The fixed value for the parameter of interest.
    fix_idx : int
        The index of the fixed parameter.
    true_params : array_like, 1D
        The physical parameters dictating the expected event yields.
    n_params : int
        Total number of parameters in the model.
    S_pdf, B_pdf : callable
        Probability density functions.
    full_grid_points : array_like, 2D
        Grid for the unconditional fit.
    cond_grid_points : array_like, 2D
        Grid for the conditional (nuisance parameter) fit.
    n_toys : int
        Number of pseudo-experiments to generate.
    S_mc_pool, B_mc_pool : np.ndarray
        Pre-generated continuous events for bootstrapping.
    compute_rates_func : callable
        User-provided physics mapping function.
    generate_toy_func : callable
        User-provided function to bootstrap the toy data.
        
    Returns:
    --------
    t_statistics : array_like, 1D
        Array of length `n_toys` containing the test statistic for each toy.
    """
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_toy_func(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, full_grid_points, compute_rates_func)
        cond_nll, _ = conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, toy_events, S_pdf, B_pdf, cond_grid_points, compute_rates_func)
        
        # Bounded at 0.0 to correct floating point inaccuracies when cond_nll ≈ uncond_nll
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics

def generate_and_fit_toys_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, true_params, n_params, S_pdf, B_pdf, 
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool, 
                                           compute_rates_func, generate_toy_func):
    """
    Sequentially generates and evaluates 2D unbinned toys entirely using grid search.
    
    Parameters:
    -----------
    test_vA, test_vB : float
        The fixed values for the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    true_params : array_like, 1D
        The physical parameters dictating the expected event yields.
    n_params : int
        Total number of parameters in the model.
    S_pdf, B_pdf : callable
        Probability density functions.
    full_grid_points : array_like, 2D
        Grid for the unconditional fit.
    cond_grid_points : array_like, 2D
        Grid for the conditional (nuisance parameter) fit.
    n_toys : int
        Number of pseudo-experiments to generate.
    S_mc_pool, B_mc_pool : np.ndarray
        Pre-generated continuous events for bootstrapping.
    compute_rates_func : callable
        User-provided physics mapping function.
    generate_toy_func : callable
        User-provided function to bootstrap the toy data.
        
    Returns:
    --------
    t_statistics : array_like, 1D
        Array of length `n_toys` containing the test statistic for each toy.
    """
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_toy_func(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, full_grid_points, compute_rates_func)
        cond_nll, _ = conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, toy_events, S_pdf, B_pdf, cond_grid_points, compute_rates_func)
        
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics