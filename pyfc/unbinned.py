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

# ==============================================================================
# --- 1. USER DEFINED PHYSICS MODEL (UNBINNED) ---
# Modify this section to define how your N parameters map to the PDFs.
# ==============================================================================
def compute_rates_unbinned(params, s_probs, b_probs):
    """
    Maps an N-dimensional parameter array to expected total events and 
    the probability density of observed events.
    
    Statistical Context:
    For an extended unbinned likelihood, the user must provide two things:
    1. The total expected number of events across the entire domain.
    2. The unnormalized rate (or probability density * expected count) for 
       each specifically observed event.
       
    Example provided: 3-parameter model where:
    - params[0] could represent a signal cross-section.
    - params[1] could represent an exposure or flux multiplier.
    - params[2] represents the expected background count.

    Parameters:
    -----------
    params : array_like
        The physical parameters being tested.
    s_probs, b_probs : np.ndarray
        The arrays of signal and background PDF values evaluated at the exact 
        kinematic coordinates of the observed events.

    Returns:
    --------
    expected_total : float
        The integral of the model over the full observable space.
    p_events : np.ndarray
        The specific rate evaluated at each observed event.
    """
    # Calculate the total expected events (integral of the extended PDF)
    expected_total = params[0] * params[1] + params[2]
    
    # Catch empty datasets to prevent array shape mismatches
    if len(s_probs) == 0 and len(b_probs) == 0:
        return expected_total, np.array([])
        
    # Calculate the event-by-event rate: lambda(x_i) = s * S(x_i) + b * B(x_i)
    p_events = params[0] * params[1] * s_probs + params[2] * b_probs
    
    return expected_total, p_events

def generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool):
    """
    Generates a mock unbinned dataset (pseudo-experiment) via bootstrapping.
    
    To rapidly generate unbinned events without running expensive PDF inverse-transform 
    sampling, this function bootstraps (draws with replacement) from pre-generated 
    Monte Carlo event pools provided by the user.

    Parameters:
    -----------
    true_params : array_like
        The physical parameters dictating the expected event yields.
    S_mc_pool, B_mc_pool : np.ndarray
        Pre-generated continuous events distributed according to the respective PDFs.

    Returns:
    --------
    np.ndarray
        A 1D array of simulated continuous events for the toy dataset.
    """
    # 1. Determine the integer number of events to draw via Poisson fluctuation
    n_sig = np.random.poisson(true_params[0] * true_params[1])
    n_bkg = np.random.poisson(true_params[2])
    
    # 2. Draw the exact kinematic values from the respective MC pools
    parts = []
    if n_sig > 0 and S_mc_pool is not None and len(S_mc_pool) > 0:
        parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
    if n_bkg > 0 and B_mc_pool is not None and len(B_mc_pool) > 0:
        parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
        
    if parts:
        return np.concatenate(parts)
    return np.array([])
# ==============================================================================


# --- 2. Core Math (Unbinned) ---
def calc_nll_unbinned(params, len_obs, s_probs, b_probs):
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

    Returns:
    --------
    float
        The calculated NLL value.
    """
    expected_total, p_events = compute_rates_unbinned(params, s_probs, b_probs)
    
    # Handle the zero-observation case perfectly
    if len_obs == 0:
        return expected_total

    # Apply a heavy penalty (1e10) if the model predicts negative or zero rate 
    # for an event that actually occurred, avoiding fatal np.log(x <= 0) math domain errors.
    if np.any(p_events <= 0):
        return 1e10 
        
    return expected_total - np.sum(np.log(p_events))


# --- 3. Grid Search Optimizers (Unbinned) ---
def unconditional_fit_grid_unbinned(obs_events, S_pdf, B_pdf, full_grid_points):
    """
    Performs a brute-force global scan over the parameter space to find the unconditional MLE.
    
    Because evaluating the PDF function S_pdf(obs_events) for every parameter combination 
    can be incredibly slow in Python, this function evaluates the PDFs exactly ONCE for 
    the observed dataset, and reuses those static arrays across the entire grid loop.
    """
    len_obs = len(obs_events)
    # Pre-compute structural PDFs to avoid massive redundant computation inside the loop
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_params = full_grid_points[0].copy()
    
    for row in range(len(full_grid_points)):
        p = full_grid_points[row]
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

def conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, obs_events, S_pdf, B_pdf, cond_grid_points):
    """
    Performs a brute-force 1D conditional profile scan for the unbinned likelihood.
    Fixes one parameter of interest and scans the remaining grid.
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
                
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

def conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, obs_events, S_pdf, B_pdf, cond_grid_points):
    """
    Performs a brute-force 2D conditional profile scan for the unbinned likelihood.
    Fixes two parameters of interest to map joint contours.
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
                
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params


# --- 4. Toy Generators (Unbinned) ---
def generate_and_fit_toys_grid_unbinned_1d(test_val, fix_idx, true_params, n_params, S_pdf, B_pdf, 
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool):
    """
    Sequentially generates and evaluates 1D unbinned toys entirely using grid search.
    
    Used when gradient optimizers fail and external parallelization is disabled. 
    It evaluates the Profile Likelihood Ratio (cond_nll - uncond_nll) for each 
    simulated unbinned dataset.
    """
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, full_grid_points)
        cond_nll, _ = conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, toy_events, S_pdf, B_pdf, cond_grid_points)
        
        # Bounded at 0.0 to correct floating point inaccuracies when cond_nll ≈ uncond_nll
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics

def generate_and_fit_toys_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, true_params, n_params, S_pdf, B_pdf, 
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool):
    """
    Sequentially generates and evaluates 2D unbinned toys entirely using grid search.
    """
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, full_grid_points)
        cond_nll, _ = conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, toy_events, S_pdf, B_pdf, cond_grid_points)
        
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics