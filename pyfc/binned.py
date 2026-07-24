"""
Binned Likelihood and Feldman-Cousins Toy Generation Module

This file contains the core mathematical and statistical routines required for 
performing binned frequentist analyses, tailored for the Feldman-Cousins construction. 
It implements expected rate calculations, standard Poisson and finite-Monte Carlo 
(Poisson-Gamma mixture) likelihood evaluations, grid-based profile likelihood 
optimizations, and parametric bootstrap pseudo-experiment (toy) generation.

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np
import math

# --- 1. Dynamic Dependency Injection for Binned Math ---
try:
    from numba import njit, prange, set_num_threads
    NUMBA_AVAILABLE = True
    print("Numba detected: JIT compilation enabled for maximum speed.")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not found: Falling back to pure Python (will be significantly slower).")
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return wrapper
    prange = range
    def set_num_threads(n):
        pass


# ==============================================================================
# --- 2. USER DEFINED PHYSICS MODEL (BINNED) ---
# Modify this section to define how your N parameters map to the expected rates.
# ==============================================================================
@njit(fastmath=True, nogil=True)
def compute_rates_binned(params, S_template, B_template, S_sigma2, B_sigma2):
    """
    Maps an N-dimensional parameter array to the expected bin counts and variances.
    
    Statistical Theory:
    In a binned analysis, the expected number of events in each bin (mu) is modeled 
    as a linear combination of signal (S) and background (B) templates, scaled by 
    the parameters of interest. The variance (sigma2) propagates the uncertainty 
    from limited Monte Carlo statistics in the templates.

    Mathematical Expression:
    mu = (params[0] * params[1]) * S_template + params[2] * B_template
    sigma^2 = (params[0] * params[1])^2 * S_sigma2 + (params[2])^2 * B_sigma2

    Parameters:
    -----------
    params : array_like, 1D
        The parameters of the physical model.
        params[0] = param1 (e.g., cross-section scaling)
        params[1] = param2 (e.g., flux normalization)
        params[2] = param3 (e.g., background normalization)
    S_template : array_like, 1D
        Expected nominal signal counts per bin.
    B_template : array_like, 1D
        Expected nominal background counts per bin.
    S_sigma2 : array_like, 1D
        Variance (squared uncertainty) of the signal template per bin due to finite MC.
    B_sigma2 : array_like, 1D
        Variance (squared uncertainty) of the background template per bin due to finite MC.

    Returns:
    --------
    mu : array_like, 1D
        The total expected counts in each bin based on the parameters.
    sigma2 : array_like, 1D
        The total expected variance in each bin due to template uncertainties.
    """
    mu = params[0] * params[1] * S_template + params[2] * B_template
    sigma2 = ((params[0] * params[1])**2) * S_sigma2 + (params[2]**2) * B_sigma2
    
    return mu, sigma2
# ==============================================================================


# --- 3. Core Math (Binned) ---
@njit(fastmath=True, nogil=True)
def calc_nll(params, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc):
    """
    [BINNED] Computes the negative log-likelihood for arbitrary N parameters.
    
    Statistical Theory & Assumptions:
    This function computes the negative log-likelihood (NLL) for comparing observed 
    data counts against expected model counts. It handles two distinct statistical regimes:
    
    1. Standard Poisson Likelihood (use_finite_mc = False):
       Assumes template predictions have zero statistical uncertainty (infinite MC).
       It uses the Baker-Cousins chi-square equivalent form (based on the likelihood 
       ratio with a saturated model).
       Expression: NLL = 2 * SUM [ mu_i - n_i + n_i * ln(n_i / mu_i) ]
       
    2. Finite Monte Carlo Likelihood (use_finite_mc = True):
       Accounts for statistical uncertainty in the simulation templates. It marginalizes 
       over the unknown true Poisson rate using a Gamma conjugate prior, resulting in a 
       Negative Binomial (Poisson-Gamma mixture) likelihood.
       Expression: 
       alpha = (mu^2 / sigma^2) + 1
       beta = mu / sigma^2
       ln(L) = alpha*ln(beta) + ln(Gamma(n+alpha)) - (n+alpha)*ln(1+beta) - ln(Gamma(alpha))
       NLL = -2 * ln(L)

    Parameters:
    -----------
    params : array_like, 1D
        The physical model parameters to evaluate.
    N_obs : array_like, 1D
        The observed data counts (or toy data) in each bin.
    S_template : array_like, 1D
        Expected nominal signal counts per bin.
    B_template : array_like, 1D
        Expected nominal background counts per bin.
    S_sigma2 : array_like, 1D
        Variance of the signal template per bin.
    B_sigma2 : array_like, 1D
        Variance of the background template per bin.
    use_finite_mc : bool
        If True, applies the Poisson-Gamma mixture likelihood. If False, uses standard Poisson.

    Returns:
    --------
    nll : float
        The calculated negative log-likelihood value. Returns a high penalty (1e10) 
        if parameters yield unphysical (negative) expected counts.
    """
    nll = 0.0
    mu, sigma2_arr = compute_rates_binned(params, S_template, B_template, S_sigma2, B_sigma2)
    
    for i in range(len(N_obs)):
        mu_i = mu[i]
        n_obs = float(N_obs[i])
        
        if mu_i <= 0:
            if n_obs > 0:
                return 1e10  # Heavy penalty for unphysical expectations
            else:
                continue
                
        if use_finite_mc:
            sigma2 = sigma2_arr[i]
            if sigma2 > 1e-10:
                alpha = (mu_i**2) / sigma2 + 1.0
                beta = max(mu_i / sigma2, 1e-300) 
                
                lnL = (alpha * math.log(beta) 
                       + math.lgamma(n_obs + alpha) 
                       - (n_obs + alpha) * math.log(1.0 + beta) 
                       - math.lgamma(alpha))
                nll += -2.0 * lnL
            else:
                lnL_poisson = n_obs * math.log(mu_i) - mu_i
                nll += -2.0 * lnL_poisson
        else:
            if n_obs > 0:
                nll += 2.0 * (mu_i - n_obs + n_obs * math.log(n_obs / mu_i))
            else:
                nll += 2.0 * mu_i
                
    return nll


# --- 4. Grid Search Optimizers (Binned) ---
@njit(fastmath=True, nogil=True)
def unconditional_fit_grid(N_obs, S_template, B_template, full_grid_points, S_sigma2, B_sigma2, use_finite_mc):
    """
    Performs an unconditional maximum likelihood fit via exhaustive grid search.
    
    Statistical Theory:
    To construct the denominator of the Profile Likelihood Ratio (PLR) test statistic,
    we must find the global maximum of the likelihood function (minimum of the NLL) 
    allowing all parameters in the model to vary freely across their physical bounds.
    
    Expression: NLL_best = min( NLL(theta) ) for all theta in parameter space.

    Parameters:
    -----------
    N_obs : array_like, 1D
        The observed data or toy counts.
    S_template : array_like, 1D
        Expected signal counts.
    B_template : array_like, 1D
        Expected background counts.
    full_grid_points : array_like, 2D
        A pre-computed matrix where each row represents a complete N-dimensional 
        parameter combination to evaluate.
    S_sigma2, B_sigma2 : array_like, 1D
        Template variances.
    use_finite_mc : bool
        Flag to use finite MC likelihood formulation.

    Returns:
    --------
    min_nll : float
        The absolute minimum negative log-likelihood found in the grid.
    best_params : array_like, 1D
        The parameter combination that produced min_nll.
    """
    min_nll = 1e10
    best_params = full_grid_points[0].copy()
    
    for row in range(len(full_grid_points)):
        p = full_grid_points[row]
        nll = calc_nll(p, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

@njit(fastmath=True, nogil=True)
def conditional_fit_grid_1d(test_val, fix_idx, n_params, N_obs, S_template, B_template, cond_grid_points, S_sigma2, B_sigma2, use_finite_mc):
    """
    Performs a conditional maximum likelihood fit with one parameter fixed (Profiling).
    
    Statistical Theory:
    To construct the numerator of the Profile Likelihood Ratio for a 1D confidence 
    interval, the likelihood is maximized (NLL minimized) while keeping the parameter 
    of interest fixed to a specific test value. The remaining (nuisance) parameters 
    are allowed to vary (profiled out).
    
    Expression: NLL_cond = min( NLL(theta_test, nu) ) over all nuisance parameters nu.

    Parameters:
    -----------
    test_val : float
        The fixed value for the parameter of interest.
    fix_idx : int
        The index of the parameter to fix.
    n_params : int
        Total number of parameters in the model.
    N_obs, S_template, B_template, S_sigma2, B_sigma2 : array_like, 1D
        Data counts, templates, and variances.
    cond_grid_points : array_like, 2D
        Pre-computed grid of the (N-1) free nuisance parameters to scan over.
    use_finite_mc : bool
        Flag to use finite MC likelihood formulation.

    Returns:
    --------
    min_nll : float
        The minimum negative log-likelihood given the fixed parameter.
    best_params : array_like, 1D
        The full parameter combination (including the fixed one) that produced min_nll.
    """
    min_nll = 1e10
    best_params = np.zeros(n_params)
    p = np.zeros(n_params)
    p[fix_idx] = test_val
    
    for row in range(len(cond_grid_points)):
        free_p = cond_grid_points[row]
        free_i = 0
        for i in range(n_params):
            if i != fix_idx:
                p[i] = free_p[free_i]
                free_i += 1
                
        nll = calc_nll(p, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

@njit(fastmath=True, nogil=True)
def conditional_fit_grid_2d(test_vA, test_vB, fix_A, fix_B, n_params, N_obs, S_template, B_template, cond_grid_points, S_sigma2, B_sigma2, use_finite_mc):
    """
    Performs a conditional maximum likelihood fit with two parameters fixed.
    
    Statistical Theory:
    Analogous to the 1D case, this computes the profiled likelihood for a 2D joint 
    confidence region. Two parameters of interest are fixed to a grid point, and 
    the NLL is minimized over all remaining (N-2) nuisance parameters.

    Parameters:
    -----------
    test_vA, test_vB : float
        The fixed values for the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    n_params : int
        Total number of parameters in the model.
    N_obs, S_template, B_template, S_sigma2, B_sigma2 : array_like, 1D
        Data counts, templates, and variances.
    cond_grid_points : array_like, 2D
        Pre-computed grid of the (N-2) free nuisance parameters to scan over.
    use_finite_mc : bool
        Flag to use finite MC likelihood formulation.

    Returns:
    --------
    min_nll : float
        The minimum negative log-likelihood given the two fixed parameters.
    best_params : array_like, 1D
        The full parameter combination that produced min_nll.
    """
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
                
        nll = calc_nll(p, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params


# --- 5. Toy Generators (Binned) ---
@njit(fastmath=True, parallel=True, nogil=True)
def generate_and_fit_toys_grid_1d(test_val, fix_idx, true_params, n_params, S_template, B_template, 
                                  full_grid_points, cond_grid_points, n_toys, S_sigma2, B_sigma2, use_finite_mc):
    """
    Generates Poisson toys and computes the 1D test statistic distribution (Feldman-Cousins).
    
    Statistical Theory & Assumptions:
    To ensure proper frequentist coverage (as per the Feldman-Cousins unified approach), 
    we cannot assume the Profile Likelihood Ratio strictly follows an asymptotic chi-square 
    distribution (Wilks' Theorem), especially near physical boundaries. 
    Instead, we build the empirical distribution of the test statistic (t) by running 
    parametric bootstraps (toys).
    
    For each toy:
    1. A synthetic dataset (toy_N) is sampled from Poisson(mu_true).
    2. The unconditional global minimum NLL is found for the toy.
    3. The conditional minimum NLL (fixing the parameter of interest) is found.
    4. The test statistic is calculated: t = NLL_cond - NLL_uncond.
    (Due to numerical imprecision in grid searches, t is bounded at a minimum of 0.0).

    Parameters:
    -----------
    test_val : float
        The fixed value for the parameter of interest.
    fix_idx : int
        The index of the fixed parameter.
    true_params : array_like, 1D
        The parameter combination used to generate the true expected counts for the toys.
    n_params : int
        Total number of parameters.
    S_template, B_template, S_sigma2, B_sigma2 : array_like, 1D
        Templates and variances.
    full_grid_points : array_like, 2D
        Grid for the unconditional fit.
    cond_grid_points : array_like, 2D
        Grid for the conditional (nuisance parameter) fit.
    n_toys : int
        The number of pseudo-experiments to generate.
    use_finite_mc : bool
        Flag to use finite MC likelihood during toy fitting.

    Returns:
    --------
    t_statistics : array_like, 1D
        Array of length `n_toys` containing the calculated test statistic for each toy.
    """
    t_statistics = np.zeros(n_toys)
    n_bins = len(S_template)
    mu_true, _ = compute_rates_binned(true_params, S_template, B_template, S_sigma2, B_sigma2)
    
    for t in prange(n_toys):
        toy_N = np.zeros(n_bins)
        for i in range(n_bins):
            toy_N[i] = np.random.poisson(mu_true[i])
            
        uncond_nll, _ = unconditional_fit_grid(toy_N, S_template, B_template, full_grid_points, S_sigma2, B_sigma2, use_finite_mc)
        cond_nll, _ = conditional_fit_grid_1d(test_val, fix_idx, n_params, toy_N, S_template, B_template, cond_grid_points, S_sigma2, B_sigma2, use_finite_mc)
        
        t_statistics[t] = max(0.0, cond_nll - uncond_nll) 
    return t_statistics

@njit(fastmath=True, parallel=True, nogil=True)
def generate_and_fit_toys_grid_2d(test_vA, test_vB, fix_A, fix_B, true_params, n_params, S_template, B_template, 
                                  full_grid_points, cond_grid_points, n_toys, S_sigma2, B_sigma2, use_finite_mc):
    """
    Generates Poisson toys and computes the 2D test statistic distribution.
    
    Statistical Theory:
    Similar to the 1D case, this function calculates empirical distributions of the 
    Profile Likelihood Ratio test statistic for 2D parameter joint combinations.
    t = NLL_cond_2D - NLL_uncond.

    Parameters:
    -----------
    test_vA, test_vB : float
        The fixed values for the two parameters of interest.
    fix_A, fix_B : int
        The indices of the parameters being fixed.
    true_params : array_like, 1D
        The parameter combination used to generate expected counts for the toys.
    n_params : int
        Total number of parameters.
    S_template, B_template, S_sigma2, B_sigma2 : array_like, 1D
        Templates and variances.
    full_grid_points : array_like, 2D
        Grid for the unconditional fit.
    cond_grid_points : array_like, 2D
        Grid for the conditional (nuisance parameter) fit.
    n_toys : int
        The number of pseudo-experiments to generate.
    use_finite_mc : bool
        Flag to use finite MC likelihood during toy fitting.

    Returns:
    --------
    t_statistics : array_like, 1D
        Array of length `n_toys` containing the calculated test statistic for each toy.
    """
    t_statistics = np.zeros(n_toys)
    n_bins = len(S_template)
    mu_true, _ = compute_rates_binned(true_params, S_template, B_template, S_sigma2, B_sigma2)
    
    for t in prange(n_toys):
        toy_N = np.zeros(n_bins)
        for i in range(n_bins):
            toy_N[i] = np.random.poisson(mu_true[i])
            
        uncond_nll, _ = unconditional_fit_grid(toy_N, S_template, B_template, full_grid_points, S_sigma2, B_sigma2, use_finite_mc)
        cond_nll, _ = conditional_fit_grid_2d(test_vA, test_vB, fix_A, fix_B, n_params, toy_N, S_template, B_template, cond_grid_points, S_sigma2, B_sigma2, use_finite_mc)
        
        t_statistics[t] = max(0.0, cond_nll - uncond_nll) 
    return t_statistics