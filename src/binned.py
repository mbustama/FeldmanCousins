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
    Example provided: 3-parameter model.
    params[0] = param1
    params[1] = param2
    params[2] = param3
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