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


# --- 2. Core Math (Binned) ---
@njit(fastmath=True, nogil=True)
def calc_nll(sig_scale, bkg_scale, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc):
    """
    [BINNED] Computes the negative log-likelihood.
    Includes optional effective likelihood correction for finite MC statistics (1901.04645v2).
    Releases the GIL to allow Python multi-threading to scale perfectly.
    """
    nll = 0.0
    for i in range(len(N_obs)):
        mu = sig_scale * S_template[i] + bkg_scale * B_template[i]
        n_obs = float(N_obs[i])
        
        if mu <= 0:
            if n_obs > 0:
                return 1e10  # Heavy penalty for unphysical expectations
            else:
                continue
                
        if use_finite_mc:
            # Scale the variance of the template: weights multiplied by scale factor => var * scale^2
            sigma2 = (sig_scale**2) * S_sigma2[i] + (bkg_scale**2) * B_sigma2[i]
            
            if sigma2 > 1e-10:
                # Effective log-likelihood from 1901.04645v2 Eq 3.16
                alpha = (mu**2) / sigma2 + 1.0
                beta = max(mu / sigma2, 1e-300) # Guard against extreme precision loss
                
                lnL = (alpha * math.log(beta) 
                       + math.lgamma(n_obs + alpha) 
                       - math.lgamma(n_obs + 1.0) 
                       - (n_obs + alpha) * math.log(1.0 + beta) 
                       - math.lgamma(alpha))
                nll += -2.0 * lnL
            else:
                # Smooth fallback to Standard Poisson NLL for very large MC statistics
                lnL_poisson = n_obs * math.log(mu) - mu - math.lgamma(n_obs + 1.0)
                nll += -2.0 * lnL_poisson
        else:
            # Standard deviance form for purely analytical Poisson
            if n_obs > 0:
                nll += 2.0 * (mu - n_obs + n_obs * math.log(n_obs / mu))
            else:
                nll += 2.0 * mu
                
    return nll


# --- 3. Grid Search Optimizers (Binned) ---
@njit(fastmath=True, nogil=True)
def unconditional_fit_grid(N_obs, S_template, B_template, sig_grid, bkg_grid, S_sigma2, B_sigma2, use_finite_mc):
    min_nll = 1e10
    best_sig = sig_grid[0]
    best_bkg = bkg_grid[0]
    
    for s in sig_grid:
        for b in bkg_grid:
            nll = calc_nll(s, b, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc)
            if nll < min_nll:
                min_nll = nll
                best_sig = s
                best_bkg = b
    return min_nll, best_sig, best_bkg

@njit(fastmath=True, nogil=True)
def conditional_fit_grid(test_sig, N_obs, S_template, B_template, bkg_grid, S_sigma2, B_sigma2, use_finite_mc):
    min_nll = 1e10
    best_bkg = bkg_grid[0]
    
    for b in bkg_grid:
        nll = calc_nll(test_sig, b, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc)
        if nll < min_nll:
            min_nll = nll
            best_bkg = b
    return min_nll, best_bkg

@njit(fastmath=True, parallel=True, nogil=True)
def generate_and_fit_toys_grid(test_sig, profiled_bkg, S_template, B_template, 
                               sig_grid, bkg_grid, n_toys, S_sigma2, B_sigma2, use_finite_mc):
    t_statistics = np.zeros(n_toys)
    n_bins = len(S_template)
    
    for t in prange(n_toys):
        toy_N = np.zeros(n_bins)
        for i in range(n_bins):
            mu_true = test_sig * S_template[i] + profiled_bkg * B_template[i]
            toy_N[i] = np.random.poisson(mu_true)
            
        uncond_nll, _, _ = unconditional_fit_grid(toy_N, S_template, B_template, sig_grid, bkg_grid, S_sigma2, B_sigma2, use_finite_mc)
        cond_nll, _ = conditional_fit_grid(test_sig, toy_N, S_template, B_template, bkg_grid, S_sigma2, B_sigma2, use_finite_mc)
        
        t_stat = cond_nll - uncond_nll
        t_statistics[t] = max(0.0, t_stat) 
    return t_statistics