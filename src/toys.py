import numpy as np
import concurrent.futures
import warnings

from binned import compute_rates_binned
from unbinned import generate_unbinned_toy
from optimizers import (
    unconditional_fit_scipy, conditional_fit_1d_scipy, conditional_fit_2d_scipy,
    unconditional_fit_ultranest, conditional_fit_1d_ultranest, conditional_fit_2d_ultranest
)

def _worker_unbinned_toy(args):
    """
    Module-level explicit worker to bypass Python closure-pickling limitations 
    inherent to multiprocessing pools.
    """
    t, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB, S_model, B_model, bounds_list, S_mc_pool, B_mc_pool, strategy = args
    
    toy_data = generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool)
    
    if strategy == "scipy" or strategy == "hybrid":
        seed_p = true_params.copy()
        uncond_nll, _ = unconditional_fit_scipy(toy_data, S_model, B_model, n_params, bounds_list, seed=seed_p, likelihood_type="unbinned")
        if fit_mode == "1d":
            seed_free_1d = [true_params[i] for i in range(n_params) if i != fix_idx] if len(true_params) > 1 else None
            cond_nll, _ = conditional_fit_1d_scipy(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, seed=seed_free_1d, likelihood_type="unbinned")
        elif fit_mode == "2d":
            seed_free_2d = [true_params[i] for i in range(n_params) if i not in (fix_A, fix_B)] if len(true_params) > 2 else None
            cond_nll, _ = conditional_fit_2d_scipy(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, seed=seed_free_2d, likelihood_type="unbinned")
            
    elif strategy == "ultranest":
        uncond_nll, _ = unconditional_fit_ultranest(toy_data, S_model, B_model, n_params, bounds_list, verbose=0, likelihood_type="unbinned")
        if fit_mode == "1d":
            cond_nll, _ = conditional_fit_1d_ultranest(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, verbose=0, likelihood_type="unbinned")
        elif fit_mode == "2d":
            cond_nll, _ = conditional_fit_2d_ultranest(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, verbose=0, likelihood_type="unbinned")
            
    return max(0.0, cond_nll - uncond_nll)

def generate_and_fit_toys_python(true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB,
                                 S_model, B_model, bounds_list, n_toys, strategy, num_cores=None, verbose=1,
                                 likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                                 S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """
    Handles threaded generation & fitting for 1D profiling and 2D fixed tests.
    Dynamically branches to ProcessPoolExecutor for GIL-locked Unbinned routines.
    """
    if likelihood_type == "unbinned":
        args_list = [(t, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB, 
                      S_model, B_model, bounds_list, S_mc_pool, B_mc_pool, strategy) for t in range(n_toys)]
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                t_stats = list(executor.map(_worker_unbinned_toy, args_list))
            return np.array(t_stats)
        except Exception as e:
            warnings.warn(f"ProcessPoolExecutor failed (likely due to non-pickleable PDF functions like Lambdas). Falling back to ThreadPoolExecutor. Error: {e}")
            pass
            
    # Binned Execution (GIL bypassed by Numba) or Unbinned Fallback
    if likelihood_type == "binned":
        mu_true, _ = compute_rates_binned(true_params, S_model, B_model, S_sigma2, B_sigma2)
        toys_binned_data = np.random.poisson(mu_true, size=(n_toys, len(S_model)))
    
    def fit_single_toy(t):
        if likelihood_type == "binned":
            toy_data = toys_binned_data[t]
        else:
            toy_data = generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool)
        
        if strategy == "scipy" or strategy == "hybrid":
            seed_p = true_params.copy()
            uncond_nll, _ = unconditional_fit_scipy(toy_data, S_model, B_model, n_params, bounds_list, seed=seed_p, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            
            if fit_mode == "1d":
                seed_free_1d = [true_params[i] for i in range(n_params) if i != fix_idx] if len(true_params) > 1 else None
                cond_nll, _ = conditional_fit_1d_scipy(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, seed=seed_free_1d, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif fit_mode == "2d":
                seed_free_2d = [true_params[i] for i in range(n_params) if i not in (fix_A, fix_B)] if len(true_params) > 2 else None
                cond_nll, _ = conditional_fit_2d_scipy(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, seed=seed_free_2d, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
                    
        elif strategy == "ultranest":
            uncond_nll, _ = unconditional_fit_ultranest(toy_data, S_model, B_model, n_params, bounds_list, verbose=0, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            
            if fit_mode == "1d":
                cond_nll, _ = conditional_fit_1d_ultranest(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, verbose=0, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif fit_mode == "2d":
                cond_nll, _ = conditional_fit_2d_ultranest(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, verbose=0, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            
        return max(0.0, cond_nll - uncond_nll)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        t_stats = list(executor.map(fit_single_toy, range(n_toys)))
        
    return np.array(t_stats)