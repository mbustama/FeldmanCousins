"""
Monte Carlo Toy Generation and Fitting Module

This module manages the core frequentist computation of the PyFC package: 
the generation and parallel evaluation of pseudo-experiments (toys). 
To establish exact coverage in the Feldman-Cousins framework, the distribution 
of the test statistic must be empirically derived at each point in the parameter 
grid by simulating data under the null hypothesis, and fitting it both unconditionally 
and conditionally.

This file specifically handles the execution when using continuous optimizers 
(SciPy, UltraNest) and manages efficient parallelization by routing workloads 
to either Thread or Process pools depending on Python's Global Interpreter Lock (GIL) 
limitations for the requested likelihood type.

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np
import concurrent.futures
import warnings

from .optimizers import (
    unconditional_fit_scipy, conditional_fit_1d_scipy, conditional_fit_2d_scipy,
    unconditional_fit_ultranest, conditional_fit_1d_ultranest, conditional_fit_2d_ultranest
)

def _worker_unbinned_toy(args):
    """
    Module-level explicit worker function for unbinned toy generation and fitting.
    
    Architecture Note:
    This function must reside at the top level of the module to bypass Python's 
    `pickle` limitations. `concurrent.futures.ProcessPoolExecutor` must serialize 
    functions to pass them to worker processes. Closures or nested functions 
    (like `fit_single_toy` below) cannot be pickled natively.

    Parameters:
    -----------
    args : tuple
        A packed tuple containing all necessary arguments:
        (toy_index, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, 
         t_vA, t_vB, S_model, B_model, bounds_list, S_mc_pool, B_mc_pool, strategy,
         compute_rates_func, generate_toy_func)

    Returns:
    --------
    float
        The computed Profile Likelihood Ratio test statistic for this single toy.
    """
    # Unpack the monolithic argument tuple required for multiprocessing
    (t, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB, 
     S_model, B_model, bounds_list, S_mc_pool, B_mc_pool, strategy, 
     compute_rates_func, generate_toy_func) = args
    
    # Generate the simulated unbinned dataset by bootstrapping from the MC pools
    toy_data = generate_toy_func(true_params, S_mc_pool, B_mc_pool)
    
    if strategy == "scipy" or strategy == "hybrid":
        # Seeding strategy: Initialize the optimizer at the true parameters that generated 
        # the toy. The global minimum is statistically guaranteed to be in this neighborhood.
        seed_p = true_params.copy()
        
        # 1. Unconditional fit (Denominator of the PLR)
        uncond_nll, _ = unconditional_fit_scipy(toy_data, S_model, B_model, n_params, bounds_list, compute_rates_func, seed=seed_p, likelihood_type="unbinned")
        
        # 2. Conditional fit (Numerator of the PLR)
        if fit_mode == "1d":
            seed_free_1d = [true_params[i] for i in range(n_params) if i != fix_idx] if len(true_params) > 1 else None
            cond_nll, _ = conditional_fit_1d_scipy(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, seed=seed_free_1d, likelihood_type="unbinned")
        elif fit_mode == "2d":
            seed_free_2d = [true_params[i] for i in range(n_params) if i not in (fix_A, fix_B)] if len(true_params) > 2 else None
            cond_nll, _ = conditional_fit_2d_scipy(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, seed=seed_free_2d, likelihood_type="unbinned")
            
    elif strategy == "ultranest":
        uncond_nll, _ = unconditional_fit_ultranest(toy_data, S_model, B_model, n_params, bounds_list, compute_rates_func, verbose=0, likelihood_type="unbinned")
        if fit_mode == "1d":
            cond_nll, _ = conditional_fit_1d_ultranest(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, verbose=0, likelihood_type="unbinned")
        elif fit_mode == "2d":
            cond_nll, _ = conditional_fit_2d_ultranest(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, verbose=0, likelihood_type="unbinned")
            
    return max(0.0, cond_nll - uncond_nll)


def generate_and_fit_toys_python(true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB,
                                 S_model, B_model, bounds_list, n_toys, strategy, num_cores=None, verbose=1,
                                 likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                                 S_sigma2=None, B_sigma2=None, use_finite_mc=False,
                                 compute_rates_func=None, generate_toy_func=None):
    """
    Handles threaded generation and fitting of MC toys for 1D profiling and 2D contours.
    
    Statistical Context:
    To evaluate if a grid point $t_{test}$ belongs in the confidence interval, we generate 
    `n_toys` pseudo-experiments assuming the null hypothesis ($\theta = t_{test}$, with 
    nuisance parameters at their conditional maximum likelihood values `true_params`). 
    Each toy is then fit unconditionally and conditionally to build the empirical distribution 
    of the Profile Likelihood Ratio.

    Parallelization Context:
    - Unbinned PDFs often rely on pure Python functions (like `scipy.stats.norm.pdf`). 
      These are strictly bound by the Global Interpreter Lock (GIL). Multithreading provides 
      zero speedup. Thus, we branch to a `ProcessPoolExecutor` to spawn distinct processes.
    - Binned operations (vectorized NumPy math) naturally release the GIL. Thus, we can 
      use a lighter-weight `ThreadPoolExecutor` and avoid the heavy overhead of process spawning 
      and memory IPC.

    Parameters:
    -----------
    true_params : array_like
        The physical parameters (test values + profiled nuisance values) used to generate toys.
    n_params : int
        Total number of parameters in the model.
    fit_mode : str
        Either "1d" or "2d".
    fix_idx, fix_A, fix_B : int or None
        Indices of the fixed parameters depending on the fit mode.
    t_vA, t_vB : float
        The test values to fix the parameters at during the conditional fit.
    S_model, B_model, bounds_list : various
        Models and boundary constraints.
    n_toys : int
        Number of pseudo-experiments to generate and evaluate.
    strategy : str
        Optimizer selection ("scipy", "ultranest", "hybrid").
    num_cores : int
        Number of parallel workers.
    verbose : int
        Logging level.
    likelihood_type : str
        "binned" or "unbinned".
    S_mc_pool, B_mc_pool : array_like, optional
        Source pools for bootstrapping unbinned events.
    S_sigma2, B_sigma2 : array_like, optional
        Template variances for finite MC corrections.
    use_finite_mc : bool
        Toggle for Poisson-Gamma mixture likelihoods.
    compute_rates_func : callable
        User-provided expected rates mapping function.
    generate_toy_func : callable
        User-provided unbinned toy bootstrapper.

    Returns:
    --------
    t_stats : np.ndarray
        Array of the computed test statistics for all `n_toys`.
    """
    # --- Branch 1: Unbinned Data (Process-based parallelism) ---
    if likelihood_type == "unbinned":
        args_list = [(t, true_params, n_params, fit_mode, fix_idx, fix_A, fix_B, t_vA, t_vB, 
                      S_model, B_model, bounds_list, S_mc_pool, B_mc_pool, strategy, 
                      compute_rates_func, generate_toy_func) for t in range(n_toys)]
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                t_stats = list(executor.map(_worker_unbinned_toy, args_list))
            return np.array(t_stats)
        except Exception as e:
            warnings.warn(f"ProcessPoolExecutor failed. Falling back to ThreadPoolExecutor. Error: {e}")
            pass
            
    # --- Branch 2: Binned Data (Thread-based parallelism) or Unbinned Fallback ---
    if likelihood_type == "binned":
        mu_true, _ = compute_rates_func(true_params, S_model, B_model, S_sigma2, B_sigma2)
        toys_binned_data = np.random.poisson(mu_true, size=(n_toys, len(S_model)))
    
    def fit_single_toy(t):
        if likelihood_type == "binned":
            toy_data = toys_binned_data[t]
        else:
            toy_data = generate_toy_func(true_params, S_mc_pool, B_mc_pool)
        
        if strategy == "scipy" or strategy == "hybrid":
            seed_p = true_params.copy()
            uncond_nll, _ = unconditional_fit_scipy(toy_data, S_model, B_model, n_params, bounds_list, compute_rates_func, seed=seed_p, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            
            if fit_mode == "1d":
                seed_free_1d = [true_params[i] for i in range(n_params) if i != fix_idx] if len(true_params) > 1 else None
                cond_nll, _ = conditional_fit_1d_scipy(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, seed=seed_free_1d, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif fit_mode == "2d":
                seed_free_2d = [true_params[i] for i in range(n_params) if i not in (fix_A, fix_B)] if len(true_params) > 2 else None
                cond_nll, _ = conditional_fit_2d_scipy(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, seed=seed_free_2d, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
                    
        elif strategy == "ultranest":
            uncond_nll, _ = unconditional_fit_ultranest(toy_data, S_model, B_model, n_params, bounds_list, compute_rates_func, verbose=0, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            
            if fit_mode == "1d":
                cond_nll, _ = conditional_fit_1d_ultranest(t_vA, fix_idx, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, verbose=0, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif fit_mode == "2d":
                cond_nll, _ = conditional_fit_2d_ultranest(t_vA, t_vB, fix_A, fix_B, n_params, toy_data, S_model, B_model, bounds_list, compute_rates_func, verbose=0, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            
        return max(0.0, cond_nll - uncond_nll)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        t_stats = list(executor.map(fit_single_toy, range(n_toys)))
        
    return np.array(t_stats)