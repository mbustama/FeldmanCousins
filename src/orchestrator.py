import numpy as np
import warnings
import logging
import concurrent.futures
import itertools
import json
import argparse
import sys
import os

# --- Local Component Imports ---
from binned import (
    calc_nll, 
    compute_rates_binned,
    unconditional_fit_grid, 
    conditional_fit_grid_1d,
    conditional_fit_grid_2d,
    generate_and_fit_toys_grid_1d,
    generate_and_fit_toys_grid_2d,
    NUMBA_AVAILABLE,
    set_num_threads
)
from unbinned import (
    calc_nll_unbinned, 
    generate_unbinned_toy, 
    unconditional_fit_grid_unbinned, 
    conditional_fit_grid_unbinned_1d,
    conditional_fit_grid_unbinned_2d,
    generate_and_fit_toys_grid_unbinned_1d,
    generate_and_fit_toys_grid_unbinned_2d
)

# --- Dynamic Dependency Injection for Orchestrator ---
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import ultranest
    ULTRANEST_AVAILABLE = True
except ImportError:
    ULTRANEST_AVAILABLE = False

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# --- 1. Scipy Continuous Optimizers ---
def unconditional_fit_scipy(data, S_model, B_model, n_params, bounds_list, seed=None, 
                            likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
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


# --- 3. Top-Level ProcessPool Worker for Unbinned ---
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


# --- 4. Python Vectorized Toy Generator ---
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


# --- 5. Corner Plot Generator (Generalized N-Dimensional) ---
def generate_corner_plot(results, config):
    if not MATPLOTLIB_AVAILABLE:
        return
        
    n_params = config.get("n_params", 1)
    p_names = config.get("param_names", [f"param{i+1}" for i in range(n_params)])
    cl_list = config["cl"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Base configuration dynamically sizing figure space based on N
    fig_size = max(5 * n_params, 10)
    fig, axs = plt.subplots(n_params, n_params, figsize=(fig_size, fig_size), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    
    # Handle the 1D exception (n_params == 1 returns a single axis, not a matrix)
    if n_params == 1:
        axs = np.array([[axs]])

    # --- Diagonal: 1D Profiles ---
    if config.get("compute_1D_intervals", True):
        for i in range(n_params):
            ax = axs[i, i]
            key_test = f"1d_test_p{i+1}"
            
            if key_test not in results: 
                continue
            
            x_test = results[key_test]
            t_dat = results[f"1d_t_data_p{i+1}"]
            
            ax.plot(x_test, t_dat, color='black', lw=1.5, label='Test Statistic')
            
            for idx, c in enumerate(cl_list):
                t_crit = results[f"1d_t_critical_p{i+1}"][c]
                acc = results[f"1d_accepted_p{i+1}"][c]
                ax.plot(x_test, t_crit, '--', color=colors[idx % len(colors)])
                ax.fill_between(x_test, 0, t_dat, where=acc, color=colors[idx % len(colors)], alpha=0.3)
                
            ax.set_ylim(bottom=0)
            ax.set_xlim(x_test[0], x_test[-1])
            ax.set_title(p_names[i])
            
            if i != n_params - 1: 
                ax.tick_params(labelbottom=False)
            else: 
                ax.set_xlabel(p_names[i])
            
            if i != 0: 
                ax.tick_params(labelleft=False)
            else: 
                ax.set_ylabel(r"$\Delta$ NLL")

    # --- Off-Diagonal Lower Triangle: 2D Contours ---
    if config.get("compute_2D_intervals", True) and n_params > 1:
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            row, col = fix_B, fix_A
            ax = axs[row, col]
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            
            key_x = f"2d_test_p{fix_A+1}_{pair_name}"
            if key_x not in results: 
                continue
                
            grid_x = results[key_x]
            grid_y = results[f"2d_test_p{fix_B+1}_{pair_name}"]
            t_dat_2d = results[f"2d_t_data_{pair_name}"]
            
            X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
            ax.pcolormesh(X, Y, t_dat_2d, cmap='Blues', shading='auto', alpha=0.2)
            
            for idx, c in enumerate(cl_list):
                acc_2d = results[f"2d_accepted_{pair_name}"][c]
                
                # Safeguard to prevent contour crashing
                if np.any(acc_2d) and not np.all(acc_2d):
                    ax.contour(X, Y, acc_2d.astype(int), levels=[0.5], colors=[colors[idx % len(colors)]], linewidths=2)
                
                # Dummy line for exact legends (only set on the very bottom-left plot)
                if row == n_params - 1 and col == 0: 
                    ax.plot([], [], color=colors[idx % len(colors)], linewidth=2, label=f'{c} CL')
            
            if "best_fit" in results:
                best_vals = results["best_fit"]
                ax.scatter([best_vals[fix_A]], [best_vals[fix_B]], color='black', marker='*', s=150)
                
            ax.set_xlim(grid_x[0], grid_x[-1])
            ax.set_ylim(grid_y[0], grid_y[-1])
            
            if row != n_params - 1: 
                ax.tick_params(labelbottom=False)
            else: 
                ax.set_xlabel(p_names[col])
            
            if col != 0: 
                ax.tick_params(labelleft=False)
            else: 
                ax.set_ylabel(p_names[row])
            
            if row == n_params - 1 and col == 0: 
                ax.legend(loc='upper right', fontsize=8)

    # Hide upper triangle completely
    for i in range(n_params):
        for j in range(n_params):
            if i < j:
                axs[i, j].axis('off')

    plot_path = os.path.join(config.get("save_directory", "."), "fc_corner_plot.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


# --- 6. Main Feldman-Cousins Wrapper ---
def compute_fc_intervals(data, S_model, B_model, grids, 
                         cl=[0.90], n_toys=2000, strategy="scipy", num_cores=None, verbose=1,
                         adaptive_toys=True, toy_batch_size=200, 
                         sparsify_grid=True, warm_start=True,
                         likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                         output_file=None, save_log=False, save_directory="fc_output",
                         use_finite_mc_correction_binned=True, S_sigma2=None, B_sigma2=None,
                         compute_1D_intervals=True, compute_2D_intervals=True, param_names=None):
    
    os.makedirs(save_directory, exist_ok=True)
    n_params = len(grids)
    
    # Setup Logging Architecture
    run_logger = logging.getLogger("FC_Orchestrator")
    run_logger.setLevel(logging.INFO)
    if save_log and not run_logger.handlers:
        fh = logging.FileHandler(os.path.join(save_directory, 'fc_run.log'))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        run_logger.addHandler(fh)
        
    def log_print(msg):
        if verbose > 0: print(msg)
        if save_log: run_logger.info(msg)

    if isinstance(cl, (float, int)): 
        cl = [float(cl)]
    if ULTRANEST_AVAILABLE and verbose < 2: 
        logging.getLogger("ultranest").setLevel(logging.WARNING)
    if NUMBA_AVAILABLE and num_cores is not None: 
        set_num_threads(num_cores)
        
    if likelihood_type == "binned":
        if S_sigma2 is None: S_sigma2 = np.zeros_like(S_model)
        if B_sigma2 is None: B_sigma2 = np.zeros_like(B_model)
    else:
        S_sigma2, B_sigma2 = None, None

    log_print(f"--- FC Construction Initiated ({n_params}-Parameter Model) ---")
    log_print(f"Modes -> 1D Intervals: {compute_1D_intervals} | 2D Intervals: {compute_2D_intervals}")
    log_print(f"Strategy: {strategy.upper()} | Cores: {num_cores if num_cores else 'Max'} | Likelihood: {likelihood_type.upper()}")
    
    results = {}
    bounds_list = [(g[0], g[-1]) for g in grids]
    disable_tqdm = (verbose == 0) or not TQDM_AVAILABLE
    
    # --- PHASE 0: Fit global unconditional data ONCE ---
    full_grid_points = np.array(list(itertools.product(*grids)), dtype=np.float64)
    
    if strategy == "grid":
        if likelihood_type == "binned":
            data_uncond_nll, best_params = unconditional_fit_grid(data, S_model, B_model, full_grid_points, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
        else:
            data_uncond_nll, best_params = unconditional_fit_grid_unbinned(data, S_model, B_model, full_grid_points)
    elif strategy in ["ultranest", "hybrid"]:
        data_uncond_nll, best_params = unconditional_fit_ultranest(data, S_model, B_model, n_params, bounds_list, verbose, likelihood_type, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
    elif strategy == "scipy":
        data_uncond_nll, best_params = unconditional_fit_scipy(data, S_model, B_model, n_params, bounds_list, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
    
    results["best_fit"] = best_params

    # --- PHASE 1: Compute 1D Intervals ---
    if compute_1D_intervals:
        log_print("Executing 1D Parameter Scans...")
        for p_idx in range(n_params):
            grid_test = grids[p_idx]
            free_grids = [grids[i] for i in range(n_params) if i != p_idx]
            
            if not free_grids:
                cond_grid_points = np.zeros((1, 0), dtype=np.float64)
            else:
                cond_grid_points = np.array(list(itertools.product(*free_grids)), dtype=np.float64)
                
            t_data_arr = np.zeros(len(grid_test))
            prof_params_arr = np.zeros((len(grid_test), n_params))
            t_crit_dict = {c: np.zeros(len(grid_test)) for c in cl}
            
            for i, pt in enumerate(tqdm(grid_test, desc=f"1D Data (p{p_idx+1})", disable=disable_tqdm)):
                if strategy == "grid":
                    if likelihood_type == "binned":
                        cond_nll, prof_p = conditional_fit_grid_1d(pt, p_idx, n_params, data, S_model, B_model, cond_grid_points, S_sigma2, B_sigma2, use_finite_mc_correction_binned) 
                    else:
                        cond_nll, prof_p = conditional_fit_grid_unbinned_1d(pt, p_idx, n_params, data, S_model, B_model, cond_grid_points)
                elif strategy in ["ultranest", "hybrid"]:
                    cond_nll, prof_p = conditional_fit_1d_ultranest(pt, p_idx, n_params, data, S_model, B_model, bounds_list, verbose, likelihood_type, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                elif strategy == "scipy":
                    cond_nll, prof_p = conditional_fit_1d_scipy(pt, p_idx, n_params, data, S_model, B_model, bounds_list, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
                
                prof_params_arr[i] = prof_p
                t_data_arr[i] = max(0.0, cond_nll - data_uncond_nll)

            for i, pt in enumerate(tqdm(grid_test, desc=f"1D Toys (p{p_idx+1})", disable=disable_tqdm)):
                true_params = prof_params_arr[i]
                
                if strategy == "grid":
                    if likelihood_type == "binned":
                        t_stats = generate_and_fit_toys_grid_1d(pt, p_idx, true_params, n_params, S_model, B_model, full_grid_points, cond_grid_points, n_toys, S_sigma2, B_sigma2, use_finite_mc_correction_binned) 
                    else:
                        t_stats = generate_and_fit_toys_grid_unbinned_1d(pt, p_idx, true_params, n_params, S_model, B_model, full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool)
                else:
                    t_stats = generate_and_fit_toys_python(true_params, n_params, "1d", p_idx, None, None, pt, None, S_model, B_model, bounds_list, n_toys, strategy, num_cores, 0, likelihood_type, S_mc_pool, B_mc_pool, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                
                t_stats.sort()
                for c in cl: 
                    t_crit_dict[c][i] = t_stats[min(int(c * n_toys), n_toys - 1)]
                    
            results[f"1d_test_p{p_idx+1}"] = grid_test
            results[f"1d_t_data_p{p_idx+1}"] = t_data_arr
            results[f"1d_prof_params_p{p_idx+1}"] = prof_params_arr
            results[f"1d_t_critical_p{p_idx+1}"] = t_crit_dict
            results[f"1d_accepted_p{p_idx+1}"] = {c: t_data_arr <= t_crit_dict[c] for c in cl}

    # --- PHASE 2: Compute 2D Intervals ---
    if compute_2D_intervals and n_params > 1:
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            gridA, gridB = grids[fix_A], grids[fix_B]
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            log_print(f"Executing 2D Grid Scan for {pair_name} ({len(gridA)}x{len(gridB)})...")
            
            free_grids = [grids[i] for i in range(n_params) if i != fix_A and i != fix_B]
            if not free_grids:
                cond_grid_points = np.zeros((1, 0), dtype=np.float64)
            else:
                cond_grid_points = np.array(list(itertools.product(*free_grids)), dtype=np.float64)
            
            results[f"2d_test_p{fix_A+1}_{pair_name}"] = gridA
            results[f"2d_test_p{fix_B+1}_{pair_name}"] = gridB
            results[f"2d_t_data_{pair_name}"] = np.zeros((len(gridA), len(gridB)))
            results[f"2d_t_critical_{pair_name}"] = {c: np.zeros((len(gridA), len(gridB))) for c in cl}
            results[f"2d_accepted_{pair_name}"] = {c: np.zeros((len(gridA), len(gridB)), dtype=bool) for c in cl}
            
            # Helper function to evaluate toys for a specific 2D coordinate
            def eval_2d_point(i, j):
                p_A, p_B = gridA[i], gridB[j]
                
                # 1. Exact Data NLL
                if strategy == "grid":
                    if likelihood_type == "binned": 
                        cond_nll, prof_p = conditional_fit_grid_2d(p_A, p_B, fix_A, fix_B, n_params, data, S_model, B_model, cond_grid_points, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                    else: 
                        cond_nll, prof_p = conditional_fit_grid_unbinned_2d(p_A, p_B, fix_A, fix_B, n_params, data, S_model, B_model, cond_grid_points)
                elif strategy in ["ultranest", "hybrid"]:
                    cond_nll, prof_p = conditional_fit_2d_ultranest(p_A, p_B, fix_A, fix_B, n_params, data, S_model, B_model, bounds_list, verbose=0, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
                elif strategy == "scipy":
                    cond_nll, prof_p = conditional_fit_2d_scipy(p_A, p_B, fix_A, fix_B, n_params, data, S_model, B_model, bounds_list, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
                
                results[f"2d_t_data_{pair_name}"][i, j] = max(0.0, cond_nll - data_uncond_nll)
                true_params = prof_p
                
                # 2. Toy evaluation
                if strategy == "grid":
                    if likelihood_type == "binned": 
                        t_stats = generate_and_fit_toys_grid_2d(p_A, p_B, fix_A, fix_B, true_params, n_params, S_model, B_model, full_grid_points, cond_grid_points, n_toys, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                    else: 
                        t_stats = generate_and_fit_toys_grid_unbinned_2d(p_A, p_B, fix_A, fix_B, true_params, n_params, S_model, B_model, full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool)
                else:
                    t_stats = generate_and_fit_toys_python(true_params, n_params, "2d", None, fix_A, fix_B, p_A, p_B, S_model, B_model, bounds_list, n_toys, strategy, num_cores, 0, likelihood_type, S_mc_pool, B_mc_pool, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                
                t_stats.sort()
                for c in cl: 
                    results[f"2d_t_critical_{pair_name}"][c][i, j] = t_stats[min(int(c * n_toys), n_toys - 1)]

            # Sparsification or Full Grid
            if sparsify_grid:
                step_A = max(1, len(gridA) // 5)
                step_B = max(1, len(gridB) // 5)
                coarse_i = sorted(list(set(list(range(0, len(gridA), step_A)) + [len(gridA)-1])))
                coarse_j = sorted(list(set(list(range(0, len(gridB), step_B)) + [len(gridB)-1])))
                coarse_pts = [(i, j) for i in coarse_i for j in coarse_j]
                
                log_print(f"2D Sparsification: Coarse pass on {len(coarse_pts)} structural points...")
                for pt in tqdm(coarse_pts, desc=f"2D Coarse {pair_name}", disable=disable_tqdm): 
                    eval_2d_point(*pt)
                    
                # Interpolate coarse results to find approximate boundaries
                if SCIPY_AVAILABLE:
                    from scipy.interpolate import RectBivariateSpline
                    for c in cl:
                        z = results[f"2d_t_critical_{pair_name}"][c][np.ix_(coarse_i, coarse_j)]
                        interp = RectBivariateSpline(gridA[coarse_i], gridB[coarse_j], z)
                        results[f"2d_t_critical_{pair_name}"][c] = interp(gridA, gridB)
                else:
                    for c in cl:
                        for i in range(len(gridA)):
                            for j in range(len(gridB)):
                                ni = min(coarse_i, key=lambda x: abs(x - i))
                                nj = min(coarse_j, key=lambda x: abs(x - j))
                                results[f"2d_t_critical_{pair_name}"][c][i, j] = results[f"2d_t_critical_{pair_name}"][c][ni, nj]
                
                # Identify perimeter points where acceptance changes
                refine_pts = set()
                for c in cl:
                    approx_acc = results[f"2d_t_data_{pair_name}"] <= results[f"2d_t_critical_{pair_name}"][c]
                    for i in range(len(gridA)):
                        for j in range(len(gridB)):
                            if approx_acc[i, j]:
                                is_bound = False
                                for di in [-1, 0, 1]:
                                    for dj in [-1, 0, 1]:
                                        ni = i + di
                                        nj = j + dj
                                        if 0 <= ni < len(gridA) and 0 <= nj < len(gridB):
                                            if not approx_acc[ni, nj]: 
                                                is_bound = True
                                                
                                if is_bound:
                                    for di in [-1, 0, 1]:
                                        for dj in [-1, 0, 1]:
                                            ni = i + di
                                            nj = j + dj
                                            if 0 <= ni < len(gridA) and 0 <= nj < len(gridB): 
                                                refine_pts.add((ni, nj))
                                                
                pts_to_eval = [p for p in refine_pts if p not in coarse_pts]
                log_print(f"2D Sparsification: Traced boundary. Running {len(pts_to_eval)} exact refinement points...")
            else:
                pts_to_eval = [(i,j) for i in range(len(gridA)) for j in range(len(gridB))]
                
            for pt in tqdm(pts_to_eval, desc=f"2D Edge {pair_name}", disable=disable_tqdm): 
                eval_2d_point(*pt)
            
            for c in cl: 
                results[f"2d_accepted_{pair_name}"][c] = results[f"2d_t_data_{pair_name}"] <= results[f"2d_t_critical_{pair_name}"][c]

    # --- Archiving & Plotting ---
    if output_file is not None:
        save_dict = {}
        save_dict["best_fit"] = results["best_fit"]
        
        if compute_1D_intervals:
            for p_idx in range(n_params):
                save_dict[f"1d_test_p{p_idx+1}"] = results[f"1d_test_p{p_idx+1}"]
                save_dict[f"1d_t_data_p{p_idx+1}"] = results[f"1d_t_data_p{p_idx+1}"]
                save_dict[f"1d_prof_params_p{p_idx+1}"] = results[f"1d_prof_params_p{p_idx+1}"]
                for c in cl:
                    save_dict[f"1d_t_critical_p{p_idx+1}_{c}"] = results[f"1d_t_critical_p{p_idx+1}"][c]
                    save_dict[f"1d_accepted_p{p_idx+1}_{c}"] = results[f"1d_accepted_p{p_idx+1}"][c]

        if compute_2D_intervals and n_params > 1:
            pairs = list(itertools.combinations(range(n_params), 2))
            for fix_A, fix_B in pairs:
                pair_name = f"p{fix_A+1}p{fix_B+1}"
                save_dict[f"2d_test_p{fix_A+1}_{pair_name}"] = results[f"2d_test_p{fix_A+1}_{pair_name}"]
                save_dict[f"2d_test_p{fix_B+1}_{pair_name}"] = results[f"2d_test_p{fix_B+1}_{pair_name}"]
                save_dict[f"2d_t_data_{pair_name}"] = results[f"2d_t_data_{pair_name}"]
                for c in cl:
                    save_dict[f"2d_t_critical_{pair_name}_{c}"] = results[f"2d_t_critical_{pair_name}"][c]
                    save_dict[f"2d_accepted_{pair_name}_{c}"] = results[f"2d_accepted_{pair_name}"][c]
                    
        np.savez(os.path.join(save_directory, output_file), **save_dict)
        log_print(f"Results explicitly saved to {os.path.join(save_directory, output_file)}.npz")

    # Pass config directly to ensure names/modes are respected during plot generation
    if compute_1D_intervals or (compute_2D_intervals and n_params > 1):
        generate_corner_plot(results, {
            "n_params": n_params,
            "compute_1D_intervals": compute_1D_intervals,
            "compute_2D_intervals": compute_2D_intervals,
            "param_names": param_names if param_names else [f"param{i+1}" for i in range(n_params)],
            "cl": cl,
            "save_directory": save_directory
        })

    return results


# --- 7. Config & CLI Logic ---
def generate_sample_config(filename="fc_config.json"):
    """Generates a sample JSON configuration file."""
    default_config = {
        "likelihood_type": "binned",
        "cl": [0.68, 0.90],
        "n_toys": 200,
        "strategy": "scipy",
        "num_cores": 4,
        "verbose": 1,
        "adaptive_toys": True,
        "toy_batch_size": 200,
        "sparsify_grid": True,
        "warm_start": True,
        "output_file": "fc_results",
        "save_log": True,
        "save_directory": "fc_output",
        "use_finite_mc_correction_binned": True,
        "compute_1D_intervals": True,
        "compute_2D_intervals": True,
        "param_names": ["param1", "param2", "param3"]
    }
    with open(filename, 'w') as f:
        json.dump(default_config, f, indent=4)
    print(f"Sample configuration written to {filename}")

def parse_arguments():
    """Parses hierarchy: Defaults -> JSON Config File -> CLI Arguments"""
    parser = argparse.ArgumentParser(description="Feldman-Cousins Confidence Intervals (N Parameter Model)")
    
    parser.add_argument('--config_file', type=str, help="Path to JSON config file", default=argparse.SUPPRESS)
    parser.add_argument('--generate_config', action='store_true', help="Generate a sample JSON config and exit")
    
    parser.add_argument('--likelihood_type', type=str, choices=["binned", "unbinned"], default=argparse.SUPPRESS)
    parser.add_argument('--cl', type=float, nargs='+', help="Confidence level(s) (e.g., 0.90 0.95)", default=argparse.SUPPRESS)
    parser.add_argument('--n_toys', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--strategy', type=str, choices=["grid", "scipy", "ultranest", "hybrid"], default=argparse.SUPPRESS)
    parser.add_argument('--num_cores', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--verbose', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--adaptive_toys', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--toy_batch_size', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--sparsify_grid', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--warm_start', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--output_file', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--save_log', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--save_directory', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--use_finite_mc_correction_binned', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--compute_1D_intervals', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--compute_2D_intervals', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--param_names', type=str, nargs='+', default=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    if args.generate_config:
        generate_sample_config()
        sys.exit(0)
        
    # Base Hardcoded Defaults
    config = {
        "likelihood_type": "binned",
        "cl": [0.90],
        "n_toys": 200,
        "strategy": "scipy",
        "num_cores": None,
        "verbose": 1,
        "adaptive_toys": True,
        "toy_batch_size": 200,
        "sparsify_grid": True,
        "warm_start": True,
        "output_file": None,
        "save_log": False,
        "save_directory": "fc_output",
        "use_finite_mc_correction_binned": True,
        "compute_1D_intervals": True,
        "compute_2D_intervals": True,
        "param_names": ["param1", "param2", "param3"]
    }
    
    if hasattr(args, 'config_file'):
        if os.path.exists(args.config_file):
            with open(args.config_file, 'r') as f:
                config.update(json.load(f))
        else:
            print(f"Warning: Config file {args.config_file} not found. Proceeding with defaults.")

    cli_args = vars(args)
    for key, value in cli_args.items():
        if key not in ['config_file', 'generate_config']:
            config[key] = value
            
    return config


# --- 8. Execution Examples ---
if __name__ == "__main__":
    
    config = parse_arguments()
    
    # Example Grids for 3-parameters
    grids = [
        np.linspace(0.5, 2.0, 15), # param1
        np.linspace(0.5, 2.0, 15), # param2
        np.linspace(0.5, 1.5, 15)  # param3
    ]
    
    if config["likelihood_type"] == "binned":
        print(f"\n--- Running BINNED Analysis Example ({len(grids)}-Parameter) | Modes -> 1D: {config['compute_1D_intervals']} | 2D: {config['compute_2D_intervals']} ---")
        S_template = np.array([0.1, 0.5, 2.0, 5.0])
        B_template = np.array([15.0, 5.0, 1.0, 0.1])
        
        S_sigma2 = S_template.copy() 
        B_sigma2 = B_template.copy()
        
        np.random.seed(42)
        # Expected from default compute_rates_binned: p1*p2*S + p3*B (Setting true params to 1.0)
        N_data_binned = np.random.poisson(1.0 * 1.0 * S_template + 1.0 * B_template)
        print(f"Mock Observed Data (Binned Counts): {N_data_binned}")
        
        fc_results = compute_fc_intervals(
            N_data_binned, S_template, B_template, grids, 
            cl=config["cl"], n_toys=config["n_toys"], strategy=config["strategy"], 
            num_cores=config["num_cores"], verbose=config["verbose"],
            adaptive_toys=config["adaptive_toys"], toy_batch_size=config["toy_batch_size"],
            sparsify_grid=config["sparsify_grid"], warm_start=config["warm_start"],
            likelihood_type="binned",
            output_file=config["output_file"], save_log=config["save_log"], save_directory=config["save_directory"],
            use_finite_mc_correction_binned=config["use_finite_mc_correction_binned"],
            S_sigma2=S_sigma2, B_sigma2=B_sigma2,
            compute_1D_intervals=config["compute_1D_intervals"],
            compute_2D_intervals=config["compute_2D_intervals"],
            param_names=config["param_names"]
        )

    elif config["likelihood_type"] == "unbinned":
        print(f"\n--- Running UNBINNED Analysis Example ({len(grids)}-Parameter) | Modes -> 1D: {config['compute_1D_intervals']} | 2D: {config['compute_2D_intervals']} ---")
        from scipy.stats import norm, expon
        
        def s_pdf_mock(x): return norm.pdf(x, loc=5.0, scale=1.0)
        def b_pdf_mock(x): return expon.pdf(x, scale=2.0)
        
        s_mc_pool = np.random.normal(loc=5.0, scale=1.0, size=5000)
        b_mc_pool = np.random.exponential(scale=2.0, size=5000)
        
        unbinned_data = np.concatenate([
            np.random.choice(s_mc_pool, size=2), # True p1*p2 = 2
            np.random.choice(b_mc_pool, size=3)  # True p3 = 3
        ])
        print(f"Mock Observed Unbinned Events: {np.round(unbinned_data, 2)}")
        
        fc_results = compute_fc_intervals(
            unbinned_data, s_pdf_mock, b_pdf_mock, 
            grids, 
            cl=config["cl"], n_toys=config["n_toys"], strategy=config["strategy"], 
            num_cores=config["num_cores"], verbose=config["verbose"],
            adaptive_toys=config["adaptive_toys"], toy_batch_size=config["toy_batch_size"],
            sparsify_grid=config["sparsify_grid"], warm_start=config["warm_start"],
            likelihood_type="unbinned", S_mc_pool=s_mc_pool, B_mc_pool=b_mc_pool,
            output_file=config["output_file"], save_log=config["save_log"], save_directory=config["save_directory"],
            compute_1D_intervals=config["compute_1D_intervals"],
            compute_2D_intervals=config["compute_2D_intervals"],
            param_names=config["param_names"]
        )