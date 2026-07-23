import numpy as np
import warnings
import logging
import concurrent.futures
import json
import argparse
import sys
import os

# --- Local Component Imports ---
from binned import (
    calc_nll, 
    unconditional_fit_grid, 
    conditional_fit_grid,
    conditional_fit_grid_profile_sig,
    generate_and_fit_toys_grid,
    generate_and_fit_toys_grid_profile_sig,
    generate_and_fit_toys_grid_2d,
    NUMBA_AVAILABLE,
    set_num_threads
)
from unbinned import (
    calc_nll_unbinned, 
    generate_unbinned_toy, 
    unconditional_fit_grid_unbinned, 
    conditional_fit_grid_unbinned,
    conditional_fit_grid_unbinned_profile_sig,
    generate_and_fit_toys_grid_unbinned,
    generate_and_fit_toys_grid_unbinned_profile_sig,
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
def unconditional_fit_scipy(data, S_model, B_model, sig_bounds, bkg_bounds, seed=None, 
                            likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        def cost(params):
            return calc_nll_unbinned(params[0], params[1], len_obs, s_probs, b_probs)
    else:
        def cost(params):
            return calc_nll(params[0], params[1], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = seed if seed is not None else [(sig_bounds[0]+sig_bounds[1])/2.0, (bkg_bounds[0]+bkg_bounds[1])/2.0]
    res = optimize.minimize(cost, x0=x0, bounds=[sig_bounds, bkg_bounds], method='L-BFGS-B')
    return res.fun, res.x[0], res.x[1]

def conditional_fit_scipy(test_sig, data, S_model, B_model, bkg_bounds, seed=None, 
                          likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        def cost(bkg):
            return calc_nll_unbinned(test_sig, bkg[0], len_obs, s_probs, b_probs)
    else:
        def cost(bkg):
            return calc_nll(test_sig, bkg[0], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = [seed] if seed is not None else [(bkg_bounds[0]+bkg_bounds[1])/2.0]
    res = optimize.minimize(cost, x0=x0, bounds=[bkg_bounds], method='L-BFGS-B')
    return res.fun, res.x[0]

def conditional_fit_scipy_profile_sig(test_bkg, data, S_model, B_model, sig_bounds, seed=None, 
                                      likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        def cost(sig):
            return calc_nll_unbinned(sig[0], test_bkg, len_obs, s_probs, b_probs)
    else:
        def cost(sig):
            return calc_nll(sig[0], test_bkg, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = [seed] if seed is not None else [(sig_bounds[0]+sig_bounds[1])/2.0]
    res = optimize.minimize(cost, x0=x0, bounds=[sig_bounds], method='L-BFGS-B')
    return res.fun, res.x[0]


# --- 2. UltraNest Optimizers ---
def unconditional_fit_ultranest(data, S_model, B_model, sig_bounds, bkg_bounds, verbose=1, 
                                likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def prior_transform(cube):
        sig = cube[0] * (sig_bounds[1] - sig_bounds[0]) + sig_bounds[0]
        bkg = cube[1] * (bkg_bounds[1] - bkg_bounds[0]) + bkg_bounds[0]
        return np.array([sig, bkg])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        def log_likelihood(params):
            return -calc_nll_unbinned(params[0], params[1], len_obs, s_probs, b_probs)
    else:
        def log_likelihood(params):
            return -calc_nll(params[0], params[1], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    sampler = ultranest.ReactiveNestedSampler(['sig_scale', 'bkg_scale'], log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    best_sig, best_bkg = result['maximum_likelihood']['point']
    return -result['maximum_likelihood']['logl'], best_sig, best_bkg

def conditional_fit_ultranest(test_sig, data, S_model, B_model, bkg_bounds, verbose=1, 
                              likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def prior_transform(cube):
        bkg = cube[0] * (bkg_bounds[1] - bkg_bounds[0]) + bkg_bounds[0]
        return np.array([bkg])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        def log_likelihood(params):
            return -calc_nll_unbinned(test_sig, params[0], len_obs, s_probs, b_probs)
    else:
        def log_likelihood(params):
            return -calc_nll(test_sig, params[0], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    sampler = ultranest.ReactiveNestedSampler(['bkg_scale'], log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    return -result['maximum_likelihood']['logl'], result['maximum_likelihood']['point'][0]

def conditional_fit_ultranest_profile_sig(test_bkg, data, S_model, B_model, sig_bounds, verbose=1, 
                                          likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def prior_transform(cube):
        sig = cube[0] * (sig_bounds[1] - sig_bounds[0]) + sig_bounds[0]
        return np.array([sig])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        def log_likelihood(params):
            return -calc_nll_unbinned(params[0], test_bkg, len_obs, s_probs, b_probs)
    else:
        def log_likelihood(params):
            return -calc_nll(params[0], test_bkg, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    sampler = ultranest.ReactiveNestedSampler(['sig_scale'], log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    return -result['maximum_likelihood']['logl'], result['maximum_likelihood']['point'][0]


# --- 3. Top-Level ProcessPool Worker for Unbinned ---
def _worker_unbinned_toy(args):
    """
    Module-level explicit worker to bypass Python closure-pickling limitations 
    inherent to multiprocessing pools.
    """
    t, test_sig, test_bkg, S_model, B_model, sig_bounds, bkg_bounds, S_mc_pool, B_mc_pool, strategy, mode = args
    toy_data = generate_unbinned_toy(test_sig, test_bkg, S_mc_pool, B_mc_pool)
    
    if strategy == "scipy" or strategy == "hybrid":
        uncond_nll, _, _ = unconditional_fit_scipy(toy_data, S_model, B_model, sig_bounds, bkg_bounds, seed=[test_sig, test_bkg], likelihood_type="unbinned")
        if mode == "1d_sig":
            cond_nll, _ = conditional_fit_scipy(test_sig, toy_data, S_model, B_model, bkg_bounds, seed=test_bkg, likelihood_type="unbinned")
        elif mode == "1d_bkg":
            cond_nll, _ = conditional_fit_scipy_profile_sig(test_bkg, toy_data, S_model, B_model, sig_bounds, seed=test_sig, likelihood_type="unbinned")
        elif mode == "2d":
            len_obs = len(toy_data)
            s_probs = S_model(toy_data) if len_obs > 0 else np.array([])
            b_probs = B_model(toy_data) if len_obs > 0 else np.array([])
            cond_nll = calc_nll_unbinned(test_sig, test_bkg, len_obs, s_probs, b_probs)
            
    elif strategy == "ultranest":
        uncond_nll, _, _ = unconditional_fit_ultranest(toy_data, S_model, B_model, sig_bounds, bkg_bounds, verbose=0, likelihood_type="unbinned")
        if mode == "1d_sig":
            cond_nll, _ = conditional_fit_ultranest(test_sig, toy_data, S_model, B_model, bkg_bounds, verbose=0, likelihood_type="unbinned")
        elif mode == "1d_bkg":
            cond_nll, _ = conditional_fit_ultranest_profile_sig(test_bkg, toy_data, S_model, B_model, sig_bounds, verbose=0, likelihood_type="unbinned")
        elif mode == "2d":
            len_obs = len(toy_data)
            s_probs = S_model(toy_data) if len_obs > 0 else np.array([])
            b_probs = B_model(toy_data) if len_obs > 0 else np.array([])
            cond_nll = calc_nll_unbinned(test_sig, test_bkg, len_obs, s_probs, b_probs)
            
    return max(0.0, cond_nll - uncond_nll)

# --- 4. Python Vectorized Toy Generator ---
def generate_and_fit_toys_python(test_sig, test_bkg, S_model, B_model, 
                                 sig_bounds, bkg_bounds, n_toys, strategy, num_cores=None, verbose=1,
                                 likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                                 S_sigma2=None, B_sigma2=None, use_finite_mc=False, mode="1d_sig"):
    """
    Handles threaded generation & fitting for 1D profiling and 2D fixed tests.
    Dynamically branches to ProcessPoolExecutor for GIL-locked Unbinned routines.
    """
    
    if likelihood_type == "unbinned":
        args_list = [(t, test_sig, test_bkg, S_model, B_model, sig_bounds, bkg_bounds, S_mc_pool, B_mc_pool, strategy, mode) for t in range(n_toys)]
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                t_stats = list(executor.map(_worker_unbinned_toy, args_list))
            return np.array(t_stats)
        except Exception as e:
            warnings.warn(f"ProcessPoolExecutor failed (likely due to non-pickleable PDF functions like Lambdas). Falling back to ThreadPoolExecutor. Error: {e}")
            # Fails gracefully through to the ThreadPool block below
            pass
            
    # Binned Execution (GIL bypassed by Numba) or Unbinned Fallback
    if likelihood_type == "binned":
        mu_true = test_sig * S_model + test_bkg * B_model
        toys_binned_data = np.random.poisson(mu_true, size=(n_toys, len(S_model)))
    
    def fit_single_toy(t):
        if likelihood_type == "binned":
            toy_data = toys_binned_data[t]
        else:
            toy_data = generate_unbinned_toy(test_sig, test_bkg, S_mc_pool, B_mc_pool)
        
        if strategy == "scipy" or strategy == "hybrid":
            uncond_nll, _, _ = unconditional_fit_scipy(toy_data, S_model, B_model, sig_bounds, bkg_bounds, seed=[test_sig, test_bkg], 
                                                       likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            if mode == "1d_sig":
                cond_nll, _ = conditional_fit_scipy(test_sig, toy_data, S_model, B_model, bkg_bounds, seed=test_bkg, 
                                                    likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif mode == "1d_bkg":
                cond_nll, _ = conditional_fit_scipy_profile_sig(test_bkg, toy_data, S_model, B_model, sig_bounds, seed=test_sig, 
                                                                likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif mode == "2d":
                if likelihood_type == "binned":
                    cond_nll = calc_nll(test_sig, test_bkg, toy_data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
                else:
                    len_obs = len(toy_data)
                    s_probs = S_model(toy_data) if len_obs > 0 else np.array([])
                    b_probs = B_model(toy_data) if len_obs > 0 else np.array([])
                    cond_nll = calc_nll_unbinned(test_sig, test_bkg, len_obs, s_probs, b_probs)
                    
        elif strategy == "ultranest":
            uncond_nll, _, _ = unconditional_fit_ultranest(toy_data, S_model, B_model, sig_bounds, bkg_bounds, verbose=0, 
                                                           likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            if mode == "1d_sig":
                cond_nll, _ = conditional_fit_ultranest(test_sig, toy_data, S_model, B_model, bkg_bounds, verbose=0, 
                                                        likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif mode == "1d_bkg":
                cond_nll, _ = conditional_fit_ultranest_profile_sig(test_bkg, toy_data, S_model, B_model, sig_bounds, verbose=0, 
                                                                    likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            elif mode == "2d":
                if likelihood_type == "binned":
                    cond_nll = calc_nll(test_sig, test_bkg, toy_data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
                else:
                    len_obs = len(toy_data)
                    s_probs = S_model(toy_data) if len_obs > 0 else np.array([])
                    b_probs = B_model(toy_data) if len_obs > 0 else np.array([])
                    cond_nll = calc_nll_unbinned(test_sig, test_bkg, len_obs, s_probs, b_probs)
            
        return max(0.0, cond_nll - uncond_nll)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        t_stats = list(executor.map(fit_single_toy, range(n_toys)))
        
    return np.array(t_stats)


# --- 5. Corner Plot Generator ---
def generate_corner_plot(results, config):
    if not MATPLOTLIB_AVAILABLE:
        return
        
    p_names = config.get("param_names", ["Signal Scale", "Background Scale"])
    cl_list = config["cl"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # gridspec_kw tightly snaps the plots together for a true corner aesthetic
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})

    # --- Top-Left: 1D Signal ---
    if config.get("compute_1D_intervals", True) and "t_data" in results:
        ax = axs[0, 0]
        sig_test = results["test_sig"]
        t_dat = results["t_data"]
        
        ax.plot(sig_test, t_dat, color='black', lw=1.5, label='Test Statistic')
        for idx, c in enumerate(cl_list):
            t_crit = results["t_critical"][c]
            acc = results["accepted"][c]
            ax.plot(sig_test, t_crit, '--', color=colors[idx%len(colors)], label=f'{c} CL Threshold')
            ax.fill_between(sig_test, 0, t_dat, where=acc, color=colors[idx%len(colors)], alpha=0.3)
            
        ax.set_ylabel(r"$\Delta$ NLL (Test Stat)")
        ax.tick_params(labelbottom=False) # Hide X labels to align with 2D contour below
        ax.legend(loc='upper center', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_title(f"1D Profile: {p_names[0]}")

    # --- Bottom-Right: 1D Background ---
    if config.get("compute_1D_intervals", True) and "1d_t_data_bkg" in results:
        ax = axs[1, 1]
        bkg_test = results["1d_test_bkg"]
        t_dat_b = results["1d_t_data_bkg"]
        
        ax.plot(bkg_test, t_dat_b, color='black', lw=1.5, label='Test Statistic')
        for idx, c in enumerate(cl_list):
            t_crit_b = results["1d_t_critical_bkg"][c]
            acc_b = results["1d_accepted_bkg"][c]
            ax.plot(bkg_test, t_crit_b, '--', color=colors[idx%len(colors)])
            ax.fill_between(bkg_test, 0, t_dat_b, where=acc_b, color=colors[idx%len(colors)], alpha=0.3)
            
        ax.set_xlabel(p_names[1])
        ax.tick_params(labelleft=False) # Hide Y labels to align with 2D contour adjacent
        ax.set_ylim(bottom=0)
        ax.set_title(f"1D Profile: {p_names[1]}")

    # --- Bottom-Left: 2D Contour ---
    ax_2d = axs[1, 0]
    if config.get("compute_2D_intervals", True) and "2d_accepted" in results:
        S, B = np.meshgrid(results["2d_test_sig"], results["2d_test_bkg"], indexing='ij')
        
        # Plot underlying Test Statistic heat map so boundaries are always visible
        t_data_2d = results["2d_t_data"]
        ax_2d.pcolormesh(S, B, t_data_2d, cmap='Blues', shading='auto', alpha=0.2)
        
        for idx, c in enumerate(cl_list):
            acc_2d = results["2d_accepted"][c]
            # Safeguard to prevent contour crashing if exactly 0% or 100% of grid is accepted
            if np.any(acc_2d) and not np.all(acc_2d):
                ax_2d.contour(S, B, acc_2d.astype(int), levels=[0.5], colors=[colors[idx%len(colors)]], linewidths=2)
            # Dummy plot handle strictly for legend mapping
            ax_2d.plot([], [], color=colors[idx%len(colors)], linewidth=2, label=f'{c} CL Region')
            
        if "best_fit_sig" in results and "best_fit_bkg" in results:
            ax_2d.scatter([results["best_fit_sig"]], [results["best_fit_bkg"]], color='black', marker='*', s=150, label='Global Best Fit')
            
        ax_2d.legend(loc='upper right', fontsize=8)
    else:
        ax_2d.text(0.5, 0.5, '2D Interval Disabled\n(compute_2D_intervals=False)', 
                   horizontalalignment='center', verticalalignment='center', transform=ax_2d.transAxes)

    ax_2d.set_xlabel(p_names[0])
    ax_2d.set_ylabel(p_names[1])
    
    # Force physical axis sharing and tight limitations
    if "test_sig" in results:
        axs[0,0].set_xlim(results["test_sig"][0], results["test_sig"][-1])
        ax_2d.set_xlim(results["test_sig"][0], results["test_sig"][-1])
    if "1d_test_bkg" in results:
        axs[1,1].set_xlim(results["1d_test_bkg"][0], results["1d_test_bkg"][-1])
        ax_2d.set_ylim(results["1d_test_bkg"][0], results["1d_test_bkg"][-1])

    # Top-Right: Remove entirely
    axs[0, 1].axis('off')

    plot_path = os.path.join(config.get("save_directory", "."), "fc_corner_plot.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


# --- 6. Main Feldman-Cousins Wrapper ---
def compute_fc_intervals(data, S_model, B_model, 
                         sig_test_points, bkg_grid, 
                         cl=[0.90], n_toys=2000, strategy="scipy", num_cores=None, verbose=1,
                         adaptive_toys=True, toy_batch_size=200, 
                         sparsify_grid=True, warm_start=True,
                         likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                         output_file=None, save_log=False, save_directory="fc_output",
                         use_finite_mc_correction_binned=True, S_sigma2=None, B_sigma2=None,
                         compute_1D_intervals=True, compute_2D_intervals=True, param_names=None):
    
    os.makedirs(save_directory, exist_ok=True)
    
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

    if isinstance(cl, (float, int)): cl = [float(cl)]
    if ULTRANEST_AVAILABLE and verbose < 2: logging.getLogger("ultranest").setLevel(logging.WARNING)
    if NUMBA_AVAILABLE and num_cores is not None: set_num_threads(num_cores)
        
    if likelihood_type == "binned":
        if S_sigma2 is None: S_sigma2 = np.zeros_like(S_model)
        if B_sigma2 is None: B_sigma2 = np.zeros_like(B_model)
    else:
        S_sigma2, B_sigma2 = None, None

    log_print(f"--- FC Construction Initiated ---")
    log_print(f"Modes -> 1D Intervals: {compute_1D_intervals} | 2D Intervals: {compute_2D_intervals}")
    log_print(f"Strategy: {strategy.upper()} | Cores: {num_cores if num_cores else 'Max'} | Likelihood: {likelihood_type.upper()}")
    
    results = {}
    sig_grid, sig_bounds = sig_test_points, (sig_test_points[0], sig_test_points[-1])
    bkg_bounds = (bkg_grid[0], bkg_grid[-1])
    disable_tqdm = (verbose == 0) or not TQDM_AVAILABLE
    
    # --- PHASE 0: Fit global unconditional data ONCE ---
    if strategy == "grid":
        if likelihood_type == "binned":
            data_uncond_nll, best_sig_data, best_bkg_data = unconditional_fit_grid(data, S_model, B_model, sig_grid, bkg_grid, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
        else:
            data_uncond_nll, best_sig_data, best_bkg_data = unconditional_fit_grid_unbinned(data, S_model, B_model, sig_grid, bkg_grid)
    elif strategy in ["ultranest", "hybrid"]:
        data_uncond_nll, best_sig_data, best_bkg_data = unconditional_fit_ultranest(data, S_model, B_model, sig_bounds, bkg_bounds, verbose, likelihood_type, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
    elif strategy == "scipy":
        data_uncond_nll, best_sig_data, best_bkg_data = unconditional_fit_scipy(data, S_model, B_model, sig_bounds, bkg_bounds, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
    
    results["best_fit_sig"] = best_sig_data
    results["best_fit_bkg"] = best_bkg_data

    # Helper function for modular 1D processing
    def process_1d_grid(test_points, mode):
        t_data_arr = np.zeros(len(test_points))
        profiled_arr = np.zeros(len(test_points))
        t_crit_dict = {c: np.zeros(len(test_points)) for c in cl}
        
        last_seed = None
        for i, pt in enumerate(tqdm(test_points, desc=f"1D Fit Data ({mode})", disable=disable_tqdm)):
            if strategy == "grid":
                if mode == "1d_sig":
                    if likelihood_type == "binned":
                        data_cond_nll, prof_val = conditional_fit_grid(pt, data, S_model, B_model, bkg_grid, S_sigma2, B_sigma2, use_finite_mc_correction_binned) 
                    else:
                        data_cond_nll, prof_val = conditional_fit_grid_unbinned(pt, data, S_model, B_model, bkg_grid)
                else:
                    if likelihood_type == "binned":
                        data_cond_nll, prof_val = conditional_fit_grid_profile_sig(pt, data, S_model, B_model, sig_grid, S_sigma2, B_sigma2, use_finite_mc_correction_binned) 
                    else:
                        data_cond_nll, prof_val = conditional_fit_grid_unbinned_profile_sig(pt, data, S_model, B_model, sig_grid)
            elif strategy in ["ultranest", "hybrid"]:
                if mode == "1d_sig":
                    data_cond_nll, prof_val = conditional_fit_ultranest(pt, data, S_model, B_model, bkg_bounds, verbose, likelihood_type, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                else:
                    data_cond_nll, prof_val = conditional_fit_ultranest_profile_sig(pt, data, S_model, B_model, sig_bounds, verbose, likelihood_type, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
            elif strategy == "scipy":
                seed_val = last_seed if warm_start else None
                if mode == "1d_sig":
                    data_cond_nll, prof_val = conditional_fit_scipy(pt, data, S_model, B_model, bkg_bounds, seed=seed_val, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
                else:
                    data_cond_nll, prof_val = conditional_fit_scipy_profile_sig(pt, data, S_model, B_model, sig_bounds, seed=seed_val, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
            
            profiled_arr[i] = prof_val
            t_data_arr[i] = max(0.0, data_cond_nll - data_uncond_nll)
            if warm_start: last_seed = prof_val

        for i, pt in enumerate(tqdm(test_points, desc=f"1D Toys ({mode})", disable=disable_tqdm)):
            if strategy == "grid":
                if mode == "1d_sig":
                    if likelihood_type == "binned":
                        t_stats = generate_and_fit_toys_grid(pt, profiled_arr[i], S_model, B_model, sig_grid, bkg_grid, n_toys, S_sigma2, B_sigma2, use_finite_mc_correction_binned) 
                    else:
                        t_stats = generate_and_fit_toys_grid_unbinned(pt, profiled_arr[i], S_model, B_model, sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool)
                else:
                    if likelihood_type == "binned":
                        t_stats = generate_and_fit_toys_grid_profile_sig(pt, profiled_arr[i], S_model, B_model, sig_grid, bkg_grid, n_toys, S_sigma2, B_sigma2, use_finite_mc_correction_binned) 
                    else:
                        t_stats = generate_and_fit_toys_grid_unbinned_profile_sig(pt, profiled_arr[i], S_model, B_model, sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool)
            else:
                p_sig, p_bkg = (pt, profiled_arr[i]) if mode == "1d_sig" else (profiled_arr[i], pt)
                t_stats = generate_and_fit_toys_python(p_sig, p_bkg, S_model, B_model, sig_bounds, bkg_bounds, n_toys, strategy, num_cores, 0, likelihood_type, S_mc_pool, B_mc_pool, S_sigma2, B_sigma2, use_finite_mc_correction_binned, mode=mode)
            
            t_stats.sort()
            for c in cl:
                t_crit_dict[c][i] = t_stats[min(int(c * n_toys), n_toys - 1)]
                
        accepted_dict = {c: t_data_arr <= t_crit_dict[c] for c in cl}
        return t_data_arr, profiled_arr, t_crit_dict, accepted_dict

    # --- PHASE 1: Compute 1D Intervals ---
    if compute_1D_intervals:
        log_print("Executing 1D Parameter Scans...")
        # Signal 1D
        results["test_sig"] = sig_grid
        t_dat, p_bkg, t_crit, acc = process_1d_grid(sig_grid, "1d_sig")
        results["t_data"] = t_dat
        results["profiled_bkg"] = p_bkg
        results["t_critical"] = t_crit
        results["accepted"] = acc
        
        # Background 1D
        results["1d_test_bkg"] = bkg_grid
        t_dat_b, p_sig, t_crit_b, acc_b = process_1d_grid(bkg_grid, "1d_bkg")
        results["1d_t_data_bkg"] = t_dat_b
        results["profiled_sig"] = p_sig
        results["1d_t_critical_bkg"] = t_crit_b
        results["1d_accepted_bkg"] = acc_b

    # --- PHASE 2: Compute 2D Intervals ---
    if compute_2D_intervals:
        log_print(f"Executing 2D Pairwise Grid Scan ({len(sig_grid)}x{len(bkg_grid)})...")
        results["2d_test_sig"] = sig_grid
        results["2d_test_bkg"] = bkg_grid
        results["2d_t_data"] = np.zeros((len(sig_grid), len(bkg_grid)))
        results["2d_t_critical"] = {c: np.zeros((len(sig_grid), len(bkg_grid))) for c in cl}
        results["2d_accepted"] = {c: np.zeros((len(sig_grid), len(bkg_grid)), dtype=bool) for c in cl}
        
        # 1. Evaluate Data NLL on exact 2D grid
        for i, s in enumerate(sig_grid):
            for j, b in enumerate(bkg_grid):
                if likelihood_type == "binned":
                    data_cond_nll = calc_nll(s, b, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                else:
                    len_obs = len(data)
                    s_probs = S_model(data) if len_obs > 0 else np.array([])
                    b_probs = B_model(data) if len_obs > 0 else np.array([])
                    data_cond_nll = calc_nll_unbinned(s, b, len_obs, s_probs, b_probs)
                results["2d_t_data"][i, j] = max(0.0, data_cond_nll - data_uncond_nll)

        # Helper function to evaluate toys for a specific 2D coordinate
        def eval_2d_point(i, j):
            s, b = sig_grid[i], bkg_grid[j]
            if strategy == "grid":
                if likelihood_type == "binned":
                    t_stats = generate_and_fit_toys_grid_2d(s, b, S_model, B_model, sig_grid, bkg_grid, n_toys, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                else:
                    t_stats = generate_and_fit_toys_grid_unbinned_2d(s, b, S_model, B_model, sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool)
            else:
                t_stats = generate_and_fit_toys_python(s, b, S_model, B_model, sig_bounds, bkg_bounds, n_toys, strategy, num_cores, 0, likelihood_type, S_mc_pool, B_mc_pool, S_sigma2, B_sigma2, use_finite_mc_correction_binned, mode="2d")
            
            t_stats.sort()
            for c in cl:
                results["2d_t_critical"][c][i, j] = t_stats[min(int(c * n_toys), n_toys - 1)]

        # 2. Sparsified Perimeter Evaluation
        if sparsify_grid:
            step_s = max(1, len(sig_grid) // 5)
            step_b = max(1, len(bkg_grid) // 5)
            coarse_i = sorted(list(set(list(range(0, len(sig_grid), step_s)) + [len(sig_grid)-1])))
            coarse_j = sorted(list(set(list(range(0, len(bkg_grid), step_b)) + [len(bkg_grid)-1])))
            coarse_pts = [(i, j) for i in coarse_i for j in coarse_j]
            
            log_print(f"2D Sparsification: Coarse pass on {len(coarse_pts)} structural points...")
            for pt in tqdm(coarse_pts, desc="2D Coarse Grid", disable=disable_tqdm):
                eval_2d_point(*pt)
                
            # Interpolate coarse results to find approximate boundaries
            if SCIPY_AVAILABLE:
                from scipy.interpolate import RectBivariateSpline
                for c in cl:
                    grid_s = sig_grid[coarse_i]
                    grid_b = bkg_grid[coarse_j]
                    z = results["2d_t_critical"][c][np.ix_(coarse_i, coarse_j)]
                    interp = RectBivariateSpline(grid_s, grid_b, z)
                    results["2d_t_critical"][c] = interp(sig_grid, bkg_grid)
            else:
                for c in cl:
                    for i in range(len(sig_grid)):
                        for j in range(len(bkg_grid)):
                            ni = min(coarse_i, key=lambda x: abs(x - i))
                            nj = min(coarse_j, key=lambda x: abs(x - j))
                            results["2d_t_critical"][c][i, j] = results["2d_t_critical"][c][ni, nj]
            
            # Identify perimeter points where acceptance changes
            refine_pts = set()
            for c in cl:
                approx_acc = results["2d_t_data"] <= results["2d_t_critical"][c]
                for i in range(len(sig_grid)):
                    for j in range(len(bkg_grid)):
                        if approx_acc[i, j]:
                            is_bound = False
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < len(sig_grid) and 0 <= nj < len(bkg_grid):
                                        if not approx_acc[ni, nj]:
                                            is_bound = True
                            if is_bound:
                                for di in [-1, 0, 1]:
                                    for dj in [-1, 0, 1]:
                                        ni, nj = i + di, j + dj
                                        if 0 <= ni < len(sig_grid) and 0 <= nj < len(bkg_grid):
                                            refine_pts.add((ni, nj))
                                            
            pts_to_eval = [p for p in refine_pts if p not in coarse_pts]
            log_print(f"2D Sparsification: Traced boundary. Running {len(pts_to_eval)} exact refinement points...")
        else:
            # Evaluate full grid
            pts_to_eval = [(i,j) for i in range(len(sig_grid)) for j in range(len(bkg_grid))]
            
        for pt in tqdm(pts_to_eval, desc="2D Edge Refinement", disable=disable_tqdm):
            eval_2d_point(*pt)
            
        # Final Evaluation
        for c in cl:
            results["2d_accepted"][c] = results["2d_t_data"] <= results["2d_t_critical"][c]

    # --- Archiving & Plotting ---
    if output_file is not None:
        save_dict = {
            "test_sig": results.get("test_sig", []),
            "t_data": results.get("t_data", []),
            "profiled_bkg": results.get("profiled_bkg", [])
        }
        if compute_1D_intervals:
            save_dict["1d_test_bkg"] = results["1d_test_bkg"]
            save_dict["1d_t_data_bkg"] = results["1d_t_data_bkg"]
            save_dict["profiled_sig"] = results["profiled_sig"]
        if compute_2D_intervals:
            save_dict["2d_test_sig"] = results["2d_test_sig"]
            save_dict["2d_test_bkg"] = results["2d_test_bkg"]
            save_dict["2d_t_data"] = results["2d_t_data"]
            
        for c in cl:
            if compute_1D_intervals:
                save_dict[f"t_critical_{c}"] = results["t_critical"][c]
                save_dict[f"accepted_{c}"] = results["accepted"][c]
                save_dict[f"1d_t_critical_bkg_{c}"] = results["1d_t_critical_bkg"][c]
                save_dict[f"1d_accepted_bkg_{c}"] = results["1d_accepted_bkg"][c]
            if compute_2D_intervals:
                save_dict[f"2d_t_critical_{c}"] = results["2d_t_critical"][c]
                save_dict[f"2d_accepted_{c}"] = results["2d_accepted"][c]
                
        np.savez(os.path.join(save_directory, output_file), **save_dict)
        log_print(f"Results explicitly saved to {os.path.join(save_directory, output_file)}.npz")

    # Pass config directly to ensure names/modes are respected during plot generation
    if compute_1D_intervals or compute_2D_intervals:
        generate_corner_plot(results, {
            "compute_1D_intervals": compute_1D_intervals,
            "compute_2D_intervals": compute_2D_intervals,
            "param_names": param_names if param_names else ["Signal Scale", "Background Scale"],
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
        "param_names": ["Signal Scale", "Background Scale"]
    }
    with open(filename, 'w') as f:
        json.dump(default_config, f, indent=4)
    print(f"Sample configuration written to {filename}")

def parse_arguments():
    """Parses hierarchy: Defaults -> JSON Config File -> CLI Arguments"""
    parser = argparse.ArgumentParser(description="Feldman-Cousins Confidence Intervals")
    
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
    parser.add_argument('--param_names', type=str, nargs=2, default=argparse.SUPPRESS)
    
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
        "param_names": ["Signal Scale", "Background Scale"]
    }
    
    if hasattr(args, 'config_file'):
        if os.path.exists(args.config_file):
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
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
    
    sig_grid = np.linspace(0.0, 3.0, 30)
    bkg_grid = np.linspace(0.5, 1.5, 20)
    
    if config["likelihood_type"] == "binned":
        print(f"\n--- Running BINNED Analysis Example | Modes -> 1D: {config['compute_1D_intervals']} | 2D: {config['compute_2D_intervals']} ---")
        S_template = np.array([0.1, 0.5, 2.0, 5.0])
        B_template = np.array([15.0, 5.0, 1.0, 0.1])
        
        S_sigma2 = S_template.copy() 
        B_sigma2 = B_template.copy()
        
        np.random.seed(42)
        N_data_binned = np.random.poisson(1.0 * S_template + 1.0 * B_template)
        print(f"Mock Observed Data (Binned Counts): {N_data_binned}")
        
        fc_results = compute_fc_intervals(
            N_data_binned, S_template, B_template, sig_grid, bkg_grid, 
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
        print(f"\n--- Running UNBINNED Analysis Example | Modes -> 1D: {config['compute_1D_intervals']} | 2D: {config['compute_2D_intervals']} ---")
        from scipy.stats import norm, expon
        
        def s_pdf_mock(x): return norm.pdf(x, loc=5.0, scale=1.0)
        def b_pdf_mock(x): return expon.pdf(x, scale=2.0)
        
        s_mc_pool = np.random.normal(loc=5.0, scale=1.0, size=5000)
        b_mc_pool = np.random.exponential(scale=2.0, size=5000)
        
        unbinned_data = np.concatenate([
            np.random.choice(s_mc_pool, size=2),
            np.random.choice(b_mc_pool, size=3)
        ])
        print(f"Mock Observed Unbinned Events: {np.round(unbinned_data, 2)}")
        
        fc_results = compute_fc_intervals(
            unbinned_data, s_pdf_mock, b_pdf_mock, 
            sig_test_points=sig_grid, bkg_grid=bkg_grid, 
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