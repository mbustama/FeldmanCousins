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
    generate_and_fit_toys_grid,
    NUMBA_AVAILABLE,
    set_num_threads
)
from unbinned import (
    calc_nll_unbinned, 
    generate_unbinned_toy, 
    unconditional_fit_grid_unbinned, 
    conditional_fit_grid_unbinned, 
    generate_and_fit_toys_grid_unbinned
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


# --- 1. Scipy Continuous Optimizers (Branched for Binned/Unbinned) ---
def unconditional_fit_scipy(data, S_model, B_model, sig_bounds, bkg_bounds, seed=None, 
                            likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def cost(params):
        if likelihood_type == "binned":
            return calc_nll(params[0], params[1], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
        else:
            return calc_nll_unbinned(params[0], params[1], data, S_model, B_model)
    
    x0 = seed if seed is not None else [(sig_bounds[0]+sig_bounds[1])/2.0, (bkg_bounds[0]+bkg_bounds[1])/2.0]
    res = optimize.minimize(cost, x0=x0, bounds=[sig_bounds, bkg_bounds], method='L-BFGS-B')
    return res.fun, res.x[0], res.x[1]

def conditional_fit_scipy(test_sig, data, S_model, B_model, bkg_bounds, seed=None, 
                          likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def cost(bkg):
        if likelihood_type == "binned":
            return calc_nll(test_sig, bkg[0], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
        else:
            return calc_nll_unbinned(test_sig, bkg[0], data, S_model, B_model)
        
    x0 = [seed] if seed is not None else [(bkg_bounds[0]+bkg_bounds[1])/2.0]
    res = optimize.minimize(cost, x0=x0, bounds=[bkg_bounds], method='L-BFGS-B')
    return res.fun, res.x[0]


# --- 2. UltraNest Optimizers (Branched for Binned/Unbinned) ---
def unconditional_fit_ultranest(data, S_model, B_model, sig_bounds, bkg_bounds, verbose=1, 
                                likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def prior_transform(cube):
        sig = cube[0] * (sig_bounds[1] - sig_bounds[0]) + sig_bounds[0]
        bkg = cube[1] * (bkg_bounds[1] - bkg_bounds[0]) + bkg_bounds[0]
        return np.array([sig, bkg])

    def log_likelihood(params):
        if likelihood_type == "binned":
            return -calc_nll(params[0], params[1], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
        else:
            return -calc_nll_unbinned(params[0], params[1], data, S_model, B_model)

    sampler = ultranest.ReactiveNestedSampler(['sig_scale', 'bkg_scale'], log_likelihood, prior_transform, log_dir=None)
    
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2)}
    if verbose < 2:
        logging.getLogger("ultranest").setLevel(logging.WARNING)
        run_kwargs['viz_callback'] = False
        
    result = sampler.run(**run_kwargs)
    best_sig, best_bkg = result['maximum_likelihood']['point']
    min_nll = -result['maximum_likelihood']['logl']
    return min_nll, best_sig, best_bkg

def conditional_fit_ultranest(test_sig, data, S_model, B_model, bkg_bounds, verbose=1, 
                              likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def prior_transform(cube):
        bkg = cube[0] * (bkg_bounds[1] - bkg_bounds[0]) + bkg_bounds[0]
        return np.array([bkg])

    def log_likelihood(params):
        if likelihood_type == "binned":
            return -calc_nll(test_sig, params[0], data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
        else:
            return -calc_nll_unbinned(test_sig, params[0], data, S_model, B_model)

    sampler = ultranest.ReactiveNestedSampler(['bkg_scale'], log_likelihood, prior_transform, log_dir=None)
    
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2)}
    if verbose < 2:
        logging.getLogger("ultranest").setLevel(logging.WARNING)
        run_kwargs['viz_callback'] = False
        
    result = sampler.run(**run_kwargs)
    best_bkg = result['maximum_likelihood']['point'][0]
    min_nll = -result['maximum_likelihood']['logl']
    return min_nll, best_bkg


# --- 3. Python Vectorized Toy Generator ---
def generate_and_fit_toys_python(test_sig, profiled_bkg, S_model, B_model, 
                                 sig_bounds, bkg_bounds, n_toys, strategy, num_cores=None, verbose=1,
                                 likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                                 S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    """Handles threaded generation & fitting for both Binned and Unbinned schemas."""
    toys_binned_data = None
    if likelihood_type == "binned":
        mu_true = test_sig * S_model + profiled_bkg * B_model
        toys_binned_data = np.random.poisson(mu_true, size=(n_toys, len(S_model)))
    
    def fit_single_toy(t):
        if likelihood_type == "binned":
            toy_data = toys_binned_data[t]
        else:
            toy_data = generate_unbinned_toy(test_sig, profiled_bkg, S_mc_pool, B_mc_pool)
        
        if strategy == "scipy" or strategy == "hybrid":
            seed_u = [test_sig, profiled_bkg]
            seed_c = profiled_bkg
            uncond_nll, _, _ = unconditional_fit_scipy(toy_data, S_model, B_model, sig_bounds, bkg_bounds, seed=seed_u, 
                                                       likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            cond_nll, _ = conditional_fit_scipy(test_sig, toy_data, S_model, B_model, bkg_bounds, seed=seed_c, 
                                                likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
        elif strategy == "ultranest":
            uncond_nll, _, _ = unconditional_fit_ultranest(toy_data, S_model, B_model, sig_bounds, bkg_bounds, verbose=0, 
                                                           likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            cond_nll, _ = conditional_fit_ultranest(test_sig, toy_data, S_model, B_model, bkg_bounds, verbose=0, 
                                                    likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc)
            
        return max(0.0, cond_nll - uncond_nll)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        t_stats = list(executor.map(fit_single_toy, range(n_toys)))
        
    return np.array(t_stats)


# --- 4. Main Feldman-Cousins Wrapper ---
def compute_fc_intervals(data, S_model, B_model, 
                         sig_test_points, bkg_grid, 
                         cl=[0.90], n_toys=2000, strategy="scipy", num_cores=None, verbose=1,
                         adaptive_toys=True, toy_batch_size=200, 
                         sparsify_grid=True, warm_start=True,
                         likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                         output_file=None, save_log=False,
                         use_finite_mc_correction_binned=True, S_sigma2=None, B_sigma2=None):
    """
    Orchestrates the Feldman-Cousins construction.
    Supports Likelihoods: 'binned' or 'unbinned'.
    """
    # Setup Logging Architecture
    run_logger = logging.getLogger("FC_Orchestrator")
    run_logger.setLevel(logging.INFO)
    if save_log and not run_logger.handlers:
        fh = logging.FileHandler('fc_run.log')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        run_logger.addHandler(fh)
        
    def log_print(msg):
        if verbose > 0: print(msg)
        if save_log: run_logger.info(msg)

    if isinstance(cl, (float, int)):
        cl = [float(cl)]

    valid_strategies = ["grid", "scipy", "ultranest", "hybrid"]
    valid_likelihoods = ["binned", "unbinned"]
    
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {valid_strategies}")
    if likelihood_type not in valid_likelihoods:
        raise ValueError(f"likelihood_type must be one of {valid_likelihoods}")
        
    if likelihood_type == "unbinned" and strategy == "grid":
        log_print("Notice: Unbinned grid strategy bypasses Numba compiler to allow Python callables.")
        
    if strategy in ["ultranest", "hybrid"] and not ULTRANEST_AVAILABLE:
        log_print("Warning: ultranest not installed. Falling back to scipy.")
        strategy = "scipy"
    if strategy in ["scipy", "hybrid"] and not SCIPY_AVAILABLE:
        log_print("Warning: scipy not installed. Falling back to grid.")
        strategy = "grid"

    if ULTRANEST_AVAILABLE and verbose < 2:
        logging.getLogger("ultranest").setLevel(logging.WARNING)

    if NUMBA_AVAILABLE and num_cores is not None:
        set_num_threads(num_cores)
        
    # Safeguard binned variance matrices if not provided
    if likelihood_type == "binned":
        if S_sigma2 is None: S_sigma2 = np.zeros_like(S_model)
        if B_sigma2 is None: B_sigma2 = np.zeros_like(B_model)
    else:
        S_sigma2, B_sigma2 = None, None

    log_print(f"Starting FC Construction: {len(sig_test_points)} test points, up to {n_toys} toys/point.")
    log_print(f"Confidence Levels: {cl}")
    log_print(f"Schema: {likelihood_type.upper()} | Strategy: {strategy.upper()} (Cores: {num_cores if num_cores else 'Max'})")
    log_print(f"Optimizations -> Warm Start: {warm_start}, Adaptive Toys: {adaptive_toys}, Grid Sparsification: {sparsify_grid}")
    if likelihood_type == "binned":
        log_print(f"Finite MC Correction: {use_finite_mc_correction_binned}")
    
    results = {
        "test_sig": sig_test_points,
        "t_data": np.zeros(len(sig_test_points)),
        "profiled_bkg": np.zeros(len(sig_test_points)),
        "t_critical": {c: np.zeros(len(sig_test_points)) for c in cl},
        "accepted": {c: np.zeros(len(sig_test_points), dtype=bool) for c in cl}
    }
    
    sig_grid = sig_test_points 
    sig_bounds = (sig_grid[0], sig_grid[-1])
    bkg_bounds = (bkg_grid[0], bkg_grid[-1])
    disable_tqdm = (verbose == 0) or not TQDM_AVAILABLE
    
    # --- Fit unconditional data ONCE ---
    if strategy == "grid":
        if likelihood_type == "binned":
            data_uncond_nll, _, _ = unconditional_fit_grid(data, S_model, B_model, sig_grid, bkg_grid, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
        else:
            data_uncond_nll, _, _ = unconditional_fit_grid_unbinned(data, S_model, B_model, sig_grid, bkg_grid)
    elif strategy in ["ultranest", "hybrid"]:
        data_uncond_nll, _, _ = unconditional_fit_ultranest(data, S_model, B_model, sig_bounds, bkg_bounds, verbose, likelihood_type, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
    elif strategy == "scipy":
        data_uncond_nll, _, _ = unconditional_fit_scipy(data, S_model, B_model, sig_bounds, bkg_bounds, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
    
    # --- PHASE 1: Fit Data across Grid ---
    last_bkg_seed = None
    iterator_data = tqdm(range(len(sig_test_points)), desc="Fitting Data", disable=disable_tqdm)
    for i in iterator_data:
        test_sig = sig_test_points[i]
        
        if strategy == "grid":
            if likelihood_type == "binned":
                data_cond_nll, bkg_data = conditional_fit_grid(test_sig, data, S_model, B_model, bkg_grid, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
            else:
                data_cond_nll, bkg_data = conditional_fit_grid_unbinned(test_sig, data, S_model, B_model, bkg_grid)
        elif strategy in ["ultranest", "hybrid"]:
            data_cond_nll, bkg_data = conditional_fit_ultranest(test_sig, data, S_model, B_model, bkg_bounds, verbose, likelihood_type, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
        elif strategy == "scipy":
            seed_val = last_bkg_seed if warm_start and last_bkg_seed is not None else None
            data_cond_nll, bkg_data = conditional_fit_scipy(test_sig, data, S_model, B_model, bkg_bounds, seed=seed_val, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
            
        results["profiled_bkg"][i] = bkg_data
        results["t_data"][i] = max(0.0, data_cond_nll - data_uncond_nll)
        
        if warm_start:
            last_bkg_seed = bkg_data

    # Helper function for Phase 2
    def run_toys_for_idx(idx, use_adaptive):
        test_sig = sig_test_points[idx]
        bkg_data = results["profiled_bkg"][idx]
        data_t = results["t_data"][idx]
        
        if use_adaptive:
            t_stats = []
            toys_done = 0
            
            while toys_done < n_toys:
                batch = min(toy_batch_size, n_toys - toys_done)
                if strategy == "grid":
                    if likelihood_type == "binned":
                        batch_stats = generate_and_fit_toys_grid(test_sig, bkg_data, S_model, B_model, sig_grid, bkg_grid, batch, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                    else:
                        batch_stats = generate_and_fit_toys_grid_unbinned(test_sig, bkg_data, S_model, B_model, sig_grid, bkg_grid, batch, S_mc_pool, B_mc_pool)
                else:
                    batch_stats = generate_and_fit_toys_python(test_sig, bkg_data, S_model, B_model, sig_bounds, bkg_bounds, batch, strategy, num_cores, verbose=0, likelihood_type=likelihood_type, S_mc_pool=S_mc_pool, B_mc_pool=B_mc_pool, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
                
                t_stats.extend(batch_stats)
                toys_done += batch
                
                k = sum(1 for t in t_stats if t >= data_t)
                toys_rem = n_toys - toys_done
                
                all_resolved = True
                for c in cl:
                    req = n_toys - min(int(c * n_toys), n_toys - 1)
                    if not (k >= req or (k + toys_rem) < req):
                        all_resolved = False
                        break
                
                if all_resolved:
                    for c in cl:
                        req = n_toys - min(int(c * n_toys), n_toys - 1)
                        if k >= req:
                            results["t_critical"][c][idx] = data_t + 1e-3
                        else:
                            results["t_critical"][c][idx] = data_t - 1e-3
                    return
            
            t_stats.sort()
            for c in cl:
                results["t_critical"][c][idx] = t_stats[min(int(c * n_toys), n_toys - 1)]
            
        else:
            if strategy == "grid":
                if likelihood_type == "binned":
                    toy_t_stats = generate_and_fit_toys_grid(test_sig, bkg_data, S_model, B_model, sig_grid, bkg_grid, n_toys, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                else:
                    toy_t_stats = generate_and_fit_toys_grid_unbinned(test_sig, bkg_data, S_model, B_model, sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool)
            else:
                toy_t_stats = generate_and_fit_toys_python(test_sig, bkg_data, S_model, B_model, sig_bounds, bkg_bounds, n_toys, strategy, num_cores, verbose=0, likelihood_type=likelihood_type, S_mc_pool=S_mc_pool, B_mc_pool=B_mc_pool, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
            toy_t_stats.sort()
            for c in cl:
                results["t_critical"][c][idx] = toy_t_stats[min(int(c * n_toys), n_toys - 1)]

    # --- PHASE 2: Generate & Fit Toys ---
    indices_to_evaluate = []
    
    if sparsify_grid:
        coarse_step = max(1, len(sig_test_points) // 10)
        coarse_indices = list(range(0, len(sig_test_points), coarse_step))
        if coarse_indices[-1] != len(sig_test_points) - 1:
            coarse_indices.append(len(sig_test_points) - 1)
            
        log_print(f"Sparsification: Running coarse grid ({len(coarse_indices)} points)...")
        for idx in tqdm(coarse_indices, desc="Coarse Toys", disable=disable_tqdm):
            run_toys_for_idx(idx, use_adaptive=False) 
            
        evaluated_x = sig_test_points[coarse_indices]
        refinement_indices = set()
        
        for c in cl:
            evaluated_tc = results["t_critical"][c][coarse_indices]
            results["t_critical"][c][:] = np.interp(sig_test_points, evaluated_x, evaluated_tc)
            approx_accepted = results["t_data"] <= results["t_critical"][c]
            
            for i in range(len(approx_accepted) - 1):
                if approx_accepted[i] != approx_accepted[i+1]:
                    start_pad = max(0, i - coarse_step)
                    end_pad = min(len(sig_test_points), i + coarse_step + 1)
                    for j in range(start_pad, end_pad):
                        if j not in coarse_indices:
                            refinement_indices.add(j)
                        
        indices_to_evaluate = sorted(list(refinement_indices))
        log_print(f"Sparsification: Found boundaries. Running {len(indices_to_evaluate)} refinement points...")
    else:
        indices_to_evaluate = list(range(len(sig_test_points)))

    if len(indices_to_evaluate) > 0:
        iterator_toys = tqdm(indices_to_evaluate, desc="Refinement Toys", disable=disable_tqdm)
        for idx in iterator_toys:
            run_toys_for_idx(idx, use_adaptive=adaptive_toys)
            
    # --- Final Acceptance Evaluation & Saving ---
    for c in cl:
        results["accepted"][c] = results["t_data"] <= results["t_critical"][c]
        
    if output_file is not None:
        save_dict = {
            "test_sig": results["test_sig"],
            "t_data": results["t_data"],
            "profiled_bkg": results["profiled_bkg"]
        }
        for c in cl:
            save_dict[f"t_critical_{c}"] = results["t_critical"][c]
            save_dict[f"accepted_{c}"] = results["accepted"][c]
            
        np.savez(output_file, **save_dict)
        log_print(f"Results explicitly saved to {output_file}.npz")
        
    return results


# --- 5. Config & CLI Logic ---
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
        "output_file": None,
        "save_log": False,
        "use_finite_mc_correction_binned": True
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
    parser.add_argument('--use_finite_mc_correction_binned', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    
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
        "use_finite_mc_correction_binned": True
    }
    
    # 1. Update with values from JSON Config File (if provided)
    if hasattr(args, 'config_file'):
        if os.path.exists(args.config_file):
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
        else:
            print(f"Warning: Config file {args.config_file} not found. Proceeding with defaults.")

    # 2. Update explicitly provided CLI Arguments (Overrides both Config & Defaults)
    cli_args = vars(args)
    for key, value in cli_args.items():
        if key not in ['config_file', 'generate_config']:
            config[key] = value
            
    return config


# --- 6. Execution Examples ---
if __name__ == "__main__":
    
    config = parse_arguments()
    
    sig_grid = np.linspace(0.0, 3.0, 30)
    bkg_grid = np.linspace(0.5, 1.5, 20)
    
    if config["likelihood_type"] == "binned":
        print("\n--- Running BINNED Analysis Example ---")
        S_template = np.array([0.1, 0.5, 2.0, 5.0])
        B_template = np.array([15.0, 5.0, 1.0, 0.1])
        
        # Simulating variance matrices (e.g. unweighted MC where sigma^2 == mu)
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
            output_file=config["output_file"], save_log=config["save_log"],
            use_finite_mc_correction_binned=config["use_finite_mc_correction_binned"],
            S_sigma2=S_sigma2, B_sigma2=B_sigma2
        )
        
        for c in config["cl"]:
            accepted_sigs = fc_results["test_sig"][fc_results["accepted"][c]]
            if len(accepted_sigs) > 0:
                print(f"Result (Binned) [{c} C.L.]: [{accepted_sigs[0]:.2f}, {accepted_sigs[-1]:.2f}]")
            else:
                print(f"Result (Binned) [{c} C.L.]: No parameters accepted.")

    elif config["likelihood_type"] == "unbinned":
        print("\n--- Running UNBINNED Analysis Example ---")
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
            output_file=config["output_file"], save_log=config["save_log"]
        )
        
        for c in config["cl"]:
            accepted_sigs = fc_results["test_sig"][fc_results["accepted"][c]]
            if len(accepted_sigs) > 0:
                print(f"Result (Unbinned) [{c} C.L.]: [{accepted_sigs[0]:.2f}, {accepted_sigs[-1]:.2f}]")
            else:
                print(f"Result (Unbinned) [{c} C.L.]: No parameters accepted.")