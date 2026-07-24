"""
Feldman-Cousins Orchestrator Module

This file acts as the central hub and main driver for the PyFC package. 
It coordinates the execution of the Feldman-Cousins confidence interval construction, 
managing the interplay between the data, physical models, likelihood optimizers, 
and Monte Carlo toy generators. It handles checkpointing, parameter space scanning 
(including an optimized 2D grid sparsification algorithm), and data archiving.

Usage (Command Line Interface):
-------------------------------
This script can be executed directly from the terminal to run a demonstration 
of the PyFC pipeline using automatically generated mock data (binned or unbinned).

To run the demonstration using the current configuration (or `fc_config.json`):
    $ python orchestrator.py
    
Alternatively, if the PyFC package is installed in your Python environment:
    $ python -m pyfc.orchestrator
    
You can override configuration parameters directly via command-line arguments. 
For example, to run an unbinned analysis with 500 toys using the SciPy optimizer:
    $ python orchestrator.py --likelihood_type unbinned --n_toys 500 --strategy scipy

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np
import itertools
import logging
import os
import json

# --- Module Imports ---
from .config import parse_arguments
from .plotting import generate_corner_plot
from .toys import generate_and_fit_toys_python
from .optimizers import (
    SCIPY_AVAILABLE, ULTRANEST_AVAILABLE,
    unconditional_fit_scipy, conditional_fit_1d_scipy, conditional_fit_2d_scipy,
    unconditional_fit_ultranest, conditional_fit_1d_ultranest, conditional_fit_2d_ultranest
)

from .binned import (
    unconditional_fit_grid,
    conditional_fit_grid_1d,
    conditional_fit_grid_2d,
    generate_and_fit_toys_grid_1d,
    generate_and_fit_toys_grid_2d,
    NUMBA_AVAILABLE, set_num_threads
)

from .unbinned import (
    unconditional_fit_grid_unbinned,
    conditional_fit_grid_unbinned_1d,
    conditional_fit_grid_unbinned_2d,
    generate_and_fit_toys_grid_unbinned_1d,
    generate_and_fit_toys_grid_unbinned_2d
)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to safely serialize NumPy data types.
    
    Standard `json` libraries cannot handle `np.ndarray` or specific NumPy 
    numerical types (like `np.float64`). This intercepts those objects during 
    the `json.dump` process and converts them to standard Python equivalents.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def _save_fc_archive(results, grids, output_path, cl, compute_1D_intervals, compute_2D_intervals, n_params):
    """
    Serializes the computational state of the Feldman-Cousins environment to a binary `.npz` archive.
    
    This function is critical for the "warm start" (checkpointing) feature. It saves 
    all currently evaluated test statistics, parameter grids, and boolean acceptance maps 
    so that execution can be halted and resumed without losing expensive MC toy computations.

    Parameters:
    -----------
    results : dict
        The central dictionary holding all computed arrays and limits.
    grids : list of np.ndarray
        The parameter scan grids used for the evaluation.
    output_path : str
        The file path for the `.npz` archive.
    cl : list of float
        Confidence levels being evaluated.
    compute_1D_intervals, compute_2D_intervals : bool
        Flags indicating which dimensional modes are active.
    n_params : int
        Total number of parameters in the model.
        
    Returns:
    --------
    None (Writes file to disk)
    """
    save_dict = {
        "best_fit": results["best_fit"],
        "data_uncond_nll": results.get("data_uncond_nll", np.nan)
    }
    
    # Store parameter grids directly to verify states on resume
    for idx, g in enumerate(grids):
        save_dict[f"grid_p{idx+1}"] = g
        
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
                
    np.savez(output_path, **save_dict)


def _save_fc_json(results, output_path, cl, compute_1D_intervals, compute_2D_intervals, n_params):
    """
    Exports computed 1D and 2D intervals to an external, human-readable JSON format.
    
    This provides an alternative to the binary `.npz` archive, facilitating 
    interoperability with web frameworks, external plotting tools, or other languages.

    Parameters:
    -----------
    results : dict
        The populated results dictionary.
    output_path : str
        The destination JSON file path.
    cl : list of float
        The confidence levels evaluated.
    compute_1D_intervals, compute_2D_intervals : bool
        Flags indicating which intervals were processed.
    n_params : int
        Number of parameters in the model.
        
    Returns:
    --------
    None (Writes file to disk)
    """
    json_dict = {
        "best_fit": results["best_fit"],
        "data_uncond_nll": results.get("data_uncond_nll", None)
    }
    
    if compute_1D_intervals:
        json_dict["1d_intervals"] = {}
        for p_idx in range(n_params):
            p_key = f"param{p_idx+1}"
            json_dict["1d_intervals"][p_key] = {
                "test_points": results.get(f"1d_test_p{p_idx+1}"),
                "t_data": results.get(f"1d_t_data_p{p_idx+1}"),
                "prof_params": results.get(f"1d_prof_params_p{p_idx+1}"),
                "thresholds": {}
            }
            for c in cl:
                json_dict["1d_intervals"][p_key]["thresholds"][str(c)] = {
                    "t_critical": results[f"1d_t_critical_p{p_idx+1}"][c],
                    "accepted": results[f"1d_accepted_p{p_idx+1}"][c]
                }

    if compute_2D_intervals and n_params > 1:
        json_dict["2d_intervals"] = {}
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            json_dict["2d_intervals"][pair_name] = {
                f"test_p{fix_A+1}": results.get(f"2d_test_p{fix_A+1}_{pair_name}"),
                f"test_p{fix_B+1}": results.get(f"2d_test_p{fix_B+1}_{pair_name}"),
                "t_data": results.get(f"2d_t_data_{pair_name}"),
                "thresholds": {}
            }
            for c in cl:
                json_dict["2d_intervals"][pair_name]["thresholds"][str(c)] = {
                    "t_critical": results[f"2d_t_critical_{pair_name}"][c],
                    "accepted": results[f"2d_accepted_{pair_name}"][c]
                }

    with open(output_path, 'w') as f:
        json.dump(json_dict, f, cls=NumpyEncoder, indent=4)


def compute_fc_intervals(data, S_model, B_model, grids, 
                         cl=[0.90], n_toys=2000, strategy="scipy", num_cores=None, verbose=1,
                         adaptive_toys=True, toy_batch_size=200, 
                         sparsify_grid=True, warm_start=True,
                         likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                         output_file=None, save_log=False, save_directory="fc_output",
                         use_finite_mc_correction_binned=True, S_sigma2=None, B_sigma2=None,
                         compute_1D_intervals=True, compute_2D_intervals=True, param_names=None,
                         smooth_1d=False, smooth_2d=False):
    """
    Main execution pipeline for the Feldman-Cousins unified approach.
    
    Statistical Theory:
    The Feldman-Cousins method constructs confidence intervals (or regions) that transition 
    smoothly between upper limits and two-sided bounds. It relies on the Profile Likelihood 
    Ratio as the test statistic to determine ordering.
    
    For a given parameter point $\theta_{\text{test}}$, the data test statistic is:
    $$ t_{\text{data}} = -2 \ln \frac{\mathcal{L}(\theta_{\text{test}}, \hat{\hat{\boldsymbol{\nu}}} | \text{data})}{\mathcal{L}(\hat{\boldsymbol{\theta}} | \text{data})} $$
    where $\hat{\boldsymbol{\theta}}$ are the unconditional Maximum Likelihood Estimate (MLE) 
    parameters, and $\hat{\hat{\boldsymbol{\nu}}}$ are the nuisance parameters profiled (maximized) 
    while fixing $\theta = \theta_{\text{test}}$. (Note: Since `calc_nll` returns $-2\ln\mathcal{L}$, 
    this simplifies in code to `cond_nll - data_uncond_nll`).
    
    Because the asymptotic distribution of $t$ (Wilks' theorem) may fail near physical boundaries, 
    we evaluate the exact critical threshold $t_{\text{critical}}$ at a required confidence level $\alpha$ 
    by generating and fitting Monte Carlo pseudo-experiments (toys) generated under the null 
    hypothesis $(\theta_{\text{test}}, \hat{\hat{\boldsymbol{\nu}}})$.
    
    The point $\theta_{\text{test}}$ is included in the final confidence set if:
    $$ t_{\text{data}} \leq t_{\text{critical}}(\alpha) $$

    Parameters:
    -----------
    data : array_like
        The observed data (binned counts or unbinned events).
    S_model, B_model : callable or array_like
        Signal and background structural models/templates.
    grids : list of np.ndarray
        A list of arrays, each defining the evaluation points for a parameter.
    cl : list of float, optional
        Confidence levels to compute (e.g., [0.68, 0.90]).
    n_toys : int, optional
        Number of pseudo-experiments to generate per point.
    strategy : str, optional
        Optimizer strategy: "grid", "scipy", "ultranest", or "hybrid".
    num_cores : int, optional
        Number of parallel threads for toy generation.
    verbose : int, optional
        0 = Silent, 1 = Normal, 2 = Debug.
    adaptive_toys, toy_batch_size, sparsify_grid, warm_start : bool/int
        Algorithmic enhancements to reduce execution time.
    likelihood_type : str
        "binned" or "unbinned".
    S_mc_pool, B_mc_pool : array_like, optional
        Pool of events to bootstrap for unbinned toy generation.
    output_file, save_log, save_directory : str/bool
        I/O file structures.
    use_finite_mc_correction_binned : bool, optional
        Toggle for Poisson-Gamma mixture likelihood.
    S_sigma2, B_sigma2 : array_like, optional
        Template variances for the finite MC correction.
    compute_1D_intervals, compute_2D_intervals : bool
        Switches for calculating 1D profiles or 2D joint contours.
    param_names : list of str, optional
        Names used for plot labels.
    smooth_1d, smooth_2d : bool, optional
        Toggles interpolation smoothing for final plot outputs.

    Returns:
    --------
    results : dict
        A massive dictionary containing all $t_{\text{data}}$, thresholds, accepted masks, 
        and best-fit geometries evaluated across the parameter space.
    """
    
    os.makedirs(save_directory, exist_ok=True)
    n_params = len(grids)
    ckpt_path = os.path.join(save_directory, "checkpoint_fc.npz")
    
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

    # Coerce input configurations
    if isinstance(cl, (float, int)): 
        cl = [float(cl)]
    if ULTRANEST_AVAILABLE and verbose < 2: 
        logging.getLogger("ultranest").setLevel(logging.WARNING)
    if NUMBA_AVAILABLE and num_cores is not None: 
        set_num_threads(num_cores)
        
    # Setup likelihood variances for Finite MC
    if likelihood_type == "binned":
        if S_sigma2 is None: S_sigma2 = np.zeros_like(S_model)
        if B_sigma2 is None: B_sigma2 = np.zeros_like(B_model)
    else:
        S_sigma2, B_sigma2 = None, None

    log_print(f"--- FC Construction Initiated ({n_params}-Parameter Model) ---")
    log_print(f"Modes -> 1D Intervals: {compute_1D_intervals} | 2D Intervals: {compute_2D_intervals}")
    log_print(f"Strategy: {strategy.upper()} | Cores: {num_cores if num_cores else 'Max'} | Likelihood: {likelihood_type.upper()}")
    
    bounds_list = [(g[0], g[-1]) for g in grids]
    disable_tqdm = (verbose == 0) or not TQDM_AVAILABLE
    
    # --- Pre-allocate Architecture with NaNs ---
    # We use np.nan to explicitly identify points in the grid that have not 
    # yet been evaluated (vital for the checkpointing and sparsification logic).
    results = {
        "best_fit": np.full(n_params, np.nan),
        "data_uncond_nll": np.nan
    }

    if compute_1D_intervals:
        for p_idx in range(n_params):
            grid_test = grids[p_idx]
            results[f"1d_test_p{p_idx+1}"] = grid_test
            results[f"1d_t_data_p{p_idx+1}"] = np.full(len(grid_test), np.nan)
            results[f"1d_prof_params_p{p_idx+1}"] = np.full((len(grid_test), n_params), np.nan)
            results[f"1d_t_critical_p{p_idx+1}"] = {c: np.full(len(grid_test), np.nan) for c in cl}
            results[f"1d_accepted_p{p_idx+1}"] = {c: np.full(len(grid_test), False) for c in cl}

    if compute_2D_intervals and n_params > 1:
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            gridA, gridB = grids[fix_A], grids[fix_B]
            results[f"2d_test_p{fix_A+1}_{pair_name}"] = gridA
            results[f"2d_test_p{fix_B+1}_{pair_name}"] = gridB
            results[f"2d_t_data_{pair_name}"] = np.full((len(gridA), len(gridB)), np.nan)
            results[f"2d_t_critical_{pair_name}"] = {c: np.full((len(gridA), len(gridB)), np.nan) for c in cl}
            results[f"2d_accepted_{pair_name}"] = {c: np.full((len(gridA), len(gridB)), False) for c in cl}

    # --- Resume Protocol Validation ---
    # Attempts to ingest an existing state dictionary from a previous incomplete run.
    # Checks strictly if the requested bounds/grids map precisely to the saved geometries.
    if warm_start and os.path.exists(ckpt_path):
        try:
            ckpt = np.load(ckpt_path, allow_pickle=True)
            loaded_grids = [ckpt[f"grid_p{i+1}"] for i in range(n_params) if f"grid_p{i+1}" in ckpt]
            
            grids_match = False
            if len(loaded_grids) == n_params:
                grids_match = all(np.array_equal(g1, g2) for g1, g2 in zip(loaded_grids, grids))
                
            if grids_match:
                log_print(f"Grids verified identically. Resuming checkpoint from {ckpt_path}.")
                if "best_fit" in ckpt: results["best_fit"] = ckpt["best_fit"]
                if "data_uncond_nll" in ckpt: results["data_uncond_nll"] = float(ckpt["data_uncond_nll"])
                
                # Restore 1D tracking arrays
                if compute_1D_intervals:
                    for p_idx in range(n_params):
                        if f"1d_t_data_p{p_idx+1}" in ckpt: results[f"1d_t_data_p{p_idx+1}"] = ckpt[f"1d_t_data_p{p_idx+1}"]
                        if f"1d_prof_params_p{p_idx+1}" in ckpt: results[f"1d_prof_params_p{p_idx+1}"] = ckpt[f"1d_prof_params_p{p_idx+1}"]
                        for c in cl:
                            if f"1d_t_critical_p{p_idx+1}_{c}" in ckpt: results[f"1d_t_critical_p{p_idx+1}"][c] = ckpt[f"1d_t_critical_p{p_idx+1}_{c}"]
                            if f"1d_accepted_p{p_idx+1}_{c}" in ckpt: results[f"1d_accepted_p{p_idx+1}"][c] = ckpt[f"1d_accepted_p{p_idx+1}_{c}"]
                            
                # Restore 2D tracking arrays
                if compute_2D_intervals and n_params > 1:
                    pairs = list(itertools.combinations(range(n_params), 2))
                    for fix_A, fix_B in pairs:
                        pair_name = f"p{fix_A+1}p{fix_B+1}"
                        if f"2d_t_data_{pair_name}" in ckpt: results[f"2d_t_data_{pair_name}"] = ckpt[f"2d_t_data_{pair_name}"]
                        for c in cl:
                            if f"2d_t_critical_{pair_name}_{c}" in ckpt: results[f"2d_t_critical_{pair_name}"][c] = ckpt[f"2d_t_critical_{pair_name}_{c}"]
                            if f"2d_accepted_{pair_name}_{c}" in ckpt: results[f"2d_accepted_{pair_name}"][c] = ckpt[f"2d_accepted_{pair_name}_{c}"]
            else:
                log_print("Parameter grids strictly conflict with checkpoint environment. Initiating fresh start.")
        except Exception as e:
            log_print(f"Failed to process checkpoint geometry: {e}. Initiating fresh start.")

    # --- PHASE 0: Fit global unconditional data ONCE ---
    # Finds the absolute minimum NLL over the full parameter space. 
    # This forms the denominator of the PLR test statistic for all points.
    full_grid_points = np.array(list(itertools.product(*grids)), dtype=np.float64)
    
    if np.isnan(results["data_uncond_nll"]):
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
        results["data_uncond_nll"] = data_uncond_nll
        _save_fc_archive(results, grids, ckpt_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
    else:
        log_print("Bypassing global unconditional fit execution (Loaded configuration found).")
        data_uncond_nll = results["data_uncond_nll"]

    # --- PHASE 1: Compute 1D Intervals ---
    if compute_1D_intervals:
        log_print("Executing 1D Parameter Scans...")
        for p_idx in range(n_params):
            grid_test = grids[p_idx]
            free_grids = [grids[i] for i in range(n_params) if i != p_idx]
            
            # Sub-grid for profiling nuisance parameters exhaustively
            if not free_grids:
                cond_grid_points = np.zeros((1, 0), dtype=np.float64)
            else:
                cond_grid_points = np.array(list(itertools.product(*free_grids)), dtype=np.float64)
                
            t_data_arr = results[f"1d_t_data_p{p_idx+1}"]
            prof_params_arr = results[f"1d_prof_params_p{p_idx+1}"]
            t_crit_dict = results[f"1d_t_critical_p{p_idx+1}"]
            
            # Sub-phase 1a: Calculate t_data across the 1D grid
            for i, pt in enumerate(tqdm(grid_test, desc=f"1D Data (p{p_idx+1})", disable=disable_tqdm)):
                if not np.isnan(t_data_arr[i]): 
                    continue
                    
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
                # Evaluate the actual PLR data statistic (bounded at 0 to fix numerical floating point noise)
                t_data_arr[i] = max(0.0, cond_nll - data_uncond_nll)

            # Sub-phase 1b: Calculate t_critical via Toy Generation
            for i, pt in enumerate(tqdm(grid_test, desc=f"1D Toys (p{p_idx+1})", disable=disable_tqdm)):
                if not np.isnan(t_crit_dict[cl[0]][i]): 
                    continue
                    
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
                    
            results[f"1d_accepted_p{p_idx+1}"] = {c: t_data_arr <= t_crit_dict[c] for c in cl}
            
            # Record loop checkpoint 
            _save_fc_archive(results, grids, ckpt_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
            log_print(f"Checkpoint successfully written mapping 1D parameter {p_idx+1}")

    # --- PHASE 2: Compute 2D Intervals ---
    if compute_2D_intervals and n_params > 1:
        pairs = list(itertools.combinations(range(n_params), 2))
        for fix_A, fix_B in pairs:
            gridA, gridB = grids[fix_A], grids[fix_B]
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            log_print(f"Executing 2D Grid Scan mapping {pair_name} ({len(gridA)}x{len(gridB)})...")
            
            free_grids = [grids[i] for i in range(n_params) if i != fix_A and i != fix_B]
            if not free_grids:
                cond_grid_points = np.zeros((1, 0), dtype=np.float64)
            else:
                cond_grid_points = np.array(list(itertools.product(*free_grids)), dtype=np.float64)
            
            # Helper function to evaluate toys for a specific (i, j) 2D coordinate 
            def eval_2d_point(i, j):
                if not np.isnan(results[f"2d_t_critical_{pair_name}"][cl[0]][i, j]):
                    return
                    
                p_A, p_B = gridA[i], gridB[j]
                
                # Step 1. Exact Data NLL Calculation (Profiling out remaining nuisance pars)
                if np.isnan(results[f"2d_t_data_{pair_name}"][i, j]):
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
                else:
                    # Rerun extremely rapid exact data fitting to retrieve the localized profiling if bypassing saved data.
                    if strategy == "scipy":
                        _, true_params = conditional_fit_2d_scipy(p_A, p_B, fix_A, fix_B, n_params, data, S_model, B_model, bounds_list, likelihood_type=likelihood_type, S_sigma2=S_sigma2, B_sigma2=B_sigma2, use_finite_mc=use_finite_mc_correction_binned)
                    else:
                        _, true_params = conditional_fit_grid_1d(p_A, fix_A, n_params, data, S_model, B_model, cond_grid_points, S_sigma2, B_sigma2, use_finite_mc_correction_binned)
                
                # Step 2. Sequential Toy Assessment to get critical threshold for coverage
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

            # --- Sparsification Array Tracing ---
            # To dramatically reduce computation time in 2D grids, this algorithm:
            # 1. Evaluates a coarse subset of grid nodes.
            # 2. Uses Bivariate Spline interpolation to estimate the t_critical surface.
            # 3. Dynamically traces and evaluates only the specific high-resolution nodes 
            #    where the condition (t_data <= t_critical) transitions state (the contour boundary).
            if sparsify_grid:
                step_A = max(1, len(gridA) // 5)
                step_B = max(1, len(gridB) // 5)
                coarse_i = sorted(list(set(list(range(0, len(gridA), step_A)) + [len(gridA)-1])))
                coarse_j = sorted(list(set(list(range(0, len(gridB), step_B)) + [len(gridB)-1])))
                coarse_pts = [(i, j) for i in coarse_i for j in coarse_j]
                
                log_print(f"2D Sparsification: Coarse array pass across {len(coarse_pts)} structural points...")
                for pt in tqdm(coarse_pts, desc=f"2D Coarse {pair_name}", disable=disable_tqdm): 
                    eval_2d_point(*pt)
                    
                # Interpolate the coarse critical surface framework
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
                
                # Dynamic Perimeter Edge Tracing
                # Scans the interpolated map and flags cells that lie on the inclusion/exclusion border.
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
                log_print(f"2D Sparsification: Boundary resolved. Integrating {len(pts_to_eval)} exact refinement matrices...")
            else:
                pts_to_eval = [(i,j) for i in range(len(gridA)) for j in range(len(gridB))]
                
            for pt in tqdm(pts_to_eval, desc=f"2D Edge {pair_name}", disable=disable_tqdm): 
                eval_2d_point(*pt)
            
            # Formally calculate final boolean acceptance mask
            for c in cl: 
                results[f"2d_accepted_{pair_name}"][c] = results[f"2d_t_data_{pair_name}"] <= results[f"2d_t_critical_{pair_name}"][c]
                
            # Record loop checkpoint 
            _save_fc_archive(results, grids, ckpt_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
            log_print(f"Checkpoint successfully written mapping 2D sector {pair_name}")

    # --- Structural Archiving & Cleanup Operations ---
    if output_file is not None:
        final_path = os.path.join(save_directory, f"{output_file}.npz")
        final_path_json = os.path.join(save_directory, f"{output_file}.json")
        _save_fc_archive(results, grids, final_path, cl, compute_1D_intervals, compute_2D_intervals, n_params)
        _save_fc_json(results, final_path_json, cl, compute_1D_intervals, compute_2D_intervals, n_params)
        log_print(f"Results archived completely to storage matrix {final_path} and JSON {final_path_json}")
        
        # Eliminate interim tracking files now that final save is secure
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    if compute_1D_intervals or (compute_2D_intervals and n_params > 1):
        generate_corner_plot(results, {
            "n_params": n_params,
            "compute_1D_intervals": compute_1D_intervals,
            "compute_2D_intervals": compute_2D_intervals,
            "param_names": param_names if param_names else [f"param{i+1}" for i in range(n_params)],
            "cl": cl,
            "save_directory": save_directory,
            "smooth_1d": smooth_1d,
            "smooth_2d": smooth_2d
        })

    return results

if __name__ == "__main__":
    config = parse_arguments()
    
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
            param_names=config["param_names"],
            smooth_1d=config["smooth_1d"],
            smooth_2d=config["smooth_2d"]
        )

    elif config["likelihood_type"] == "unbinned":
        print(f"\n--- Running UNBINNED Analysis Example ({len(grids)}-Parameter) | Modes -> 1D: {config['compute_1D_intervals']} | 2D: {config['compute_2D_intervals']} ---")
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
            grids, 
            cl=config["cl"], n_toys=config["n_toys"], strategy=config["strategy"], 
            num_cores=config["num_cores"], verbose=config["verbose"],
            adaptive_toys=config["adaptive_toys"], toy_batch_size=config["toy_batch_size"],
            sparsify_grid=config["sparsify_grid"], warm_start=config["warm_start"],
            likelihood_type="unbinned", S_mc_pool=s_mc_pool, B_mc_pool=b_mc_pool,
            output_file=config["output_file"], save_log=config["save_log"], save_directory=config["save_directory"],
            compute_1D_intervals=config["compute_1D_intervals"],
            compute_2D_intervals=config["compute_2D_intervals"],
            param_names=config["param_names"],
            smooth_1d=config["smooth_1d"],
            smooth_2d=config["smooth_2d"]
        )