"""
Configuration and Argument Parsing Module

This file defines the configuration management system for the PyFC framework. 
It establishes a hierarchical parameter loading mechanism (Hardcoded Defaults -> 
JSON Configuration File -> Command Line Arguments) to control the execution 
of the Feldman-Cousins confidence interval construction.

While this file does not execute the mathematical routines, it defines the 
statistical, algorithmic, and computational parameters that govern the 
underlying profile likelihood ratio tests and Monte Carlo toy generation.

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import json
import argparse
import sys
import os

def generate_sample_config(filename="../config/example_fc_config.json"):
    """
    Generates a sample JSON configuration file containing default execution parameters.
    
    Statistical & Algorithmic Context:
    The default configuration initializes a standard binned likelihood analysis 
    calculating confidence intervals at 68% and 90% Confidence Levels (CL). 
    It enables the finite Monte Carlo correction (assuming template statistical 
    uncertainties require a Poisson-Gamma mixture likelihood) and utilizes an 
    adaptive Monte Carlo toy generation strategy to optimize computational 
    resources when empirical p-values are far from the critical threshold alpha.

    Parameters:
    -----------
    filename : str, optional
        The destination file path where the JSON configuration will be written. 
        Defaults to "../config/example_fc_config.json".

    Returns:
    --------
    None
        The function writes a file to disk and prints a confirmation message.
    """
    default_config = {
        "likelihood_type": "binned",
        "cl": [0.68, 0.90],
        "n_toys": 200,
        "strategy": "scipy",
        "num_cores": 8,
        "verbose": 1,
        "adaptive_toys": True,
        "toy_batch_size": 200,
        "sparsify_grid": True,
        "warm_start": True,
        "output_file": "fc_results",
        "save_log": True,
        "save_directory": "../output/example_fc_output",
        "use_finite_mc_correction_binned": True,
        "compute_1D_intervals": True,
        "compute_2D_intervals": True,
        "param_names": ["param1", "param2", "param3"],
        "smooth_1d": False,
        "smooth_2d": False
    }
    with open(filename, 'w') as f:
        json.dump(default_config, f, indent=4)
    print(f"Sample configuration written to {filename}")

def parse_arguments():
    """
    Parses execution parameters using a strict hierarchy: 
    Defaults -> JSON Config File -> CLI Arguments.
    
    Statistical Theory & Configuration Parameters:
    - likelihood_type: Defines the probability density function (PDF) form. 
      'binned' uses Poisson/Negative Binomial counting statistics; 'unbinned' 
      uses continuous event likelihoods.
    - cl: The target Confidence Level(s) (1 - alpha), defining the required 
      frequentist coverage probability (e.g., 0.90 for 90% coverage).
    - n_toys: The baseline number of parametric bootstrap pseudo-experiments 
      used to build the empirical Profile Likelihood Ratio (PLR) distribution.
    - use_finite_mc_correction_binned: If True, convolves the standard Poisson 
      likelihood with a Gamma prior to account for limited statistics in MC templates.
    - strategy: The optimization algorithm used to minimize the Negative 
      Log-Likelihood (NLL). Options include exhaustive 'grid', gradient-based 
      'scipy', or nested sampling 'ultranest'.

    Parameters:
    -----------
    None (Reads directly from command line via sys.argv)

    Returns:
    --------
    config : dict
        A dictionary containing the fully resolved configuration parameters 
        necessary to execute the Feldman-Cousins orchestrator.
    """
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
    parser.add_argument('--smooth_1d', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    parser.add_argument('--smooth_2d', type=lambda x: str(x).lower() in ['true', '1', 'yes'], default=argparse.SUPPRESS)
    
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
        "param_names": ["param1", "param2", "param3"],
        "smooth_1d": False,
        "smooth_2d": False
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