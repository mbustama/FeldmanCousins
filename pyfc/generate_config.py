"""
Configuration Generator Module (Interactive CLI)

This file contains a standalone interactive command-line interface (CLI) 
designed to generate a properly formatted JSON configuration file (`fc_config.json`) 
for the PyFC (Feldman-Cousins) orchestrator. It handles user input collection, 
type casting, rule validation, and default parameter fallback.

While this script does not perform mathematical calculations directly, it defines 
the statistical hyper-parameters (such as Confidence Levels, likelihood formulations, 
and Monte Carlo toy counts) that dictate the rigorousness and underlying assumptions 
of the frequentist confidence interval construction.

Usage (Command Line Interface):
-------------------------------
To launch the interactive configuration generator, run the script directly from 
your terminal:
    $ python generate_config.py
    
Alternatively, if the PyFC package is installed in your Python environment:
    $ python -m pyfc.generate_config
    
Interactive Instructions:
- You will be prompted sequentially for various configuration values.
- Type your desired value and press [Enter].
- To accept the default value displayed in brackets (e.g., [Default: binned]), 
  simply press [Enter] without typing anything.
- For boolean (yes/no) questions, accepted inputs include 'y', 'yes', 't', 'true', '1' 
  for True, and 'n', 'no', 'f', 'false', '0' for False (case-insensitive).
- For list inputs (like Confidence Levels), provide values separated by commas 
  (e.g., 0.68, 0.90, 0.95).

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import json
import os

def parse_bool(value):
    """
    Safely converts string inputs to boolean values.
    
    Statistical Context:
    Boolean flags in the configuration control critical methodological choices, 
    such as whether to apply the finite Monte Carlo correction (which shifts the 
    likelihood from a standard Poisson distribution to a Negative Binomial 
    Poisson-Gamma mixture) or whether to use adaptive sampling for the toys.

    Parameters:
    -----------
    value : str or bool
        The raw input provided by the user (e.g., 'y', 'n', 'true', '1').

    Returns:
    --------
    bool
        The parsed boolean True or False.
        
    Raises:
    -------
    ValueError:
        If the input string cannot be mapped to a valid boolean state.
    """
    if isinstance(value, bool):
        return value
    val_lower = str(value).strip().lower()
    if val_lower in ['true', 't', 'yes', 'y', '1']:
        return True
    elif val_lower in ['false', 'f', 'no', 'n', '0']:
        return False
    else:
        raise ValueError("Please enter 'y' for True or 'n' for False.")

def parse_float_list(value):
    """
    Converts a comma-separated string to a list of floats.
    
    Statistical Context:
    This is primarily used to parse the Confidence Levels (CL). In the 
    Feldman-Cousins approach, the confidence level (1 - alpha) determines the 
    critical value of the Profile Likelihood Ratio test statistic. The framework 
    will compute intervals ensuring exact frequentist coverage at these specified 
    probabilities (e.g., 0.68 for 1-sigma, 0.90 for 90% CL).

    Parameters:
    -----------
    value : str or list
        The raw input string containing comma-separated numbers (e.g., "0.68, 0.90").

    Returns:
    --------
    list of float
        A list of floating-point numbers parsed from the input.
    """
    if isinstance(value, list):
        return value
    return [float(x.strip()) for x in str(value).split(',')]

def parse_str_list(value):
    """
    Converts a comma-separated string to a list of strings.
    
    Parameters:
    -----------
    value : str or list
        The raw input string containing comma-separated names.

    Returns:
    --------
    list of str
        A list of stripped string values (typically representing physical parameter names).
    """
    if isinstance(value, list):
        return value
    return [str(x.strip()) for x in str(value).split(',')]

def get_input(prompt_text, default_val, cast_func, choices=None, validator=None, error_msg="Invalid input."):
    """
    Core interactive prompt loop for CLI data entry.
    
    Handles rendering the prompt, intercepting empty inputs for default fallback, 
    parsing the input via the provided casting function, and executing all 
    validation rules (like bounds checking for statistical parameters).

    Parameters:
    -----------
    prompt_text : str
        The question or prompt presented to the user.
    default_val : any
        The fallback value used if the user submits an empty response (presses Enter).
    cast_func : callable
        A function (like int, str, or parse_bool) used to transform the string input 
        into the desired target data type.
    choices : list, optional
        A specific list of allowed values. If provided, input must exist in this list.
    validator : callable, optional
        A function that takes the parsed value and returns True if valid, False otherwise.
        Used for mathematical bounds (e.g., ensuring n_toys > 0).
    error_msg : str, optional
        The message displayed if the validator function fails.

    Returns:
    --------
    parsed_val : any
        The fully validated, type-casted user input (or the default value).
    """
    # Format the prompt text to include defaults and allowed choices
    formatted_prompt = f"{prompt_text}"
    if choices:
        formatted_prompt += f"\n  Allowed values: [{', '.join(map(str, choices))}]"
    formatted_prompt += f"\n  [Default: {default_val}]: "

    while True:
        try:
            user_input = input(formatted_prompt).strip()
            
            # Fall back to default if user just presses Enter
            if not user_input:
                current_val = default_val
            else:
                current_val = user_input
            
            # Cast the input to the required type
            parsed_val = cast_func(current_val)

            # Validate against allowed choices
            if choices is not None and parsed_val not in choices:
                print(f"  -> Error: '{parsed_val}' is not in the allowed list.\n")
                continue
                
            # Execute custom validation logic (e.g., range checks)
            if validator is not None and not validator(parsed_val):
                print(f"  -> Error: {error_msg}\n")
                continue

            print("") # Empty line for readability
            return parsed_val

        except ValueError as e:
            # Handle specific type-casting errors 
            err_str = str(e) if str(e) else "Invalid data type provided."
            print(f"  -> Error: {err_str}\n")
        except Exception:
            print(f"  -> Error: Invalid input format.\n")

def main():
    """
    Main entry point for the configuration generator.
    
    Statistical Theory & Configuration Impact:
    This sequence builds the `fc_config.json` matrix. It defines:
    - PDF formulation (Binned vs Unbinned) which changes the NLL definition.
    - Confidence Levels (CL) which sets the coverage integration targets.
    - Monte Carlo sampling size (n_toys), controlling the resolution of the empirical 
      test statistic PDF. Higher toys reduce statistical noise in the p-value calculation 
      at the expense of computation time.
    - Likelihood optimization strategies (SciPy, UltraNest, etc.) used to find the global 
      and conditional minima of the parameter space during profiling.
    - Advanced topological settings (adaptive toys, grid sparsification) which employ 
      heuristics to skip unnecessary toy generations in regions definitively outside 
      the targeted confidence bounds.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
        Outputs a serialized JSON file containing the validated configuration.
    """
    print("="*60)
    print(" Feldman-Cousins Configuration Generator")
    print(" Press [Enter] on any question to accept the default value.")
    print("="*60 + "\n")

    config = {}

    # --- Physics & Stats Constraints ---
    config["likelihood_type"] = get_input(
        "1. Likelihood Type?",
        default_val="binned",
        cast_func=str,
        choices=["binned", "unbinned"]
    )

    config["cl"] = get_input(
        "2. Confidence Levels (comma-separated)?",
        default_val="0.68, 0.90",
        cast_func=parse_float_list,
        validator=lambda lst: all(0.0 < x < 1.0 for x in lst),
        error_msg="Confidence levels must be floats between 0.0 and 1.0."
    )

    config["n_toys"] = get_input(
        "3. Number of Toys to generate?",
        default_val=2000,
        cast_func=int,
        validator=lambda x: x > 0,
        error_msg="Number of toys must be a positive integer."
    )

    # --- Optimizer Configurations ---
    config["strategy"] = get_input(
        "4. Optimizer Strategy?",
        default_val="scipy",
        cast_func=str,
        choices=["scipy", "ultranest", "hybrid", "grid"]
    )

    config["use_finite_mc_correction_binned"] = get_input(
        "5. Use finite MC correction for binned likelihoods (y/n)?",
        default_val="y",
        cast_func=parse_bool
    )

    # --- Compute Modes ---
    config["compute_1D_intervals"] = get_input(
        "6. Compute 1D Intervals (y/n)?",
        default_val="y",
        cast_func=parse_bool
    )

    config["compute_2D_intervals"] = get_input(
        "7. Compute 2D Intervals (y/n)?",
        default_val="y",
        cast_func=parse_bool
    )

    # --- Hardware & Execution Setup ---
    config["num_cores"] = get_input(
        "8. Number of CPU cores to use (enter 0 for maximum available)?",
        default_val=0,
        cast_func=int,
        validator=lambda x: x >= 0,
        error_msg="Number of cores must be 0 or a positive integer."
    )
    # Map 0 back to None for the orchestrator
    if config["num_cores"] == 0:
        config["num_cores"] = None

    config["verbose"] = get_input(
        "9. Verbosity level (0=Silent, 1=Standard, 2=Detailed)?",
        default_val=1,
        cast_func=int,
        choices=[0, 1, 2]
    )

    config["warm_start"] = get_input(
        "10. Enable Warm Start / Checkpointing (y/n)?",
        default_val="y",
        cast_func=parse_bool
    )

    # --- Plotting & Visuals ---
    config["param_names"] = get_input(
        "11. Parameter names for plotting (comma-separated)?",
        default_val="param1, param2, param3",
        cast_func=parse_str_list
    )

    config["smooth_1d"] = get_input(
        "12. Apply smoothing to 1D corner plot contours (y/n)?",
        default_val="n",
        cast_func=parse_bool
    )

    config["smooth_2d"] = get_input(
        "13. Apply smoothing to 2D corner plot contours (y/n)?",
        default_val="n",
        cast_func=parse_bool
    )

    # --- Advanced Optimizer Flags ---
    config["adaptive_toys"] = get_input(
        "14. Enable Adaptive Toys (y/n)?",
        default_val="y",
        cast_func=parse_bool
    )

    config["toy_batch_size"] = get_input(
        "15. Toy generation batch size?",
        default_val=200,
        cast_func=int,
        validator=lambda x: x > 0,
        error_msg="Batch size must be a positive integer."
    )

    config["sparsify_grid"] = get_input(
        "16. Enable Grid Sparsification (y/n)?",
        default_val="y",
        cast_func=parse_bool
    )

    # --- File I/O ---
    config["save_log"] = get_input(
        "17. Save output to a log file (y/n)?",
        default_val="n",
        cast_func=parse_bool
    )

    config["save_directory"] = get_input(
        "18. Output directory path?",
        default_val="fc_output",
        cast_func=str
    )

    config["output_file"] = get_input(
        "19. Output file prefix (without extension)?",
        default_val="fc_results",
        cast_func=str
    )

    out_json_path = get_input(
        "20. Path to save this configuration file to?",
        default_val="fc_config.json",
        cast_func=str
    )

    # --- Finalize and Export ---
    print("="*60)
    print(" Validating and generating configuration matrix...")
    
    try:
        # Ensure parent directories for the config file exist
        parent_dir = os.path.dirname(out_json_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(out_json_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f" SUCCESS: Configuration successfully saved to -> {out_json_path}")
        
    except Exception as e:
        print(f" ERROR: Failed to write configuration file: {e}")

if __name__ == "__main__":
    main()