"""
Plotting and Visualization Module

This module provides the visualization routines for the PyFC package, generating 
publication-quality corner plots that map the multi-dimensional parameter space. 
It visualizes both the 1D Profile Likelihood Ratios (with integrated Monte Carlo 
thresholds) and the 2D joint confidence contours using matplotlib.

Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at 
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np
import itertools
import os

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from scipy.ndimage import gaussian_filter1d, gaussian_filter
    SCIPY_NDIMAGE_AVAILABLE = True
except ImportError:
    SCIPY_NDIMAGE_AVAILABLE = False


def generate_corner_plot(results, config):
    """
    Generates a lower-triangle corner plot of the Feldman-Cousins confidence intervals.
    
    Statistical & Visual Context:
    The diagonal axes display the 1D Profile Likelihood Ratio (PLR). The solid black 
    curve represents the data test statistic ($t_{\text{data}}$), while the dashed 
    colored lines represent the exact critical thresholds ($t_{\text{critical}}$) 
    derived from Monte Carlo toys at specified Confidence Levels (CL). The shaded 
    regions denote the accepted parameter intervals where $t_{\text{data}} \leq t_{\text{critical}}$.
    
    The off-diagonal axes display the 2D joint contours. The contours are drawn exactly 
    where the differential surface $z = t_{\text{data}} - t_{\text{critical}}$ crosses zero, 
    providing a smooth boundary of the rigorous frequentist confidence region.

    Parameters:
    -----------
    results : dict
        The comprehensive results dictionary produced by `compute_fc_intervals`, containing 
        evaluated test statistics, parameter grids, and critical thresholds.
    config : dict
        Visualization configuration dictionary containing:
        - n_params (int): Number of physical parameters.
        - param_names (list of str): Axis labels.
        - cl (list of float): Confidence levels to plot.
        - smooth_1d, smooth_2d (bool): Toggles for Gaussian interpolation smoothing.
        - save_directory (str): Output path.

    Returns:
    --------
    None
        The figure is saved to disk as `fc_corner_plot.pdf` and the matplotlib 
        environment is closed to free memory.
    """
    if not MATPLOTLIB_AVAILABLE:
        return
        
    n_params = config.get("n_params", 1)
    p_names = config.get("param_names", [f"param{i+1}" for i in range(n_params)])
    cl_list = config["cl"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    smooth_1d = config.get("smooth_1d", False) and SCIPY_NDIMAGE_AVAILABLE
    smooth_2d = config.get("smooth_2d", False) and SCIPY_NDIMAGE_AVAILABLE
    
    # Base configuration: Dynamically scale figure size based on parameter space dimensions
    fig_size = max(5 * n_params, 10)
    fig, axs = plt.subplots(n_params, n_params, figsize=(fig_size, fig_size), gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    
    # Handle the 1D exception: Matplotlib returns a single Axes object rather than an array when N=1
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
            
            # Apply optional Gaussian smoothing to reduce numerical jitter in the test statistic
            if smooth_1d:
                t_dat = gaussian_filter1d(t_dat, sigma=1.0)
            
            ax.plot(x_test, t_dat, color='black', lw=1.5, label='Test Statistic')
            
            for idx, c in enumerate(cl_list):
                t_crit = results[f"1d_t_critical_p{i+1}"][c]
                
                if smooth_1d:
                    t_crit = gaussian_filter1d(t_crit, sigma=1.0)
                
                # Re-evaluate boolean acceptance logic to shade the exact enclosed envelope
                acc = t_dat <= t_crit
                
                ax.plot(x_test, t_crit, '--', color=colors[idx % len(colors)], label=f'{c} CL Threshold')
                ax.fill_between(x_test, 0, t_dat, where=acc, color=colors[idx % len(colors)], alpha=0.3)
                
            ax.set_ylim(bottom=0)
            ax.set_xlim(x_test[0], x_test[-1])
            ax.set_title(p_names[i])
            
            # De-duplicate legend labels and anchor to the upper right corner
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
            
            # Format tick labels to only display on the outer boundary of the corner plot
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
            # Map strictly to the lower triangle (row > col)
            row, col = fix_B, fix_A
            ax = axs[row, col]
            pair_name = f"p{fix_A+1}p{fix_B+1}"
            
            key_x = f"2d_test_p{fix_A+1}_{pair_name}"
            if key_x not in results: 
                continue
                
            grid_x = results[key_x]
            grid_y = results[f"2d_test_p{fix_B+1}_{pair_name}"]
            t_dat_2d = results[f"2d_t_data_{pair_name}"]
            
            if smooth_2d:
                t_dat_2d = gaussian_filter(t_dat_2d, sigma=1.0)
            
            X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
            # Render a faint background heatmap of the global test statistic
            ax.pcolormesh(X, Y, t_dat_2d, cmap='Blues', shading='auto', alpha=0.2)
            
            for idx, c in enumerate(cl_list):
                t_crit_2d = results[f"2d_t_critical_{pair_name}"][c]
                
                if smooth_2d:
                    t_crit_2d = gaussian_filter(t_crit_2d, sigma=1.0)
                
                # Compute continuous differential surface for exact root-finding interpolation
                z_diff = t_dat_2d - t_crit_2d
                
                # Ensure a mathematical zero-crossing exists within the evaluated array 
                # to prevent matplotlib contouring warnings when bounds are unconstrained
                if np.min(z_diff) <= 0.0 <= np.max(z_diff):
                    ax.contour(X, Y, z_diff, levels=[0.0], colors=[colors[idx % len(colors)]], linewidths=2)
                
                # Generate a dummy line exclusively to populate the legend on the bottom-left plot
                if row == n_params - 1 and col == 0: 
                    ax.plot([], [], color=colors[idx % len(colors)], linewidth=2, label=f'{c} CL Contour')
            
            # Highlight the global unconditional Maximum Likelihood Estimate (MLE)
            if "best_fit" in results:
                best_vals = results["best_fit"]
                ax.scatter([best_vals[fix_A]], [best_vals[fix_B]], color='black', marker='*', s=150)
                
            ax.set_xlim(grid_x[0], grid_x[-1])
            ax.set_ylim(grid_y[0], grid_y[-1])
            
            if row == n_params - 1 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
            
            # Format tick labels to adhere to the corner plot exterior
            if row != n_params - 1: 
                ax.tick_params(labelbottom=False)
            else: 
                ax.set_xlabel(p_names[col])
            
            if col != 0: 
                ax.tick_params(labelleft=False)
            else: 
                ax.set_ylabel(p_names[row])

    # Visually disable and hide the entire upper triangle of the subplot matrix
    for i in range(n_params):
        for j in range(n_params):
            if i < j:
                axs[i, j].axis('off')

    plot_path = os.path.join(config.get("save_directory", "."), "fc_corner_plot.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()