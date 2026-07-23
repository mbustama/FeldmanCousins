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
    if not MATPLOTLIB_AVAILABLE:
        return
        
    n_params = config.get("n_params", 1)
    p_names = config.get("param_names", [f"param{i+1}" for i in range(n_params)])
    cl_list = config["cl"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    smooth_1d = config.get("smooth_1d", False) and SCIPY_NDIMAGE_AVAILABLE
    smooth_2d = config.get("smooth_2d", False) and SCIPY_NDIMAGE_AVAILABLE
    
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
            
            if smooth_1d:
                t_dat = gaussian_filter1d(t_dat, sigma=1.0)
            
            ax.plot(x_test, t_dat, color='black', lw=1.5, label='Test Statistic')
            
            for idx, c in enumerate(cl_list):
                t_crit = results[f"1d_t_critical_p{i+1}"][c]
                
                if smooth_1d:
                    t_crit = gaussian_filter1d(t_crit, sigma=1.0)
                
                # Re-evaluate acceptance logic for the fill envelope
                acc = t_dat <= t_crit
                
                ax.plot(x_test, t_crit, '--', color=colors[idx % len(colors)], label=f'{c} CL Threshold')
                ax.fill_between(x_test, 0, t_dat, where=acc, color=colors[idx % len(colors)], alpha=0.3)
                
            ax.set_ylim(bottom=0)
            ax.set_xlim(x_test[0], x_test[-1])
            ax.set_title(p_names[i])
            
            # De-duplicate legend labels and set to upper right
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
            
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
            
            if smooth_2d:
                t_dat_2d = gaussian_filter(t_dat_2d, sigma=1.0)
            
            X, Y = np.meshgrid(grid_x, grid_y, indexing='ij')
            ax.pcolormesh(X, Y, t_dat_2d, cmap='Blues', shading='auto', alpha=0.2)
            
            for idx, c in enumerate(cl_list):
                t_crit_2d = results[f"2d_t_critical_{pair_name}"][c]
                
                if smooth_2d:
                    t_crit_2d = gaussian_filter(t_crit_2d, sigma=1.0)
                
                # Compute continuous differential surface for exact interpolation
                z_diff = t_dat_2d - t_crit_2d
                
                # Ensure a zero-crossing exists within the array to prevent contour warnings
                if np.min(z_diff) <= 0.0 <= np.max(z_diff):
                    ax.contour(X, Y, z_diff, levels=[0.0], colors=[colors[idx % len(colors)]], linewidths=2)
                
                # Dummy line for exact legends
                if row == n_params - 1 and col == 0: 
                    ax.plot([], [], color=colors[idx % len(colors)], linewidth=2, label=f'{c} CL Contour')
            
            if "best_fit" in results:
                best_vals = results["best_fit"]
                ax.scatter([best_vals[fix_A]], [best_vals[fix_B]], color='black', marker='*', s=150)
                
            ax.set_xlim(grid_x[0], grid_x[-1])
            ax.set_ylim(grid_y[0], grid_y[-1])
            
            if row == n_params - 1 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
            
            if row != n_params - 1: 
                ax.tick_params(labelbottom=False)
            else: 
                ax.set_xlabel(p_names[col])
            
            if col != 0: 
                ax.tick_params(labelleft=False)
            else: 
                ax.set_ylabel(p_names[row])

    # Hide upper triangle completely
    for i in range(n_params):
        for j in range(n_params):
            if i < j:
                axs[i, j].axis('off')

    plot_path = os.path.join(config.get("save_directory", "."), "fc_corner_plot.pdf")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()