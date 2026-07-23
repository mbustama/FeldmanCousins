import numpy as np

# --- 1. Core Math (Unbinned) ---
def calc_nll_unbinned(sig_scale, bkg_scale, obs_events, S_pdf, B_pdf):
    """
    [UNBINNED] Computes the Extended Unbinned Maximum Likelihood (EUML).
    Uses standard numpy vectorization since Numba cannot easily JIT arbitrary callables.
    """
    expected_total = sig_scale + bkg_scale
    if len(obs_events) == 0:
        return expected_total

    # S_pdf and B_pdf should be callable functions (e.g., splines) returning probabilities
    s_probs = S_pdf(obs_events)
    b_probs = B_pdf(obs_events)
    
    p_events = sig_scale * s_probs + bkg_scale * b_probs
    
    if np.any(p_events <= 0):
        return 1e10 # Heavy penalty for impossible kinematics
        
    # Extended NLL formulation
    return expected_total - np.sum(np.log(p_events))

def generate_unbinned_toy(sig_scale, bkg_scale, S_mc_pool, B_mc_pool):
    """
    [UNBINNED] Generates an unbinned toy using Monte Carlo bootstrapping (resampling).
    """
    n_sig = np.random.poisson(sig_scale)
    n_bkg = np.random.poisson(bkg_scale)
    
    toy_events = []
    # Draw kinematics with replacement from the raw MC pools
    if n_sig > 0 and S_mc_pool is not None and len(S_mc_pool) > 0:
        toy_events.extend(np.random.choice(S_mc_pool, size=n_sig, replace=True))
    if n_bkg > 0 and B_mc_pool is not None and len(B_mc_pool) > 0:
        toy_events.extend(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
        
    return np.array(toy_events)


# --- 2. Grid Search Optimizers (Unbinned) ---
def unconditional_fit_grid_unbinned(obs_events, S_pdf, B_pdf, sig_grid, bkg_grid):
    min_nll = 1e10
    best_sig, best_bkg = sig_grid[0], bkg_grid[0]
    
    for s in sig_grid:
        for b in bkg_grid:
            nll = calc_nll_unbinned(s, b, obs_events, S_pdf, B_pdf)
            if nll < min_nll:
                min_nll = nll
                best_sig, best_bkg = s, b
    return min_nll, best_sig, best_bkg

def conditional_fit_grid_unbinned(test_sig, obs_events, S_pdf, B_pdf, bkg_grid):
    min_nll = 1e10
    best_bkg = bkg_grid[0]
    
    for b in bkg_grid:
        nll = calc_nll_unbinned(test_sig, b, obs_events, S_pdf, B_pdf)
        if nll < min_nll:
            min_nll = nll
            best_bkg = b
    return min_nll, best_bkg

def generate_and_fit_toys_grid_unbinned(test_sig, profiled_bkg, S_pdf, B_pdf, 
                                        sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool):
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(test_sig, profiled_bkg, S_mc_pool, B_mc_pool)
        uncond_nll, _, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, sig_grid, bkg_grid)
        cond_nll, _ = conditional_fit_grid_unbinned(test_sig, toy_events, S_pdf, B_pdf, bkg_grid)
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics