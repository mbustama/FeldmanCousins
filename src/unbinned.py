import numpy as np

# --- 1. Core Math (Unbinned) ---
def calc_nll_unbinned(sig_scale, bkg_scale, len_obs, s_probs, b_probs):
    """
    [UNBINNED] Computes the Extended Unbinned Maximum Likelihood (EUML).
    Uses standard numpy vectorization since Numba cannot easily JIT arbitrary callables.
    """
    expected_total = sig_scale + bkg_scale
    if len_obs == 0:
        return expected_total

    # S_pdf and B_pdf should be callable functions (e.g., splines) returning probabilities
    # These are now evaluated prior to the optimizer call and passed as s_probs/b_probs
    
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
    
    parts = []
    # Draw kinematics with replacement from the raw MC pools
    if n_sig > 0 and S_mc_pool is not None and len(S_mc_pool) > 0:
        parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
    if n_bkg > 0 and B_mc_pool is not None and len(B_mc_pool) > 0:
        parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
        
    if parts:
        return np.concatenate(parts)
    return np.array([])


# --- 2. Grid Search Optimizers (Unbinned) ---
def unconditional_fit_grid_unbinned(obs_events, S_pdf, B_pdf, sig_grid, bkg_grid):
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_sig, best_bkg = sig_grid[0], bkg_grid[0]
    
    for s in sig_grid:
        for b in bkg_grid:
            nll = calc_nll_unbinned(s, b, len_obs, s_probs, b_probs)
            if nll < min_nll:
                min_nll = nll
                best_sig, best_bkg = s, b
    return min_nll, best_sig, best_bkg

def conditional_fit_grid_unbinned(test_sig, obs_events, S_pdf, B_pdf, bkg_grid):
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_bkg = bkg_grid[0]
    
    for b in bkg_grid:
        nll = calc_nll_unbinned(test_sig, b, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_bkg = b
    return min_nll, best_bkg

def conditional_fit_grid_unbinned_profile_sig(test_bkg, obs_events, S_pdf, B_pdf, sig_grid):
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_sig = sig_grid[0]
    
    for s in sig_grid:
        nll = calc_nll_unbinned(s, test_bkg, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_sig = s
    return min_nll, best_sig

# --- 3. Toy Generators (Unbinned) ---
def generate_and_fit_toys_grid_unbinned(test_sig, profiled_bkg, S_pdf, B_pdf, 
                                        sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool):
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(test_sig, profiled_bkg, S_mc_pool, B_mc_pool)
        uncond_nll, _, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, sig_grid, bkg_grid)
        cond_nll, _ = conditional_fit_grid_unbinned(test_sig, toy_events, S_pdf, B_pdf, bkg_grid)
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics

def generate_and_fit_toys_grid_unbinned_profile_sig(test_bkg, profiled_sig, S_pdf, B_pdf, 
                                                    sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool):
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(profiled_sig, test_bkg, S_mc_pool, B_mc_pool)
        uncond_nll, _, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, sig_grid, bkg_grid)
        cond_nll, _ = conditional_fit_grid_unbinned_profile_sig(test_bkg, toy_events, S_pdf, B_pdf, sig_grid)
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics

def generate_and_fit_toys_grid_unbinned_2d(test_sig, test_bkg, S_pdf, B_pdf, 
                                           sig_grid, bkg_grid, n_toys, S_mc_pool, B_mc_pool):
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(test_sig, test_bkg, S_mc_pool, B_mc_pool)
        uncond_nll, _, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, sig_grid, bkg_grid)
        
        len_obs = len(toy_events)
        s_probs = S_pdf(toy_events) if len_obs > 0 else np.array([])
        b_probs = B_pdf(toy_events) if len_obs > 0 else np.array([])
        cond_nll = calc_nll_unbinned(test_sig, test_bkg, len_obs, s_probs, b_probs)
        
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics