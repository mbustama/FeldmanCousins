import numpy as np

# ==============================================================================
# --- 1. USER DEFINED PHYSICS MODEL (UNBINNED) ---
# Modify this section to define how your N parameters map to the PDFs.
# ==============================================================================
def compute_rates_unbinned(params, s_probs, b_probs):
    """
    Maps an N-dimensional parameter array to expected total events and 
    the probability density of observed events.
    Example provided: 3-parameter model.
    """
    expected_total = params[0] * params[1] + params[2]
    
    if len(s_probs) == 0 and len(b_probs) == 0:
        return expected_total, np.array([])
        
    p_events = params[0] * params[1] * s_probs + params[2] * b_probs
    
    return expected_total, p_events

def generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool):
    """
    Maps an N-dimensional parameter array to the toy resampling process.
    Example provided: 3-parameter model.
    """
    n_sig = np.random.poisson(true_params[0] * true_params[1])
    n_bkg = np.random.poisson(true_params[2])
    
    parts = []
    if n_sig > 0 and S_mc_pool is not None and len(S_mc_pool) > 0:
        parts.append(np.random.choice(S_mc_pool, size=n_sig, replace=True))
    if n_bkg > 0 and B_mc_pool is not None and len(B_mc_pool) > 0:
        parts.append(np.random.choice(B_mc_pool, size=n_bkg, replace=True))
        
    if parts:
        return np.concatenate(parts)
    return np.array([])
# ==============================================================================


# --- 2. Core Math (Unbinned) ---
def calc_nll_unbinned(params, len_obs, s_probs, b_probs):
    """
    [UNBINNED] Computes the Extended Unbinned Maximum Likelihood (EUML).
    """
    expected_total, p_events = compute_rates_unbinned(params, s_probs, b_probs)
    
    if len_obs == 0:
        return expected_total

    if np.any(p_events <= 0):
        return 1e10 # Heavy penalty for impossible kinematics
        
    return expected_total - np.sum(np.log(p_events))


# --- 3. Grid Search Optimizers (Unbinned) ---
def unconditional_fit_grid_unbinned(obs_events, S_pdf, B_pdf, full_grid_points):
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_params = full_grid_points[0].copy()
    
    for row in range(len(full_grid_points)):
        p = full_grid_points[row]
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

def conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, obs_events, S_pdf, B_pdf, cond_grid_points):
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_params = np.zeros(n_params)
    p = np.zeros(n_params)
    p[fix_idx] = test_val
    
    for row in range(len(cond_grid_points)):
        free_p = cond_grid_points[row]
        free_i = 0
        for i in range(n_params):
            if i != fix_idx:
                p[i] = free_p[free_i]
                free_i += 1
                
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params

def conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, obs_events, S_pdf, B_pdf, cond_grid_points):
    len_obs = len(obs_events)
    s_probs = S_pdf(obs_events) if len_obs > 0 else np.array([])
    b_probs = B_pdf(obs_events) if len_obs > 0 else np.array([])
    
    min_nll = 1e10
    best_params = np.zeros(n_params)
    p = np.zeros(n_params)
    p[fix_A] = test_vA
    p[fix_B] = test_vB
    
    for row in range(len(cond_grid_points)):
        free_p = cond_grid_points[row]
        free_i = 0
        for i in range(n_params):
            if i != fix_A and i != fix_B:
                p[i] = free_p[free_i]
                free_i += 1
                
        nll = calc_nll_unbinned(p, len_obs, s_probs, b_probs)
        if nll < min_nll:
            min_nll = nll
            best_params = p.copy()
            
    return min_nll, best_params


# --- 4. Toy Generators (Unbinned) ---
def generate_and_fit_toys_grid_unbinned_1d(test_val, fix_idx, true_params, n_params, S_pdf, B_pdf, 
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool):
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, full_grid_points)
        cond_nll, _ = conditional_fit_grid_unbinned_1d(test_val, fix_idx, n_params, toy_events, S_pdf, B_pdf, cond_grid_points)
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics

def generate_and_fit_toys_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, true_params, n_params, S_pdf, B_pdf, 
                                           full_grid_points, cond_grid_points, n_toys, S_mc_pool, B_mc_pool):
    t_statistics = np.zeros(n_toys)
    for t in range(n_toys):
        toy_events = generate_unbinned_toy(true_params, S_mc_pool, B_mc_pool)
        uncond_nll, _ = unconditional_fit_grid_unbinned(toy_events, S_pdf, B_pdf, full_grid_points)
        cond_nll, _ = conditional_fit_grid_unbinned_2d(test_vA, test_vB, fix_A, fix_B, n_params, toy_events, S_pdf, B_pdf, cond_grid_points)
        t_statistics[t] = max(0.0, cond_nll - uncond_nll)
    return t_statistics