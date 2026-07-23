import numpy as np
from .binned import calc_nll
from .unbinned import calc_nll_unbinned

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import ultranest
    ULTRANEST_AVAILABLE = True
except ImportError:
    ULTRANEST_AVAILABLE = False


# --- 1. Scipy Continuous Optimizers ---
def unconditional_fit_scipy(data, S_model, B_model, n_params, bounds_list, seed=None, 
                            likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def cost(params):
            return calc_nll_unbinned(params, len_obs, s_probs, b_probs)
    else:
        def cost(params):
            return calc_nll(params, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = seed if seed is not None else [(b[0] + b[1]) / 2.0 for b in bounds_list]
    res = optimize.minimize(cost, x0=x0, bounds=bounds_list, method='L-BFGS-B')
    return res.fun, res.x

def conditional_fit_1d_scipy(test_val, fix_idx, n_params, data, S_model, B_model, bounds_list, seed=None, 
                             likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    free_indices = [i for i in range(n_params) if i != fix_idx]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = seed if seed is not None else [(b[0] + b[1]) / 2.0 for b in free_bounds]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_idx] = test_val
        return cost(np.array([])), p
        
    res = optimize.minimize(cost, x0=x0, bounds=free_bounds, method='L-BFGS-B')
    
    best_p = np.zeros(n_params)
    best_p[fix_idx] = test_val
    for idx, free_val in zip(free_indices, res.x):
        best_p[idx] = free_val
        
    return res.fun, best_p

def conditional_fit_2d_scipy(test_vA, test_vB, fix_A, fix_B, n_params, data, S_model, B_model, bounds_list, seed=None, 
                             likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    free_indices = [i for i in range(n_params) if i not in (fix_A, fix_B)]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def cost(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    x0 = seed if seed is not None else [(b[0] + b[1]) / 2.0 for b in free_bounds]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_A] = test_vA
        p[fix_B] = test_vB
        return cost(np.array([])), p
        
    res = optimize.minimize(cost, x0=x0, bounds=free_bounds, method='L-BFGS-B')
    
    best_p = np.zeros(n_params)
    best_p[fix_A] = test_vA
    best_p[fix_B] = test_vB
    for idx, free_val in zip(free_indices, res.x):
        best_p[idx] = free_val
        
    return res.fun, best_p


# --- 2. UltraNest Optimizers ---
def unconditional_fit_ultranest(data, S_model, B_model, n_params, bounds_list, verbose=1, 
                                likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    def prior_transform(cube):
        return np.array([cube[i] * (bounds_list[i][1] - bounds_list[i][0]) + bounds_list[i][0] for i in range(n_params)])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def log_likelihood(p): 
            return -calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def log_likelihood(p): 
            return -calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    param_names = [f'p{i+1}' for i in range(n_params)]
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    return -result['maximum_likelihood']['logl'], result['maximum_likelihood']['point']

def conditional_fit_1d_ultranest(test_val, fix_idx, n_params, data, S_model, B_model, bounds_list, verbose=1, 
                                 likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    free_indices = [i for i in range(n_params) if i != fix_idx]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_idx] = test_val
        if likelihood_type == "unbinned":
            s_probs = S_model(data) if len(data) > 0 else np.array([])
            b_probs = B_model(data) if len(data) > 0 else np.array([])
            return calc_nll_unbinned(p, len(data), s_probs, b_probs), p
        else:
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc), p

    def prior_transform(cube):
        return np.array([cube[i] * (free_bounds[i][1] - free_bounds[i][0]) + free_bounds[i][0] for i in range(len(free_bounds))])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_idx] = test_val
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    param_names = [f'f{i+1}' for i in range(len(free_bounds))]
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    
    best_p = np.zeros(n_params)
    best_p[fix_idx] = test_val
    for idx, free_val in zip(free_indices, result['maximum_likelihood']['point']):
        best_p[idx] = free_val
        
    return -result['maximum_likelihood']['logl'], best_p

def conditional_fit_2d_ultranest(test_vA, test_vB, fix_A, fix_B, n_params, data, S_model, B_model, bounds_list, verbose=1, 
                                 likelihood_type="binned", S_sigma2=None, B_sigma2=None, use_finite_mc=False):
    free_indices = [i for i in range(n_params) if i not in (fix_A, fix_B)]
    free_bounds = [bounds_list[i] for i in free_indices]
    
    if len(free_bounds) == 0:
        p = np.zeros(n_params)
        p[fix_A] = test_vA
        p[fix_B] = test_vB
        if likelihood_type == "unbinned":
            s_probs = S_model(data) if len(data) > 0 else np.array([])
            b_probs = B_model(data) if len(data) > 0 else np.array([])
            return calc_nll_unbinned(p, len(data), s_probs, b_probs), p
        else:
            return calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc), p
            
    def prior_transform(cube):
        return np.array([cube[i] * (free_bounds[i][1] - free_bounds[i][0]) + free_bounds[i][0] for i in range(len(free_bounds))])
        
    if likelihood_type == "unbinned":
        len_obs = len(data)
        s_probs = S_model(data) if len_obs > 0 else np.array([])
        b_probs = B_model(data) if len_obs > 0 else np.array([])
        
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll_unbinned(p, len_obs, s_probs, b_probs)
    else:
        def log_likelihood(free_p):
            p = np.zeros(n_params)
            p[fix_A] = test_vA
            p[fix_B] = test_vB
            for idx, free_val in zip(free_indices, free_p):
                p[idx] = free_val
            return -calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2, use_finite_mc)
            
    param_names = [f'f{i+1}' for i in range(len(free_bounds))]
    sampler = ultranest.ReactiveNestedSampler(param_names, log_likelihood, prior_transform, log_dir=None)
    run_kwargs = {'min_num_live_points': 50, 'dKL': np.inf, 'min_ess': 50, 'show_status': (verbose == 2), 'viz_callback': False}
    result = sampler.run(**run_kwargs)
    
    best_p = np.zeros(n_params)
    best_p[fix_A] = test_vA
    best_p[fix_B] = test_vB
    for idx, free_val in zip(free_indices, result['maximum_likelihood']['point']):
        best_p[idx] = free_val
        
    return -result['maximum_likelihood']['logl'], best_p