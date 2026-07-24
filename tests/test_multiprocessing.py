"""
Tests for unbinned execution and multi-processing concurrency.
"""
import concurrent.futures

import numpy as np


# Note: The worker function MUST be defined at the top-level of the module.
# If it is nested inside the test function, ProcessPoolExecutor will raise 
# a PicklingError because nested functions cannot be serialized across processes.
def mock_unbinned_worker(seed):
    """
    A top-level mock worker function to simulate unbinned toy generation 
    and fitting across multiple CPU cores.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility in the sub-process.
        
    Returns
    -------
    float
        A simulated test statistic (the mean of the toy data).
    """
    # Initialize the random state local to this specific process
    np.random.seed(seed)
    
    # Simulate generating unbinned kinematic data (e.g., normal distribution)
    simulated_data = np.random.normal(loc=5.0, scale=1.0, size=100)
    
    return float(np.mean(simulated_data))


def test_process_pool_execution():
    """
    Verify that the ProcessPoolExecutor can successfully spawn workers, 
    serialize the target function, and gather results without pickling errors.
    This is critical for ensuring the unbinned Feldman-Cousins pipeline 
    will scale safely on HPC nodes.
    """
    n_toys = 20
    seeds = list(range(n_toys))
    results = []

    # Simulate the orchestrator's unbinned multiprocessing pool
    # Restrict max_workers to 2 so it safely runs on any standard CI/CD or local machine
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        # Map the mock worker to the list of seeds
        for res in executor.map(mock_unbinned_worker, seeds):
            results.append(res)
    
    # Verify we got exactly the requested number of toys back
    assert len(results) == n_toys, f"Expected {n_toys} results, but got {len(results)}."
    
    # Verify the math executed correctly in the isolated child processes
    # The mean of 100 draws from N(5, 1) should be tightly clustered around 5.0
    assert all(4.0 < r < 6.0 for r in results), "Worker returned an unexpected mathematical result."