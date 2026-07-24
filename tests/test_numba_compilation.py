"""
Tests to ensure Numba JIT compilation executes correctly on the host architecture.
"""
import numpy as np
from numba import njit


def test_numba_jit_compilation():
    """
    Force a simple Numba JIT compilation to ensure the user's 
    LLVM and compiler toolchain were installed correctly.
    """
    @njit(fastmath=True, nogil=True)
    def dummy_compute_rates(params, S_temp, B_temp):
        return params[0] * S_temp + params[1] * B_temp

    params = np.array([1.0, 2.0])
    S = np.array([0.5, 0.5])
    B = np.array([1.0, 1.0])
    
    # The first call triggers the LLVM compilation phase
    result = dummy_compute_rates(params, S, B)
    
    assert result is not None
    assert len(result) == 2