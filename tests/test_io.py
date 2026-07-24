"""
Tests for File I/O and checkpoint recovery mechanisms.
"""
import os
import numpy as np
import tempfile

def test_checkpoint_read_write():
    """
    Verify that intermediate array states can be saved to an .npz 
    checkpoint and perfectly reconstructed.
    """
    # Create a temporary directory so we don't clutter the user's system
    with tempfile.TemporaryDirectory() as tmpdirname:
        checkpoint_path = os.path.join(tmpdirname, "checkpoint_fc.npz")
        
        # 1. Mock some intermediate data (e.g., halfway through a 1D scan)
        mock_grid = np.linspace(0, 10, 50)
        mock_t_data = np.random.uniform(0, 5, size=50)
        
        # 2. Save the state
        np.savez(
            checkpoint_path,
            grid_p1=mock_grid,
            t_data_p1=mock_t_data
        )
        
        # 3. Assert the file was actually written
        assert os.path.exists(checkpoint_path)
        
        # 4. Load the state and verify integrity
        loaded = np.load(checkpoint_path)
        assert "grid_p1" in loaded.files
        assert "t_data_p1" in loaded.files
        
        np.testing.assert_array_equal(loaded["grid_p1"], mock_grid)
        np.testing.assert_array_equal(loaded["t_data_p1"], mock_t_data)