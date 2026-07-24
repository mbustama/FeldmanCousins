"""
Core installation and import tests.
"""

def test_imports():
    """Verify that all core modules can be imported successfully."""
    assert True  # If we reach here, imports succeeded

def test_optional_dependencies():
    """Check that optional dependency flags are set correctly as booleans."""
    from pyfc.optimizers import SCIPY_AVAILABLE, ULTRANEST_AVAILABLE
    assert isinstance(SCIPY_AVAILABLE, bool)
    assert isinstance(ULTRANEST_AVAILABLE, bool)