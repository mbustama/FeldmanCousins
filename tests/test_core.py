"""
Core installation and import tests.
"""

def test_imports():
    """
    Verify that all core modules can be imported successfully.

    Regression test: this previously never actually imported anything --
    it was `assert True` with a comment claiming imports had already
    succeeded, so it passed unconditionally even if every import below
    was broken.
    """
    import pyfc
    import pyfc.binned
    import pyfc.config
    import pyfc.generate_config
    import pyfc.optimizers
    import pyfc.orchestrator
    import pyfc.plotting
    import pyfc.toys
    import pyfc.unbinned

    assert pyfc.compute_fc_intervals is pyfc.orchestrator.compute_fc_intervals

def test_optional_dependencies():
    """Check that optional dependency flags are set correctly as booleans."""
    from pyfc.optimizers import SCIPY_AVAILABLE, ULTRANEST_AVAILABLE
    assert isinstance(SCIPY_AVAILABLE, bool)
    assert isinstance(ULTRANEST_AVAILABLE, bool)