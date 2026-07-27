"""
Feldman-Cousins Frequentist Analysis Framework

Created: v0.1.0 (July 24, 2026)
Last modified: v0.10.0
Author: Mauricio Bustamante (mbustamante@gmail.com)
"""

# Re-exported at the top level so `from pyfc import X` works without
# knowing PyFC's internal module layout.
from .generate_config import main as generate_config
from .orchestrator import compute_fc_intervals
from .plotting import generate_corner_plot

__all__ = [
    "compute_fc_intervals",
    "generate_config",
    "generate_corner_plot",
]