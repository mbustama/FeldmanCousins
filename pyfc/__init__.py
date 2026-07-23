"""
Feldman-Cousins Frequentist Analysis Framework
"""

# Assuming the file is named orchestrator.py based on the previous code base
from .orchestrator import compute_fc_intervals

# You can also expose the config generator if you want it easily accessible
from .generate_config import main as generate_config

__all__ = [
    "compute_fc_intervals",
    "generate_config"
]