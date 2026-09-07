"""
Feldman-Cousins Frequentist Analysis Framework

Created: v0.1.0 (July 24, 2026)
Author: Mauricio Bustamante (mbustamante@gmail.com)
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

# Read the version from the installed distribution's metadata rather than repeating it
# here. `pyproject.toml` is the single source: a second copy in the source tree is a copy
# that will eventually disagree, and nothing fails when it does -- `docs/source/conf.py`
# hardcoded `release = '0.1.0'` and went unnoticed through nine releases, which is exactly
# the failure this avoids. conf.py now imports this value rather than repeating the lookup.
#
# The argument is the DISTRIBUTION name (`PyFeldmanCousins`), not the import package
# (`pyfc`). They are deliberately different here, and asking for the wrong one raises
# PackageNotFoundError, so getting it wrong would silently pin every consumer to the
# fallback below rather than failing loudly.
try:
    __version__ = _distribution_version("PyFeldmanCousins")
except PackageNotFoundError:
    # Running from a source tree that was never installed. A visibly wrong version is the
    # right answer here: anything plausible would be indistinguishable from a real one.
    __version__ = "0.0.0+unknown"

# Re-exported at the top level so `from pyfc import X` works without
# knowing PyFC's internal module layout.
from .generate_config import main as generate_config  # noqa: E402
from .orchestrator import compute_fc_intervals  # noqa: E402
from .plotting import generate_corner_plot  # noqa: E402
from .priors import (  # noqa: E402
    combine_priors,
    gaussian_block,
    gaussian_block_from_correlation,
)

__all__ = [
    "__version__",
    "combine_priors",
    "compute_fc_intervals",
    "gaussian_block",
    "gaussian_block_from_correlation",
    "generate_config",
    "generate_corner_plot",
]