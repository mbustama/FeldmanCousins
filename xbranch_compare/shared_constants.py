"""
Shared numeric constants for the dev vs dev-no-templates cross-branch
regression harness.

Deliberately has NO import of `pyfc` -- it must be importable unchanged
regardless of which worktree's `pyfc` package ends up first on sys.path,
since xbranch_old_api.py and xbranch_new_api.py each need to see the exact
same numbers to rule out transcription drift masquerading as a "real"
numerical discrepancy.
"""
import numpy as np

RNG_SEED = 20260727

# --- Scenarios (a) & (b): binned, 3-parameter model ---
# 6 bins, so scenario (b) can reshape this into a genuine 2D (2, 3)
# histogram using the exact same physical model/data as scenario (a)'s
# 1D version -- see xbranch_new_api.py.
S_TEMPLATE_FLAT = np.array([0.1, 0.5, 2.0, 5.0, 1.0, 3.0])
B_TEMPLATE_FLAT = np.array([15.0, 5.0, 1.0, 0.1, 2.0, 4.0])
BINNED_2D_SHAPE = (2, 3)

TRUE_PARAMS_BINNED = np.array([1.0, 1.0, 1.0])

GRIDS_3PARAM = [
    np.linspace(0.5, 2.0, 5),
    np.linspace(0.5, 2.0, 5),
    np.linspace(0.5, 1.5, 5),
]

# --- Scenario (c): unbinned, 2 components (signal + background) ---
S_PDF_LOC, S_PDF_SCALE = 5.0, 1.0
B_PDF_SCALE = 2.0
TRUE_PARAMS_UNBINNED_2C = np.array([5.0, 3.0])
GRIDS_2PARAM = [
    np.linspace(1.0, 10.0, 5),
    np.linspace(1.0, 10.0, 5),
]
UNBINNED_EVENTS_2C = np.array([4.8, 5.2, 6.1, 4.5, 0.9, 1.8, 3.2])
MC_POOL_SIZE = 3000

# --- Scenario (d): unbinned, 3 components (new-branch-only capability,
# no old-branch equivalent -- validated against a hand-computed reference
# NLL instead of a cross-branch diff) ---
UNBINNED_EVENTS_3C = np.array([0.0, 1.0, 2.0])
TEST_PARAMS_3C = [
    np.array([1.0, 2.0, 3.0]),
    np.array([2.0, 0.5, 1.5]),
]

# Deliberately small for harness speed (this runs on every invocation, unlike
# the dedicated test suite). Below toys.ADAPTIVE_MIN_TOYS (100), so this
# harness never exercises adaptive_toys's actual early-stopping behavior --
# that needs its own dedicated coverage instead, see tests/test_adaptive_toys.py.
N_TOYS = 15
CL = [0.90]
NUM_CORES = 1
