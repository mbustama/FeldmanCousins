"""
Cross-branch regression harness: OLD API script, targets `dev`'s pyfc
(S_model/B_model still required positional args; compute_rates_func takes
S_template/B_template or s_probs/b_probs directly).

Must be run with a `dev`-branch worktree's root passed via --pyfc-root, so
that `import pyfc` resolves to that checkout and not whatever `pyfc` (if
any) is already installed/importable in the current environment -- a stale
pip-installed pyfc has silently shadowed the local checkout before in this
repo's history, so this is verified explicitly below rather than assumed.

Produces scenario_a.npz and scenario_c.npz for xbranch_new_api.py /
comparator.py to diff against the new-API equivalents.
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--pyfc-root", required=True, help="Root of the `dev`-branch worktree containing pyfc/")
parser.add_argument("--output-dir", required=True, help="Directory to write scenario_*.npz into")
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # shared_constants.py
sys.path.insert(0, os.path.abspath(args.pyfc_root))

import numpy as np
from numba import njit

import pyfc
import shared_constants as sc

resolved = os.path.abspath(pyfc.__file__)
expected_root = os.path.abspath(args.pyfc_root)
print(f"[xbranch-old] pyfc.__file__ = {resolved}")
if not resolved.startswith(expected_root):
    raise RuntimeError(
        f"pyfc resolved from '{resolved}', which is NOT under the requested "
        f"--pyfc-root '{expected_root}' -- a different pyfc (stale pip install?) "
        f"is shadowing the worktree checkout. Refusing to trust these numbers."
    )

from pyfc.orchestrator import compute_fc_intervals

os.makedirs(args.output_dir, exist_ok=True)


# --- Scenario (a): binned, 1D data, 3-parameter model ---
@njit(fastmath=True, nogil=True)
def compute_rates_binned_old(params, S_template, B_template, S_sigma2, B_sigma2):
    mu = params[0] * S_template + params[1] + params[2] * B_template
    sigma2 = (params[0] ** 2) * S_sigma2 + (params[2] ** 2) * B_sigma2
    return mu, sigma2


def run_scenario_a():
    np.random.seed(sc.RNG_SEED)
    mu_true = (sc.TRUE_PARAMS_BINNED[0] * sc.S_TEMPLATE_FLAT + sc.TRUE_PARAMS_BINNED[1]
               + sc.TRUE_PARAMS_BINNED[2] * sc.B_TEMPLATE_FLAT)
    data = np.random.poisson(mu_true).astype(float)
    s_sigma2 = np.zeros_like(data)
    b_sigma2 = np.zeros_like(data)

    np.random.seed(sc.RNG_SEED)  # re-seed so toy generation starts from a known state
    compute_fc_intervals(
        data, sc.S_TEMPLATE_FLAT, sc.B_TEMPLATE_FLAT, sc.GRIDS_3PARAM,
        compute_rates_func=compute_rates_binned_old,
        cl=sc.CL, n_toys=sc.N_TOYS, strategy="scipy", num_cores=sc.NUM_CORES, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="binned", S_sigma2=s_sigma2, B_sigma2=b_sigma2,
        compute_1D_intervals=True, compute_2D_intervals=False,
        output_file="scenario_a", save_directory=args.output_dir,
    )
    print("[xbranch-old] scenario (a) done -> scenario_a.npz")


# --- Scenario (c): unbinned, 2 pdf components (signal + background) ---
def s_pdf(x):
    from scipy.stats import norm
    return norm.pdf(x, loc=sc.S_PDF_LOC, scale=sc.S_PDF_SCALE)


def b_pdf(x):
    from scipy.stats import expon
    return expon.pdf(x, scale=sc.B_PDF_SCALE)


def compute_rates_unbinned_old(params, s_probs, b_probs):
    p_events = params[0] * s_probs + params[1] * b_probs
    return params[0] + params[1], p_events


def generate_toy_unbinned(true_params, S_mc_pool, B_mc_pool):
    n_sig = np.random.poisson(true_params[0])
    n_bkg = np.random.poisson(true_params[1])
    idx_s = np.random.choice(len(S_mc_pool), size=n_sig, replace=True)
    idx_b = np.random.choice(len(B_mc_pool), size=n_bkg, replace=True)
    return np.concatenate([S_mc_pool[idx_s], B_mc_pool[idx_b]])


def run_scenario_c():
    events = sc.UNBINNED_EVENTS_2C

    np.random.seed(sc.RNG_SEED)
    s_mc_pool = np.random.normal(loc=sc.S_PDF_LOC, scale=sc.S_PDF_SCALE, size=sc.MC_POOL_SIZE)
    b_mc_pool = np.random.exponential(scale=sc.B_PDF_SCALE, size=sc.MC_POOL_SIZE)

    np.random.seed(sc.RNG_SEED)  # re-seed so toy generation starts from a known state
    compute_fc_intervals(
        events, s_pdf, b_pdf, sc.GRIDS_2PARAM,
        compute_rates_func=compute_rates_unbinned_old,
        generate_toy_func=generate_toy_unbinned,
        cl=sc.CL, n_toys=sc.N_TOYS, strategy="scipy", num_cores=sc.NUM_CORES, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="unbinned", S_mc_pool=s_mc_pool, B_mc_pool=b_mc_pool,
        compute_1D_intervals=True, compute_2D_intervals=False,
        output_file="scenario_c", save_directory=args.output_dir,
    )
    print("[xbranch-old] scenario (c) done -> scenario_c.npz")


if __name__ == "__main__":
    run_scenario_a()
    run_scenario_c()
