"""
Cross-branch regression harness: NEW API script, targets
`dev-no-templates`'s pyfc (no S_model/B_model; pdf_components list for
unbinned; compute_rates_func closes over fixed template arrays).

Must be run with the `dev-no-templates` worktree/checkout root passed via
--pyfc-root, verified explicitly below (see xbranch_old_api.py's docstring
for why this is checked rather than assumed).

Produces scenario_a.npz and scenario_c.npz (for cross-branch diffing
against xbranch_old_api.py's outputs via comparator.py), plus runs
scenario (b)'s same-branch flatten-invariance check and scenario (d)'s
hand-computed-reference check directly (both raise AssertionError on
failure, so a non-zero exit code alone signals a problem).
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--pyfc-root", required=True, help="Root of the dev-no-templates checkout containing pyfc/")
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
print(f"[xbranch-new] pyfc.__file__ = {resolved}")
if not resolved.startswith(expected_root):
    raise RuntimeError(
        f"pyfc resolved from '{resolved}', which is NOT under the requested "
        f"--pyfc-root '{expected_root}' -- a different pyfc (stale pip install?) "
        f"is shadowing the worktree checkout. Refusing to trust these numbers."
    )

from pyfc.orchestrator import compute_fc_intervals
from pyfc.unbinned import calc_nll_unbinned

os.makedirs(args.output_dir, exist_ok=True)


# --- Scenario (a): binned, 1D data, 3-parameter model ---
@njit(fastmath=True, nogil=True)
def compute_rates_binned_new(params, S_sigma2, B_sigma2):
    mu = params[0] * sc.S_TEMPLATE_FLAT + params[1] + params[2] * sc.B_TEMPLATE_FLAT
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
        data=data, grids=sc.GRIDS_3PARAM,
        compute_rates_func=compute_rates_binned_new,
        cl=sc.CL, n_toys=sc.N_TOYS, strategy="scipy", num_cores=sc.NUM_CORES, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="binned", S_sigma2=s_sigma2, B_sigma2=b_sigma2,
        compute_1D_intervals=True, compute_2D_intervals=False,
        output_file="scenario_a", save_directory=args.output_dir,
    )
    print("[xbranch-new] scenario (a) done -> scenario_a.npz")
    return data


# --- Scenario (b): same physical model, genuine 2D histogram -- must match
# scenario (a)'s new-branch result exactly (no old-branch equivalent, since
# `dev` doesn't support N-D data at all). ---
S_TEMPLATE_2D = sc.S_TEMPLATE_FLAT.reshape(sc.BINNED_2D_SHAPE)
B_TEMPLATE_2D = sc.B_TEMPLATE_FLAT.reshape(sc.BINNED_2D_SHAPE)


@njit(fastmath=True, nogil=True)
def compute_rates_binned_2d(params, S_sigma2, B_sigma2):
    mu = params[0] * S_TEMPLATE_2D + params[1] + params[2] * B_TEMPLATE_2D
    sigma2 = (params[0] ** 2) * S_sigma2 + (params[2] ** 2) * B_sigma2
    return mu, sigma2


def run_scenario_b(data_flat):
    data_2d = data_flat.reshape(sc.BINNED_2D_SHAPE)
    s_sigma2 = np.zeros_like(data_2d)
    b_sigma2 = np.zeros_like(data_2d)

    np.random.seed(sc.RNG_SEED)
    results_2d, _ = compute_fc_intervals(
        data=data_2d, grids=sc.GRIDS_3PARAM,
        compute_rates_func=compute_rates_binned_2d,
        cl=sc.CL, n_toys=sc.N_TOYS, strategy="scipy", num_cores=sc.NUM_CORES, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="binned", S_sigma2=s_sigma2, B_sigma2=b_sigma2,
        compute_1D_intervals=True, compute_2D_intervals=False,
        output_file="scenario_b", save_directory=args.output_dir,
    )

    scenario_a_path = os.path.join(args.output_dir, "scenario_a.npz")
    ref = np.load(scenario_a_path, allow_pickle=True)
    assert np.isclose(results_2d["data_uncond_nll"], float(ref["data_uncond_nll"])), (
        f"scenario (b) same-branch invariant FAILED: 2D data_uncond_nll="
        f"{results_2d['data_uncond_nll']} != 1D-flattened data_uncond_nll={float(ref['data_uncond_nll'])}"
    )
    for p_idx in range(3):
        key = f"1d_t_data_p{p_idx+1}"
        assert np.allclose(results_2d[key], ref[key], equal_nan=True), (
            f"scenario (b) same-branch invariant FAILED for {key}"
        )
    print("[xbranch-new] scenario (b) same-branch invariant: MATCH (2D histogram == 1D-flattened)")


# --- Scenario (c): unbinned, 2 pdf_components (signal + background) ---
def s_pdf(x):
    from scipy.stats import norm
    return norm.pdf(x, loc=sc.S_PDF_LOC, scale=sc.S_PDF_SCALE)


def b_pdf(x):
    from scipy.stats import expon
    return expon.pdf(x, scale=sc.B_PDF_SCALE)


def compute_rates_unbinned_new(params, probs):
    s_probs, b_probs = probs[0], probs[1]
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
        data=events, grids=sc.GRIDS_2PARAM,
        compute_rates_func=compute_rates_unbinned_new,
        generate_toy_func=generate_toy_unbinned,
        pdf_components=[s_pdf, b_pdf],
        cl=sc.CL, n_toys=sc.N_TOYS, strategy="scipy", num_cores=sc.NUM_CORES, verbose=0,
        sparsify_grid=False, warm_start=False,
        likelihood_type="unbinned", S_mc_pool=s_mc_pool, B_mc_pool=b_mc_pool,
        compute_1D_intervals=True, compute_2D_intervals=False,
        output_file="scenario_c", save_directory=args.output_dir,
    )
    print("[xbranch-new] scenario (c) done -> scenario_c.npz")


# --- Scenario (d): unbinned, 3 pdf_components -- new-branch-only
# capability (impossible to express under the old S_model/B_model pair).
# Validated against an independently-hand-computed reference NLL rather
# than a cross-branch diff, since `dev` has no equivalent. ---
def m_pdf(x):
    from scipy.stats import expon
    return expon.pdf(x, scale=1.5)


def compute_rates_3c(params, probs):
    total = 0.0
    p_events = np.zeros_like(probs[0])
    for i, p in enumerate(probs):
        total += params[i]
        p_events = p_events + params[i] * p
    return total, p_events


def run_scenario_d():
    from scipy.stats import norm
    events = sc.UNBINNED_EVENTS_3C
    pdf_components = [lambda x: norm.pdf(x, loc=1.0, scale=1.0), m_pdf, lambda x: np.full_like(x, 0.1)]
    probs = [pdf(events) for pdf in pdf_components]

    all_ok = True
    for params in sc.TEST_PARAMS_3C:
        nll = calc_nll_unbinned(params, len(events), probs, compute_rates_3c)

        # Independently hand-computed reference: recompute expected_total and
        # p_events directly (same formula, evaluated outside calc_nll_unbinned)
        # and apply the textbook EUML NLL expression by hand.
        expected_total_ref = float(np.sum(params))
        p_events_ref = np.zeros_like(probs[0])
        for i, p in enumerate(probs):
            p_events_ref = p_events_ref + params[i] * p
        reference_nll = expected_total_ref - np.sum(np.log(p_events_ref))

        ok = np.isclose(nll, reference_nll)
        all_ok = all_ok and ok
        print(f"[xbranch-new] scenario (d) params={params}: calc_nll_unbinned={nll:.6f}, "
              f"reference={reference_nll:.6f}, match={ok}")

    assert all_ok, "scenario (d) 3-component hand-computed reference check FAILED"
    print("[xbranch-new] scenario (d): MATCH (3-component NLL agrees with hand-computed reference)")


if __name__ == "__main__":
    data_flat = run_scenario_a()
    run_scenario_b(data_flat)
    run_scenario_c()
    run_scenario_d()
