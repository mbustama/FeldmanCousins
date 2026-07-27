# Changelog

All notable changes to PyFC are documented in this file.

## [Unreleased]

Hardens the optimizer/likelihood layer against joint (non-box) parameter
constraints, informed by a concrete failure case from a downstream project
that needed to fit flavor fractions subject to `f_e + f_mu <= 1`.

### Changed

- **`calc_nll` (binned.py) and `calc_nll_unbinned` (unbinned.py): behavior
  change in the unphysical (`mu_i <= 0` / `p_events[k] <= 0`) region.**
  Previously, both functions returned a flat constant penalty (`1e10`) the
  instant a bin/event's expected rate went non-positive -- `binned.py`'s
  version was additionally an early return that discarded whatever NLL had
  already accumulated from other, physical bins in the same call. Because
  `scipy.optimize.minimize(..., method='L-BFGS-B')` estimates gradients by
  finite differences, that flat region reads as "gradient ~ 0, already
  converged," and could permanently trap the optimizer if its starting
  guess (the bounds midpoint, by default) landed there.
  Both functions now instead add a smooth, continuously-varying
  contribution for the unphysical region: a continuous extension of the
  real NLL term at a tiny positive floor, plus a quadratic barrier that
  grows with how far into the unphysical region the point actually is.
  **NLL values in the previously-unphysical region will differ from prior
  runs** (they were an arbitrary constant before, so this is not a
  regression, but anyone diffing raw NLL outputs against pre-fix runs where
  the fit passed through an unphysical region should be aware of this.) For
  all physical inputs (`mu_i > 0` / all `p_events[k] > 0`), output is
  byte-for-byte identical to before.

### Added

- **`bounds_func` hook** on `unconditional_fit_scipy`/`unconditional_fit_ultranest`
  and `conditional_fit_1d/2d_scipy`/`conditional_fit_1d/2d_ultranest`
  (optimizers.py), plus a `bounds_func=None` kwarg on
  `compute_fc_intervals` (orchestrator.py): lets the box bounds of the
  profiled (free) parameters depend on whatever parameter(s) the scan
  currently has fixed, so a joint constraint like `f_mu <= 1 - f_e_test`
  can be expressed without a user-side penalty function. Default `None`
  preserves prior behavior exactly.
- **`constraints` support** on the same scipy/ultranest fit functions, plus
  `constraints=None`/`scipy_method=None` kwargs on `compute_fc_intervals`:
  accepts a list of `scipy.optimize.LinearConstraint`/`NonlinearConstraint`
  objects (expressed in the full parameter-vector space) for constraints
  among multiple *simultaneously-free* nuisance parameters, which
  `bounds_func` cannot express. Automatically switches the scipy method
  from `L-BFGS-B` to `SLSQP` when constraints are supplied (overridable via
  `scipy_method`, e.g. `"trust-constr"`); a new `--scipy_method` CLI/JSON
  config option is also available. A shared `_project_linear_constraint`
  helper projects full-space constraints down to the free-parameter
  subspace for the conditional fit functions. Default `None`/empty
  preserves prior behavior and method exactly.
- **Optimizer restarts and `res.success` checking** (optimizers.py's new
  `_minimize_with_restarts` helper, used by all three scipy fit
  functions): `res.success`/`res.status` from `scipy.optimize.minimize`
  were previously read nowhere in the codebase -- a silently non-converged
  fit's result was used exactly like a converged one. Now, a non-converged
  result triggers one cheap perturbed-restart retry and a warning if that
  also fails. A new `n_restarts` parameter (default `1`, unchanged
  behavior aside from the check above) allows trying multiple starting
  points and keeping the best-converged one.
- **Neighbor warm-starting for the data fit** (orchestrator.py): Phase 1's
  1D scan and Phase 2's 2D scan previously started every single grid
  point's data fit fresh from the bounds midpoint, with zero information
  sharing between adjacent, already-solved grid points, even though these
  loops already iterate the grid in order. (MC toy fits were already
  well-seeded from the conditional MLE and are unaffected.) A new
  `neighbor_seeding=True` kwarg on `compute_fc_intervals` seeds each grid
  point's data fit from an adjacent, already-evaluated point's profiled
  parameters; whenever used, `n_restarts` is forced to at least 2 (the
  neighbor-seeded start plus a fresh bounds-midpoint start) so a bad
  neighbor optimum can't silently cascade forward. Set `False` to restore
  the always-fresh-start behavior.
- README section "Handling Joint/Simplex-Constrained Parameters" (also
  mirrored, condensed, in `docs/source/methodology.rst`): a worked
  end-to-end tutorial for the `f_e + f_mu <= 1` example, covering both
  `bounds_func` and `constraints` and when to use each.

## [0.9.2] and earlier

See git history.
