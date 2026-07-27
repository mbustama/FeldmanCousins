# Changelog

<!--
This file is mirrored by hand into docs/source/changelog.rst, and CI
(.github/workflows/changelog-sync.yml) checks the two stay structurally in
sync on every change. See scripts/check_changelog_sync.py's module
docstring for exactly what "in sync" means (and doesn't) before editing
either file.
-->

All notable changes to PyFC are documented in this file.

## [0.10.0]

Hardens the optimizer/likelihood layer against joint (non-box) parameter
constraints, informed by a concrete failure case from a downstream project
that needed to fit flavor fractions subject to `f_e + f_mu <= 1`. Also
removes the `S_model`/`B_model` templating mechanism in favor of a more
general `pdf_components` list, and adds support for N-dimensional binned
data.

### BREAKING CHANGES

- **`S_model`/`B_model` removed entirely from `compute_fc_intervals` and
  every optimizer/toy-generation function** (`unconditional_fit_scipy`,
  `conditional_fit_1d_scipy`, `conditional_fit_2d_scipy`,
  `unconditional_fit_ultranest`, `conditional_fit_1d_ultranest`,
  `conditional_fit_2d_ultranest`, `generate_and_fit_toys_python`, and the
  binned/unbinned grid-search and toy-generation functions in `binned.py`/
  `unbinned.py`). An audit established that for binned models,
  `S_model`/`B_model` were pure pass-through into `compute_rates_func` --
  `calc_nll` never read them itself. For unbinned models they were a
  hardcoded signal/background *pair*, which was itself an unnecessary
  rigidity on top of being pass-through. `compute_fc_intervals` now takes
  only `data` and `grids` as required positional arguments (down from 4:
  `data, S_model, B_model, grids`).
  **Migration hazard:** because `grids` moves from position 4 to position
  2, any caller still using positional arguments will silently pass the
  wrong value into the wrong parameter instead of raising an error.
  **Convert all calls to fully keyword-form**
  (`compute_fc_intervals(data=..., grids=..., compute_rates_func=..., ...)`).
- **New `pdf_components` mechanism replaces `S_model`/`B_model` for
  unbinned models.** `pdf_components` is a `list[callable]` of any length
  (no longer hardcoded to exactly 2), passed to `compute_fc_intervals` and
  to every optimizer/toy function. Each is evaluated exactly once per fit
  as `pdf(data)` and the resulting arrays are reused across every
  subsequent NLL evaluation in that fit -- the same caching behavior as
  before, generalized beyond a fixed signal/background pair. Required
  (and validated) when `likelihood_type="unbinned"`; ignored with a
  warning for `"binned"`.
- **`compute_rates_func` signature changes for both likelihood types:**
  - Binned: `(params, S_template, B_template, S_sigma2, B_sigma2)` becomes
    `(params, S_sumw2, B_sumw2)`. Fixed template arrays that used to be
    passed as arguments must now be referenced via closure (module-level
    constant or nested-function capture) instead -- this is already
    numba-`@njit`-compatible and is the pattern used throughout
    `examples/pyfc_joint_constraints_tutorial.ipynb`'s own templates.
  - Unbinned: `(params, s_probs, b_probs)` becomes `(params, probs)`,
    where `probs` is a `list[np.ndarray]`, one entry per
    `pdf_components[i]`, in the same order.
- **`S_sigma2`/`B_sigma2` renamed to `S_sumw2`/`B_sumw2`** everywhere they
  appear (`compute_fc_intervals`, every optimizer/toy-generation function,
  `calc_nll`, and the binned grid-search functions). These arrays are the
  per-bin sum of squared MC weights (`sum(w_i^2)`) -- the standard
  variance estimator for a weighted MC sample -- used by the finite-MC
  (Poisson-Gamma) correction. The `S_`/`B_` prefixes now dangle less
  meaningfully without `S_model`/`B_model` as their counterpart, but the
  `sigma2` suffix was also imprecise (it names a *derived* quantity
  computed *by* `compute_rates_func`, not what these arguments actually
  hold): `sumw2` is unambiguous and matches the term HEP users already
  know from `TH1::Sumw2()`-style per-bin weight-squared tracking.
- **`data`, `mu`, and `sigma2` now support arbitrary N-dimensional shapes
  for binned models**, not just flat 1D vectors -- a genuine 2D
  `(E, cos_theta)` histogram (or any other shape) can be evaluated
  directly, with no need to flatten it yourself first. `calc_nll` flattens
  internally before summing, so the NLL is identical regardless of how the
  bins are laid out spatially. `grids` (the parameter *scan*, as opposed to
  the *data*) is unaffected -- it was already N-parameter-general and
  remains a list of 1D per-parameter arrays. Unbinned events already
  supported N-D feature vectors (`data.shape == (n_events, n_features)`);
  this is now documented as the indexing convention, along with a gotcha:
  bootstrapping an N-D MC pool requires index-based resampling
  (`idx = np.random.choice(len(mc_pool), size=n, replace=True); toy_events
  = mc_pool[idx]`), since `np.random.choice` applied directly to an N-D
  pool silently samples along the wrong axis.
- **Migration:** see the rewritten Quick Start Guide in `README.md` and
  `docs/source/quickstart.rst` for both binned and unbinned examples under
  the new signatures.

### Changed

- **`examples/pyfc_tutorial.ipynb` replaced by four focused notebooks:**
  `pyfc_quickstart_tutorial.ipynb` (binned + unbinned, the runnable
  counterpart to the README Quick Start Guide), `pyfc_high_dimensional_
  tutorial.ipynb` (scaling to 5 and 10 parameters, including the
  combinatorial cost of 2D contours), `pyfc_algorithmic_features_
  tutorial.ipynb` (`sparsify_grid`/`smooth_1d`/`smooth_2d`/
  `use_finite_mc_correction_binned`, each demonstrated on vs. off), and
  `pyfc_strategy_comparison_tutorial.ipynb` (`"grid"`/`"scipy"`/`"hybrid"`
  timing comparison, plus building a custom Matplotlib contour directly
  from PyFC's saved `.json`/`.npz` output). The old monolithic notebook's
  heaviest cells (`n_toys=100`, dense 40-point grids, `sparsify_grid=False`,
  2D contours) took 20+ minutes *each* to execute, which made it
  impractical to keep re-running end-to-end; the four replacements use
  small grids/`n_toys` (in the style already established by
  `pyfc_joint_constraints_tutorial.ipynb`) so each runs in well under a
  minute, and each carries more narrative markdown/comments explaining
  *why* a cell does what it does, not just what it does.
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

### Fixed

- **`calc_nll` (binned.py) and the two grid-strategy toy generators
  (`generate_and_fit_toys_grid_1d`/`_2d`) crashed on non-contiguous N-D
  binned arrays** -- e.g. a transposed histogram (`data.T`) -- with a
  low-level `numba` `NotImplementedError: incompatible shape for array`
  instead of working correctly. `numba`'s `nopython`-mode `.reshape(-1)`
  only accepts C-contiguous input, and this was never triggered during
  the original N-D-support work, so the guard was never added. Since the
  entire point of N-D binned support is letting users lay out bins
  however is natural for them, this is a plausible real-world trigger,
  not a contrived edge case: anyone who transposes their histogram
  before calling `compute_fc_intervals` hit this crash. Fixed by copying
  to a contiguous layout first with `np.ascontiguousarray(...)`, a no-op
  (zero cost, no copy) on the already-contiguous common path. Covered by
  new regression tests in `tests/test_nd_binned.py`, both at the
  `calc_nll` level and end-to-end through `compute_fc_intervals` under
  both the `"scipy"` and `"grid"` toy-generation strategies.
- **`save_directory`'s default was inconsistent across the codebase.**
  `compute_fc_intervals`'s own Python keyword default (`"fc_output"`) and
  `generate_config.py`'s interactive-wizard default disagreed with
  `pyfc/config.py`'s CLI/JSON-config default (`"output/example_fc_output"`),
  which every doc and example already assumed was *the* default. Calling
  `compute_fc_intervals` directly without a config dict or an explicit
  `save_directory` -- the style the README's own Quick Start Guide uses
  for its inline code snippets -- silently wrote output somewhere
  different from what the docs describe. Unified all three to
  `"output/example_fc_output"`.

### Added

- **`generate_corner_plot` now importable as `from pyfc import generate_corner_plot`**
  (previously only reachable as `from pyfc.plotting import generate_corner_plot`),
  matching `compute_fc_intervals`'s existing top-level export. README.md,
  `docs/source/quickstart.rst`, and all example notebooks now import both
  functions this way -- code copied from the docs/tutorials no longer needs
  to know PyFC's internal module layout (`orchestrator.py`/`plotting.py`).
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
  mirrored, with the same worked code examples, in
  `docs/source/methodology.rst`): a worked end-to-end tutorial for the
  `f_e + f_mu <= 1` example, covering both `bounds_func` and `constraints`
  and when to use each. `docs/source/configuration.rst` and `quickstart.rst`
  cross-reference it, and a new `docs/source/changelog.rst` page (wired
  into the Sphinx toctree) mirrors this file.
- New example notebook, `examples/pyfc_joint_constraints_tutorial.ipynb`:
  a hands-on walkthrough of both mechanisms using a toy 3-parameter
  neutrino flavor-fraction model (`f_e`, `f_mu`, `norm`) constrained to the
  standard 2-simplex. Reproduces the flat-gradient plateau trap with a
  hard-cutoff rate function and no constraint handling, then fixes it with
  `bounds_func`; demonstrates `constraints` for the case `bounds_func`
  cannot express (two simultaneously-free parameters); runs a real
  `compute_fc_intervals` pipeline and plots the resulting 2D confidence
  region confined to the simplex; and includes a bonus
  `NonlinearConstraint` (unit-disk) example.
- `tests/test_nd_binned.py`: flatten-invariance checks (a genuine 2D
  histogram and its 1D-flattened equivalent must give byte-identical NLL)
  plus a full `compute_fc_intervals` smoke test on 2D-shaped data.
- `tests/test_pdf_components.py`: correctness checks for the new
  `pdf_components` mechanism with both 2 and 3+ components, including a
  hand-computed reference NLL for the 3-component case (no old-API
  equivalent to diff against).
- `xbranch_compare/`: a reusable cross-branch regression harness (kept in
  the repo for future breaking API changes, not a one-off validation) that
  empirically proves this refactor changed signatures only, never
  numerics, by running equivalent models against both `dev` (old API, via
  a temporary `git worktree`) and `dev-no-templates` (new API) and diffing
  the results.
- New example notebook, `examples/pyfc_non_contiguous_data_tutorial.ipynb`:
  demonstrates the non-contiguous-array fix above -- passes a transposed
  histogram straight into `compute_fc_intervals` and confirms the result
  matches a contiguous copy of the identical data, after first
  illustrating the underlying `numba`/`numpy` contiguity restriction in
  isolation.
- `.github/workflows/changelog-sync.yml` and `scripts/check_changelog_sync.py`:
  a CI check that fails the build if this file and
  `docs/source/changelog.rst` structurally drift out of sync (version/
  subsection headings, bullet counts, and each bullet's opening wording),
  since the two are hand-maintained in parallel and RST-only enhancements
  (admonitions, cross-references) mean a byte-for-byte diff or a generic
  Markdown-to-RST converter isn't viable. See the script's module
  docstring for exactly what is and isn't checked.

## [0.9.2] and earlier

See git history.
