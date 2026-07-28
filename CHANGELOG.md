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
- **`fc_results.json`'s 1D `interval_bounds` schema changed from a flat
  `[lo, hi]` pair to a list of `[lo, hi]` pairs, one per disconnected
  accepted interval.** Previously, `_save_fc_json` computed a single
  `[min(accepted), max(accepted)]` span across every accepted grid point.
  Under the Feldman-Cousins unified construction, the accepted region for
  a parameter can in principle be genuinely disconnected near certain
  physical boundaries (e.g. a non-monotonic rate function crossing the
  data at more than one point); a flat min/max span silently merges those
  separate intervals into one, including whatever rejected gap sits
  between them -- misrepresenting the actual confidence region. A new
  `_find_contiguous_intervals` helper in `orchestrator.py` now scans the
  scan-ordered `test_points`/`accepted` arrays for every maximal
  contiguous accepted run and reports each as its own `[lo, hi]` pair.
  **This always applies, even in the single-interval case** (still a
  one-element list, `[[lo, hi]]`, not flattened) -- a consistent schema
  regardless of how many disjoint pieces the accepted region has.
  **Migration:** any downstream code reading `interval_bounds` as
  `[lo, hi]` directly (e.g. `lo, hi = interval_bounds`) must be updated to
  iterate the list of pairs instead. 2D intervals are unaffected -- no
  `interval_bounds`-style precomputed field existed there before or now;
  `2d_intervals` stores the full accepted boolean grid, from which any
  region shape can already be reconstructed directly. `fc_results.npz`
  and `generate_corner_plot` are both unaffected -- neither ever consumed
  `interval_bounds` (the `.npz` stores raw per-point arrays, and plotting
  shades directly from those, not from the JSON's precomputed field).
  Documented in README.md's "Stored Results & Custom Plotting" section
  and `docs/source/outputs.rst`, with a worked example added to
  `pyfc_algorithmic_features_tutorial.ipynb`. New tests for
  `_find_contiguous_intervals` and the JSON-writing logic in
  `tests/test_disconnected_intervals.py`, including a genuine (not
  hand-mocked) end-to-end disconnected-region example.

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
  practically identical to before (matches to within 1e-12 relative/absolute
  tolerance; see `tests/test_smoothing.py`).
- **Example notebooks renamed with a `01`-`06` numeric prefix reflecting
  suggested reading order**, and a new "Tutorial Notebooks" section added
  to README.md (plus a mirroring `docs/source/tutorials.rst`, linked from
  the Sphinx toctree) describing what each one covers and when to reach
  for it. Old name -> new name: `pyfc_quickstart_tutorial.ipynb` ->
  `01_pyfc_quickstart_tutorial.ipynb`, `pyfc_high_dimensional_tutorial.ipynb`
  -> `02_pyfc_high_dimensional_tutorial.ipynb`,
  `pyfc_non_contiguous_data_tutorial.ipynb` ->
  `03_pyfc_non_contiguous_data_tutorial.ipynb`,
  `pyfc_algorithmic_features_tutorial.ipynb` ->
  `04_pyfc_algorithmic_features_tutorial.ipynb`,
  `pyfc_strategy_comparison_tutorial.ipynb` ->
  `05_pyfc_strategy_comparison_tutorial.ipynb`,
  `pyfc_joint_constraints_tutorial.ipynb` ->
  `06_pyfc_joint_constraints_tutorial.ipynb`. **Migration:** update any
  bookmarks, scripts, or CI steps that reference the old filenames
  directly (nothing in this repository's own code or CI did). Each
  notebook's own "Next steps" cross-references and every mention in
  README.md/`docs/source/installation.rst` were updated to match; past
  CHANGELOG entries below that predate this rename still use the old
  names, as a historical record of what was true when they were written.

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
- **`strategy="ultranest"`/`"hybrid"` (2D intervals) crashed with
  `IndexError: too many indices for array: array is 1-dimensional, but 2
  were indexed`**, raised inside `ultranest.ReactiveNestedSampler`'s own
  internal `_find_strategy` method (confirmed present through at least
  UltraNest v4.5.0 and the current GitHub master, so this is not fixed by
  upgrading). An internal random subsample can occasionally select too
  few saved iterations, producing a degenerate 1D array where UltraNest's
  own code expects 2D. Since this stems from UltraNest's own internal
  exploration randomness rather than anything about the model being
  fitted, `optimizers.py`'s three `*_ultranest` fit functions now retry
  with a fresh sampler instance on this specific error (new
  `_run_ultranest_with_retry` helper, mirroring `_minimize_with_restarts`'s
  existing retry-on-failure pattern for the scipy path; covered by new
  tests in `tests/test_ultranest_retry.py`). Separately,
  `pyfc_strategy_comparison_tutorial.ipynb`'s grid/`n_toys` were scaled
  down, since its original size didn't finish even in 30 minutes with
  `ultranest` actually installed -- as far as could be determined, this
  strategy combination had never been exercised end-to-end before.
- **`sparsify_grid` defaulted to `True` in `compute_fc_intervals`'s own
  Python keyword default, while `config.py`'s CLI/JSON default and every
  doc already documented `False` as *the* default.** Anyone calling
  `compute_fc_intervals` directly without an explicit `sparsify_grid`
  argument silently got the (buggy, see below) `True` behavior. Unified
  to `False` everywhere, including `generate_config.py`'s interactive
  wizard, which independently defaulted to `"y"`.
- **`sparsify_grid=True`'s 2D boundary-refinement pass was a complete
  no-op.** `eval_2d_point`'s memoization guard checked `2d_t_critical` for
  `NaN`-ness to decide whether a cell had already been evaluated, but the
  coarse-to-fine `RectBivariateSpline` interpolation step fills in
  `2d_t_critical` for *every* cell (real or not) before the refinement
  pass runs -- so the guard always fired, and every non-coarse cell's
  `2d_t_data` stayed `NaN` forever. Since `accepted = t_data <= t_critical`
  is always `False` against `NaN`, this silently excluded the vast
  majority of the grid from the accepted region regardless of its true
  classification (confirmed by direct reproduction: 220/256 cells
  force-excluded on a 16x16 grid, accepted count 8/256 instead of the
  true, much larger region). Fixed by keying the guard off `2d_t_data`
  instead, which is only ever set by an actual conditional fit and
  untouched by interpolation. New regression tests in
  `tests/test_sparsify_grid.py`. **Known remaining limitation:** boundary
  refinement is still a single, non-iterative pass with a 1-cell-wide
  halo around ~5-per-axis coarse nodes, so for grids much larger than
  ~20x20 per axis (the scale this feature is meant for) it still
  under-counts the true accepted region -- see the caveat now documented
  in `compute_fc_intervals`'s docstring and README's "Contour Edge
  Tracing" section. Tracked as follow-up work.
- **`compute_fc_intervals`'s own Python keyword defaults disagreed with
  README.md's documented defaults for six parameters**: `n_toys` (`2000`
  vs. documented `500`), `save_log` (`False` vs. `True`), `smooth_1d`/
  `smooth_2d` (`False`/`False` vs. `True`/`True`), and `cl` (resolved to
  `[0.90]` when omitted, vs. documented `[0.68, 0.90]`). Anyone calling
  `compute_fc_intervals` directly without a config dict -- again, the
  README's own Quick Start Guide style -- silently got different
  behavior than what the docs describe (e.g. 4x fewer toys, no persistent
  log, unsmoothed plots, only a single CL). Unified all six to match
  README.md. `num_cores`, `output_file`, and `param_names` are
  deliberately left as their existing `None`-sentinel defaults (auto-
  detect hardware threads, skip writing a results file, and auto-generate
  `param{i}` names matching the model's actual `n_params`, respectively)
  rather than hardcoded to README's example-config values, since those
  three sentinels encode real auto-detect/opt-out behavior that a literal
  copy would remove or, for `param_names`, actively break on any model
  where `n_params != 3`; this is now spelled out explicitly in the
  docstring for each. Five example notebooks that omitted `save_log`/
  `smooth_1d`/`smooth_2d` (relying on the old implicit defaults) were
  re-executed to refresh their embedded (now-smoothed) plot images:
  `pyfc_algorithmic_features_tutorial.ipynb`,
  `pyfc_high_dimensional_tutorial.ipynb`,
  `pyfc_joint_constraints_tutorial.ipynb`,
  `pyfc_non_contiguous_data_tutorial.ipynb`,
  `pyfc_quickstart_tutorial.ipynb`, and
  `pyfc_strategy_comparison_tutorial.ipynb`.
- **A razor-thin (~1e-12-wide) NLL discontinuity in the unphysical-region
  smoothing** (`binned.calc_nll`, `unbinned.calc_nll_unbinned`), left over
  from the smoothing fix documented above. For a bin/event with real
  observed data, the physical branch's Poisson term correctly diverges to
  `+inf` as `mu_i`/`p_events` -> 0+, but the unphysical branch's base term
  is evaluated at a *fixed* floor (`mu_floor`/`p_floor` = `1e-12`) rather
  than at the actual value, and was triggered only at `mu_i <= 0`/
  `p_events <= 0`. That left the tiny physical sliver `(0, 1e-12)` on the
  diverging physical branch, so crossing from `+epsilon` to `-epsilon`
  made the NLL *drop* right at the boundary instead of continuing to
  climb -- the opposite of the smoothing fix's whole purpose. Very
  unlikely to be hit in practice (no realistic optimizer step lands
  within `1e-12` of exactly zero), but fixed defensively anyway: the
  trigger is now `mu_i <= mu_floor` / `p_events <= p_floor`, so the
  transition point is single and consistent, approached from both
  directions. `mu_floor`/`p_floor` are astronomically smaller than any
  physically meaningful value, so this has no practical effect on real
  fits (verified: existing backward-compatibility tests for the
  physical-branch formula, `test_binned_calc_nll_identical_for_physical_bins`/
  `test_unbinned_calc_nll_identical_for_physical_events`, are unaffected).
  New fine-resolution monotonicity-sweep tests in `tests/test_smoothing.py`
  covering the exact old discontinuity point.
- **README.md's and `docs/source/methodology.rst`'s displayed finite-MC
  correction formula did not match `calc_nll`'s actual implementation at
  all** (not just a missing citation): the docs' formula had no `lgamma`/
  `Gamma` terms whatsoever, while the code implements an exact Negative
  Binomial log-likelihood built from `lgamma`. Numerically evaluating both
  at the same `(mu, sigma2, n)` gives substantially different values --
  the docs' formula was simply wrong, not an approximation of the code.
  Checked the code's `alpha = mu^2/sigma^2 + 1` (previously flagged as
  possibly a bug, since a naive mean-and-variance-matching derivation
  gives `alpha = mu^2/sigma^2` with no `+1`) against Argüelles, Schneider
  & Yuan, "A binned likelihood for stochastic models", JHEP 06 (2019) 030
  [arXiv:1901.04645]: `calc_nll`'s formula is *exactly* the paper's `L_Eff`
  (Eq. 3.16, their main, recommended result) -- the `+1` is a deliberate,
  load-bearing part of that specific (best-coverage) parameterization, not
  a bug, and no code change was needed. New hand-computed-reference tests
  in `tests/test_finite_mc_likelihood.py` verify `calc_nll`'s finite-MC
  branch against the paper's Eq. 3.16 directly. Added an explicit
  citation and provenance note to `calc_nll`'s docstring (this paper was
  already cited
  in `docs/source/methodology.rst`/`docs/source/refs.bib` and README.md's
  own "Methodology References" section, just not from the code itself),
  and corrected both docs' displayed formula to match the paper/code
  exactly. Also corrected README.md's feature-list description, which
  called this the "Beeston-Barlow technique" -- a different, *profiled*
  (not marginalized) likelihood that the paper itself distinguishes from
  `L_Eff` (and which this codebase does not implement).
- **`adaptive_toys` and `toy_batch_size` were documented, threaded through
  `config.py`/`generate_config.py`'s CLI/JSON/wizard machinery, and part
  of `compute_fc_intervals`'s own signature -- but neither was ever read
  anywhere to make an actual decision.** README's own OOM-troubleshooting
  advice ("reduce `toy_batch_size` from 200 to 50") would have had zero
  effect. Made both real, for `strategy` in `"scipy"`/`"ultranest"`/
  `"hybrid"` (`toys.generate_and_fit_toys_python`); no effect for
  `strategy="grid"`, whose toy generators are a single vectorized `numba`
  `@njit(parallel=True)` call per point rather than a pool of per-toy
  tasks, not a natural fit for either batching or a sequential early-exit
  check.
  - `toy_batch_size`: toys are now submitted/collected from the
    `ThreadPoolExecutor`/`ProcessPoolExecutor` in batches of this size
    instead of one `executor.map` call for all `n_toys`, bounding how
    many toy datasets / in-flight results are alive at once (matters most
    for `likelihood_type="unbinned"`, matching README's OOM scenario).
    Same total `n_toys` are always generated -- purely a dispatch/memory
    change, zero effect on results (proved directly: for binned toys, the
    same seed produces bit-for-bit identical output regardless of
    `toy_batch_size`, since the underlying `np.random.poisson` call that
    generates the toy data happens once up front either way).
  - `adaptive_toys`: after each `toy_batch_size` batch, computes the
    running fraction of toys with `t_toy >= t_data` (a p-value estimate
    for that point's accept/reject decision) and a strict 99.9% Wilson
    score confidence interval around it. If that interval lies entirely
    on one side of the target significance `alpha = 1 - CL` (using the
    *largest* requested `cl` when several are given, so stopping there
    guarantees every less-stringent `cl`'s decision is also settled),
    the verdict cannot plausibly flip with more toys and generation stops
    early for that point. Never considered before 100 toys
    (`toys.ADAPTIVE_MIN_TOYS`), regardless of how extreme the running
    estimate looks -- small-sample binomial confidence intervals are
    unreliable. Meaningfully saves time only for points far from the
    accept/reject boundary; points near it correctly run the full
    `n_toys`. `compute_fc_intervals`'s own quantile-index computation
    (`t_stats[int(cl * n_toys)]`) now uses the *actual* number of toys
    generated for that point instead of the original `n_toys`, since
    `adaptive_toys` can make that shorter.
  - **Validated, not just implemented** (per the risk this fix was
    flagged with -- a wrong stopping rule would silently bias confidence
    intervals, worse than the previous do-nothing state): new tests in
    `tests/test_adaptive_toys.py` cover the Wilson-interval/stopping-
    decision primitives directly, confirm well-inside/well-outside points
    stop at exactly the minimum toy count while a point near the true
    critical value runs the full `n_toys`, and -- the real test of "did
    this introduce bias" -- confirm `adaptive_toys=True` reaches the
    *same* accept/reject verdict as `adaptive_toys=False` with the same
    `n_toys` cap, both for isolated toy-generation calls across many
    independent trials and for a full end-to-end `compute_fc_intervals`
    run. README's `adaptive_toys`/`toy_batch_size` table rows, the OOM-
    troubleshooting paragraph, and `docs/source/configuration.rst`'s
    table are updated to describe what actually happens now, plus a new
    README "Adaptive Toy Generation" section (mirroring the existing
    "Contour Edge Tracing" section's depth).
- **`generate_config.py`'s interactive wizard still offered four default
  answers that disagreed with `compute_fc_intervals`'s own (already
  README-aligned) defaults**, missed when those defaults were unified
  earlier in this release: `n_toys` (wizard default `2000` vs. `500`),
  `smooth_1d`/`smooth_2d` (wizard default `n` vs. `True`), and `save_log`
  (wizard default `n` vs. `True`). Anyone who ran the wizard and accepted
  every default (the documented, intended workflow) would get a config
  that silently diverged from what README.md and `pyfc/config.py`'s own
  base config dict both call "the" default. Unified all four; the
  wizard's defaults now match everywhere else exactly, so pressing Enter
  on every question reproduces `compute_fc_intervals`'s own defaults.

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
