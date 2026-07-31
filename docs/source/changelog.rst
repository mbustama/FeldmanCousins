Changelog
=========

.. This page is a hand-maintained RST mirror of CHANGELOG.md, and CI
   (.github/workflows/changelog-sync.yml) checks the two stay structurally
   in sync on every change. See scripts/check_changelog_sync.py's module
   docstring for exactly what "in sync" means (and doesn't) before editing
   either file.

All notable changes to PyFC are documented here. This page mirrors the
project's `CHANGELOG.md <https://github.com/mbustama/FeldmanCousins/blob/main/CHANGELOG.md>`_
at the repository root.

Unreleased
----------

Added
~~~~~
* **Test-coverage measurement.** ``pytest-cov`` joins the ``test`` and ``dev``
  extras, the settings live in ``[tool.coverage.run]`` in ``pyproject.toml`` so
  a local run measures exactly what CI measures, and a separate Coverage job in
  ``pytest.yml`` reports the figure on each run's summary page and uploads
  ``coverage.xml`` as an artifact. Branch coverage is enabled: a line-coverage
  number flatters this codebase, because the orchestrator is a dispatch machine
  whose failure modes are branches only ever taken one way -- ``strategy="grid"``
  vs ``"adaptive"``, a resumed checkpoint vs a fresh run, the
  disconnected-interval reporting path, the ``if ultranest is None`` guards --
  and line coverage counts all of those as covered the moment the function runs
  at all. The job reports rather than gates: there is deliberately no
  ``--cov-fail-under``, since a threshold invented before the first measurement
  either sits below the real figure and never fires, or above it and blocks
  unrelated work. It also carries a Codecov upload step gated on a
  ``CODECOV_TOKEN`` secret, skipped entirely when that secret is absent so the
  workflow stays safe to run in a fork; the secret now exists, so uploads are
  live, the coverage badge resolves, and Codecov comments per-PR diff coverage.
  See :doc:`installation` for local usage.
* **``scripts/run_coverage.sh``, which splits the suite around Numba's JIT.**
  ``coverage.py`` traces Python bytecode, and Numba executes none: it compiles
  ``@njit`` functions to machine code, so every line inside ``binned.py`` -- six
  compiled functions including ``calc_nll``, the NLL everything else is built on
  -- is reported as missed no matter how hard the suite hits it. Measured over
  the three test files that target ``calc_nll`` directly, ``binned.py`` reports
  **10%** with the JIT on and **90%** under ``NUMBA_DISABLE_JIT=1``; publishing
  the former would be actively misleading about the part of PyFC that most
  needs to be trustworthy. Disabling the JIT wholesale is not an option either,
  since ``strategy="grid"``'s ``@njit(parallel=True)`` toy generators degrade
  into plain Python loops and the suite then fails to finish in any reasonable
  time. The script therefore runs the pathological files with the JIT on,
  everything else with it off, and lets ``coverage`` combine the halves. The
  classification lives in the script rather than in the workflow so that there
  is exactly one place it can go stale, and its failure mode is benign: a newly
  added test file falls into the JIT-disabled half automatically, so it is
  measured correctly and at worst runs slowly, never silently skipped.
* **Structural and behavioural tests for the configuration layer**, closing what
  that first coverage run found: ``config.py`` sat at 10% and
  ``generate_config.py`` at 6%, meaning essentially nothing tested the CLI/JSON
  config layer -- a layer edited on almost every release, whose correctness had
  until now been established by running the CLI by hand. They take those two
  modules to 100% and 98% respectively, and the package as a whole from 66% to
  75%. The structural sweeps
  check that every analysis parameter lines up across all four places it is
  written down: the hardcoded defaults, the ``argparse`` flags, the interactive
  wizard's questions, and ``compute_fc_intervals``' own signature and defaults.
  They are self-discovering -- each derives its subjects from the code rather
  than from a hardcoded list -- so a parameter added later is swept without
  anyone remembering to extend a list, which is the only reason they would have
  caught the two real v0.10.0 bugs they are modeled on (``scipy_method``
  reaching ``parse_arguments`` but never being added to the wizard, and a wizard
  default disagreeing with the one ``compute_fc_intervals`` actually uses).
  Alongside them, the layer's stated ``Defaults -> JSON -> CLI`` precedence
  chain is now tested in both directions, as is ``get_input``'s rejection of a
  bad cast, a value outside ``choices``, and a failed validator. Every one of
  these tests was verified to fail against a deliberately broken version of the
  code before being kept.
* **Tests for the unbinned grid-search path**
  (``tests/test_unbinned_grid.py``), the second gap the measurement exposed.
  ``unbinned.py`` sat at 34%, and reading which lines were cold showed the
  shortfall was not scattered: four whole functions --
  ``conditional_fit_grid_unbinned_1d``, ``conditional_fit_grid_unbinned_2d``,
  ``generate_and_fit_toys_grid_unbinned_1d`` and
  ``generate_and_fit_toys_grid_unbinned_2d`` -- were executed by nothing at
  all, despite every one being imported and dispatched to by
  ``compute_fc_intervals``. Across the entire suite
  ``likelihood_type="unbinned"`` appeared exactly once, and that test used
  ``strategy="scipy"``, so the unbinned grid path shipped and was reachable by
  users while never being run. That combination is the one place in PyFC where
  profiling is exhaustive rather than numerical, which makes a silent error
  there especially hard to notice: a scan that quietly profiles over the wrong
  axis still returns plausible, finite, monotonic-looking numbers. The tests
  therefore compare against oracles computed independently in the test body --
  an explicit loop over the same grid calling ``calc_nll_unbinned`` directly --
  and against the invariants that define a profile likelihood ratio: the
  conditional NLL never undercuts the unconditional minimum, and the two
  coincide exactly at the best-fit point. The end-to-end run reconstructs the
  expected ``t_data`` array rather than asserting finiteness, because
  ``t_data`` is clamped at zero and a mis-wired dispatch would otherwise hide
  behind a valid-looking array of zeros. Also closes the unbinned twins of two
  guards ``binned.py`` has had tested since v0.10.0: the NaN per-event-density
  rejection and the quadratic barrier for unphysical densities. Takes
  ``unbinned.py`` from 34% to 100%; all 13 mutations tried against these tests
  were detected.

Changed
~~~~~~~
* **CI now runs on ``dev`` and ``dev-*`` branches, not only ``main``.** The
  previous filter meant a topic branch got no CI signal at all until a pull
  request was opened, which is the point at which a failure is most expensive to
  discover.
* **The Coverage job installs ``.[test,optimizers]``** while the Python-version
  matrix deliberately stays on ``.[test]`` alone. Without ``ultranest`` the
  three ``*_ultranest`` fit functions count as missed and the reported figure
  for ``optimizers.py`` understates its real coverage, which invites someone to
  go and "fix" coverage that is not actually missing; but that ``ultranest`` is
  optional is a promise PyFC makes to its users, and the matrix is where that
  promise is checked.

Fixed
~~~~~
* **The wheel no longer installs ``tests``, ``scripts``, ``docs`` and
  ``xbranch_compare`` as top-level packages.**
  ``[tool.setuptools.packages.find]`` had ``where = ["."]`` with no include
  filter, so setuptools treated every top-level directory in the repository as
  something to install: the 0.10.0 wheel declares eight top-level names and
  ships four of them with real content. The practical consequence is that
  ``pip install PyFeldmanCousins`` dropped a top-level ``tests`` package into
  the user's ``site-packages``, where it shadows or is shadowed by any other
  project's ``tests`` -- a collision that is hard to diagnose precisely because
  nothing about it points back at this package. Now ``include = ["pyfc*"]``,
  verified by building the wheel before and after and installing it into a
  clean virtualenv: ``import pyfc`` resolves to ``site-packages`` and none of
  the other names are importable. The ``pyfc-config`` entry point and the
  bundled licence are unaffected, and the sdist still carries ``tests/`` so a
  source checkout can run the suite.
* **Coverage now traces inside ``ProcessPoolExecutor`` workers.** ``toys.py``
  reported 62% with ``_worker_unbinned_toy`` counted as entirely missed. It was
  not missed: the end-to-end unbinned test dispatches every toy through it, but
  coverage does not follow child processes by default. An A/B on the same
  single test file, changing nothing but the setting, moved ``toys.py`` from
  26% to 37% without a single new test being written. This is the same class of
  error as the Numba problem ``scripts/run_coverage.sh`` exists for, and the
  more dangerous of the two: the Numba figure was absurd enough to force a
  second look, whereas a plausible 62% instead invites someone to write tests
  for code that is already covered.

0.10.0
------
Hardens the optimizer/likelihood layer against joint (non-box) parameter
constraints, informed by a concrete failure case from a downstream project
that needed to fit flavor fractions subject to ``f_e + f_mu <= 1``. Also
removes the ``S_model``/``B_model`` templating mechanism in favor of a
more general ``pdf_components`` list, and adds support for N-dimensional
binned data.

BREAKING CHANGES
~~~~~~~~~~~~~~~~
* **``S_model``/``B_model`` removed entirely from ``compute_fc_intervals``
  and every optimizer/toy-generation function** (``unconditional_fit_scipy``,
  ``conditional_fit_1d_scipy``, ``conditional_fit_2d_scipy``,
  ``unconditional_fit_ultranest``, ``conditional_fit_1d_ultranest``,
  ``conditional_fit_2d_ultranest``, ``generate_and_fit_toys_python``, and
  the binned/unbinned grid-search and toy-generation functions in
  ``binned.py``/``unbinned.py``). An audit established that for binned
  models, ``S_model``/``B_model`` were pure pass-through into
  ``compute_rates_func`` -- ``calc_nll`` never read them itself. For
  unbinned models they were a hardcoded signal/background *pair*, which
  was itself an unnecessary rigidity on top of being pass-through.
  ``compute_fc_intervals`` now takes only ``data`` and ``grids`` as
  required positional arguments (down from 4:
  ``data, S_model, B_model, grids``).

  .. warning::
     **Migration hazard:** because ``grids`` moves from position 4 to
     position 2, any caller still using positional arguments will silently
     pass the wrong value into the wrong parameter instead of raising an
     error. Convert all calls to fully keyword-form
     (``compute_fc_intervals(data=..., grids=..., compute_rates_func=..., ...)``).
* **New ``pdf_components`` mechanism replaces ``S_model``/``B_model`` for
  unbinned models.** ``pdf_components`` is a ``list[callable]`` of any
  length (no longer hardcoded to exactly 2), passed to
  ``compute_fc_intervals`` and to every optimizer/toy function. Each is
  evaluated exactly once per fit as ``pdf(data)`` and the resulting
  arrays are reused across every subsequent NLL evaluation in that fit --
  the same caching behavior as before, generalized beyond a fixed
  signal/background pair. Required (and validated) when
  ``likelihood_type="unbinned"``; ignored with a warning for ``"binned"``.
* **``compute_rates_func`` signature changes for both likelihood types:**

  * Binned: ``(params, S_template, B_template, S_sigma2, B_sigma2)``
    becomes ``(params, S_sumw2, B_sumw2)``. Fixed template arrays that
    used to be passed as arguments must now be referenced via closure
    (module-level constant or nested-function capture) instead -- this is
    already numba-``@njit``-compatible and is the pattern used throughout
    ``examples/pyfc_joint_constraints_tutorial.ipynb``'s own templates.
  * Unbinned: ``(params, s_probs, b_probs)`` becomes ``(params, probs)``,
    where ``probs`` is a ``list[np.ndarray]``, one entry per
    ``pdf_components[i]``, in the same order.
* **``S_sigma2``/``B_sigma2`` renamed to ``S_sumw2``/``B_sumw2``**
  everywhere they appear (``compute_fc_intervals``, every optimizer/
  toy-generation function, ``calc_nll``, and the binned grid-search
  functions). These arrays are the per-bin sum of squared MC weights
  (``sum(w_i^2)``) -- the standard variance estimator for a weighted MC
  sample -- used by the finite-MC (Poisson-Gamma) correction. The
  ``S_``/``B_`` prefixes now dangle less meaningfully without
  ``S_model``/``B_model`` as their counterpart, but the ``sigma2`` suffix
  was also imprecise (it names a *derived* quantity computed *by*
  ``compute_rates_func``, not what these arguments actually hold):
  ``sumw2`` is unambiguous and matches the term HEP users already know
  from ``TH1::Sumw2()``-style per-bin weight-squared tracking.
* **``data``, ``mu``, and ``sigma2`` now support arbitrary N-dimensional
  shapes for binned models**, not just flat 1D vectors -- a genuine 2D
  ``(E, cos_theta)`` histogram (or any other shape) can be evaluated
  directly, with no need to flatten it yourself first. ``calc_nll``
  flattens internally before summing, so the NLL is identical regardless
  of how the bins are laid out spatially. ``grids`` (the parameter
  *scan*, as opposed to the *data*) is unaffected -- it was already
  N-parameter-general and remains a list of 1D per-parameter arrays.
  Unbinned events already supported N-D feature vectors
  (``data.shape == (n_events, n_features)``); this is now documented as
  the indexing convention, along with a gotcha: bootstrapping an N-D MC
  pool requires index-based resampling (``idx = np.random.choice(len(mc_pool),
  size=n, replace=True); toy_events = mc_pool[idx]``), since
  ``np.random.choice`` applied directly to an N-D pool silently samples
  along the wrong axis.
* **Migration:** see the rewritten :doc:`quickstart` for both binned and
  unbinned examples under the new signatures.
* **``fc_results.json``'s 1D ``interval_bounds`` schema changed from a flat
  ``[lo, hi]`` pair to a list of ``[lo, hi]`` pairs, one per disconnected
  accepted interval.** Previously, ``_save_fc_json`` computed a single
  ``[min(accepted), max(accepted)]`` span across every accepted grid
  point. Under the Feldman-Cousins unified construction, the accepted
  region for a parameter can in principle be genuinely disconnected near
  certain physical boundaries (e.g. a non-monotonic rate function
  crossing the data at more than one point); a flat min/max span silently
  merges those separate intervals into one, including whatever rejected
  gap sits between them -- misrepresenting the actual confidence region.
  A new ``_find_contiguous_intervals`` helper in ``orchestrator.py`` now
  scans the scan-ordered ``test_points``/``accepted`` arrays for every
  maximal contiguous accepted run and reports each as its own
  ``[lo, hi]`` pair. **This always applies, even in the single-interval
  case** (still a one-element list, ``[[lo, hi]]``, not flattened) -- a
  consistent schema regardless of how many disjoint pieces the accepted
  region has. **Migration:** any downstream code reading
  ``interval_bounds`` as ``[lo, hi]`` directly (e.g.
  ``lo, hi = interval_bounds``) must be updated to iterate the list of
  pairs instead. 2D intervals are unaffected -- no ``interval_bounds``-
  style precomputed field existed there before or now; ``2d_intervals``
  stores the full accepted boolean grid, from which any region shape can
  already be reconstructed directly. ``fc_results.npz`` and
  ``generate_corner_plot`` are both unaffected -- neither ever consumed
  ``interval_bounds`` (the ``.npz`` stores raw per-point arrays, and
  plotting shades directly from those, not from the JSON's precomputed
  field). Documented in README.md's "Stored Results & Custom Plotting"
  section and :doc:`outputs`, with a worked example added to
  ``pyfc_algorithmic_features_tutorial.ipynb``. New tests for
  ``_find_contiguous_intervals`` and the JSON-writing logic in
  ``tests/test_disconnected_intervals.py``, including a genuine (not
  hand-mocked) end-to-end disconnected-region example.

Changed
~~~~~~~
* **``examples/pyfc_tutorial.ipynb`` replaced by four focused notebooks:**
  ``pyfc_quickstart_tutorial.ipynb`` (binned + unbinned, the runnable
  counterpart to the README Quick Start Guide), ``pyfc_high_dimensional_
  tutorial.ipynb`` (scaling to 5 and 10 parameters, including the
  combinatorial cost of 2D contours), ``pyfc_algorithmic_features_
  tutorial.ipynb`` (``sparsify_grid``/``smooth_1d``/``smooth_2d``/
  ``use_finite_mc_correction_binned``, each demonstrated on vs. off), and
  ``pyfc_strategy_comparison_tutorial.ipynb`` (``"grid"``/``"scipy"``/
  ``"hybrid"`` timing comparison, plus building a custom Matplotlib
  contour directly from PyFC's saved ``.json``/``.npz`` output). The old
  monolithic notebook's heaviest cells (``n_toys=100``, dense 40-point
  grids, ``sparsify_grid=False``, 2D contours) took 20+ minutes *each* to
  execute, which made it impractical to keep re-running end-to-end; the
  four replacements use small grids/``n_toys`` (in the style already
  established by ``pyfc_joint_constraints_tutorial.ipynb``) so each runs
  in well under a minute, and each carries more narrative markdown/
  comments explaining *why* a cell does what it does, not just what it
  does.
* **``calc_nll`` (binned.py) and ``calc_nll_unbinned`` (unbinned.py): behavior
  change in the unphysical (``mu_i <= 0`` / ``p_events[k] <= 0``) region.**
  Previously, both functions returned a flat constant penalty (``1e10``) the
  instant a bin/event's expected rate went non-positive -- ``binned.py``'s
  version was additionally an early return that discarded whatever NLL had
  already accumulated from other, physical bins in the same call. Because
  ``scipy.optimize.minimize(..., method='L-BFGS-B')`` estimates gradients by
  finite differences, that flat region reads as "gradient ~ 0, already
  converged," and could permanently trap the optimizer if its starting guess
  (the bounds midpoint, by default) landed there.

  Both functions now instead add a smooth, continuously-varying contribution
  for the unphysical region: a continuous extension of the real NLL term at a
  tiny positive floor, plus a quadratic barrier that grows with how far into
  the unphysical region the point actually is.

  .. warning::
     **NLL values in the previously-unphysical region will differ from prior
     runs** (they were an arbitrary constant before, so this is not a
     regression, but anyone diffing raw NLL outputs against pre-fix runs
     where the fit passed through an unphysical region should be aware of
     this). For all physical inputs (``mu_i > 0`` / all ``p_events[k] > 0``),
     output is practically identical to before (matches to within 1e-12
     relative/absolute tolerance; see ``tests/test_smoothing.py``).
* **Example notebooks renamed with a ``01``-``06`` numeric prefix
  reflecting suggested reading order**, and a new "Tutorial Notebooks"
  section added to README.md (plus a mirroring :doc:`tutorials`, linked
  from the Sphinx toctree) describing what each one covers and when to
  reach for it. Old name -> new name: ``pyfc_quickstart_tutorial.ipynb``
  -> ``01_pyfc_quickstart_tutorial.ipynb``,
  ``pyfc_high_dimensional_tutorial.ipynb`` ->
  ``02_pyfc_high_dimensional_tutorial.ipynb``,
  ``pyfc_non_contiguous_data_tutorial.ipynb`` ->
  ``03_pyfc_non_contiguous_data_tutorial.ipynb``,
  ``pyfc_algorithmic_features_tutorial.ipynb`` ->
  ``04_pyfc_algorithmic_features_tutorial.ipynb``,
  ``pyfc_strategy_comparison_tutorial.ipynb`` ->
  ``05_pyfc_strategy_comparison_tutorial.ipynb``,
  ``pyfc_joint_constraints_tutorial.ipynb`` ->
  ``06_pyfc_joint_constraints_tutorial.ipynb``.

  .. warning::
     **Migration:** update any bookmarks, scripts, or CI steps that
     reference the old filenames directly (nothing in this repository's
     own code or CI did). Each notebook's own "Next steps" cross-
     references and every mention in README.md/:doc:`installation` were
     updated to match; past changelog entries below that predate this
     rename still use the old names, as a historical record of what was
     true when they were written.
* **README.md's "Salient Features" list, and its (previously absent)
  counterpart on the Sphinx docs homepage, audited for accuracy against
  the current codebase and augmented.** Fixes: "Advanced optimizers" now
  names the ``"hybrid"`` strategy explicitly (it was already documented
  elsewhere and used throughout the example notebooks, just missing from
  this list and from both files' shared intro paragraph, which said only
  "SciPy, UltraNest, Grid"); "Dynamic 2D sparsification" no longer claims
  an unqualified "radically reducing computational overhead" now that
  ``sparsify_grid`` is known to default to ``False`` and not scale
  reliably past ~20x20 grid nodes per axis (both established earlier in
  this release). Additions: "Unified binned and unbinned analysis" now
  mentions N-dimensional/non-contiguous binned data support; "Exact
  coverage" now mentions disconnected-interval reporting; a new "Joint/
  simplex-constrained parameters" bullet covers ``bounds_func``/
  ``constraints``, previously undocumented at this level despite having
  its own dedicated tutorial notebook; "Massive parallelization" now
  mentions the validated ``adaptive_toys``/``toy_batch_size`` behavior
  from this same release. :doc:`index` gained its own "Salient Features"
  section mirroring README.md's (it previously had none), plus an
  "Important Links" pointer to :doc:`tutorials`.

Fixed
~~~~~
* **``calc_nll`` (binned.py) and the two grid-strategy toy generators
  (``generate_and_fit_toys_grid_1d``/``_2d``) crashed on non-contiguous N-D
  binned arrays** -- e.g. a transposed histogram (``data.T``) -- with a
  low-level ``numba`` ``NotImplementedError: incompatible shape for array``
  instead of working correctly. ``numba``'s ``nopython``-mode ``.reshape(-1)``
  only accepts C-contiguous input, and this was never triggered during the
  original N-D-support work, so the guard was never added. Since the entire
  point of N-D binned support is letting users lay out bins however is
  natural for them, this is a plausible real-world trigger, not a contrived
  edge case: anyone who transposes their histogram before calling
  ``compute_fc_intervals`` hit this crash. Fixed by copying to a contiguous
  layout first with ``np.ascontiguousarray(...)``, a no-op (zero cost, no
  copy) on the already-contiguous common path. Covered by new regression
  tests in ``tests/test_nd_binned.py``, both at the ``calc_nll`` level and
  end-to-end through ``compute_fc_intervals`` under both the ``"scipy"`` and
  ``"grid"`` toy-generation strategies.
* **``save_directory``'s default was inconsistent across the codebase.**
  ``compute_fc_intervals``'s own Python keyword default (``"fc_output"``) and
  ``generate_config.py``'s interactive-wizard default disagreed with
  ``pyfc/config.py``'s CLI/JSON-config default
  (``"output/example_fc_output"``), which every doc and example already
  assumed was *the* default. Calling ``compute_fc_intervals`` directly
  without a config dict or an explicit ``save_directory`` -- the style the
  README's own Quick Start Guide uses for its inline code snippets --
  silently wrote output somewhere different from what the docs describe.
  Unified all three to ``"output/example_fc_output"``.
* **``strategy="ultranest"``/``"hybrid"`` (2D intervals) crashed with
  ``IndexError: too many indices for array: array is 1-dimensional, but 2
  were indexed``**, raised inside ``ultranest.ReactiveNestedSampler``'s own
  internal ``_find_strategy`` method (confirmed present through at least
  UltraNest v4.5.0 and the current GitHub master, so this is not fixed by
  upgrading). An internal random subsample can occasionally select too few
  saved iterations, producing a degenerate 1D array where UltraNest's own
  code expects 2D. Since this stems from UltraNest's own internal
  exploration randomness rather than anything about the model being
  fitted, ``optimizers.py``'s three ``*_ultranest`` fit functions now
  retry with a fresh sampler instance on this specific error (new
  ``_run_ultranest_with_retry`` helper, mirroring
  ``_minimize_with_restarts``'s existing retry-on-failure pattern for the
  scipy path; covered by new tests in ``tests/test_ultranest_retry.py``).
  Separately, ``pyfc_strategy_comparison_tutorial.ipynb``'s grid/``n_toys``
  were scaled down, since its original size didn't finish even in 30
  minutes with ``ultranest`` actually installed -- as far as could be
  determined, this strategy combination had never been exercised
  end-to-end before.
* **``sparsify_grid`` defaulted to ``True`` in ``compute_fc_intervals``'s
  own Python keyword default, while ``config.py``'s CLI/JSON default and
  every doc already documented ``False`` as *the* default.** Anyone
  calling ``compute_fc_intervals`` directly without an explicit
  ``sparsify_grid`` argument silently got the (buggy, see below) ``True``
  behavior. Unified to ``False`` everywhere, including
  ``generate_config.py``'s interactive wizard, which independently
  defaulted to ``"y"``.
* **``sparsify_grid=True``'s 2D boundary-refinement pass was a complete
  no-op.** ``eval_2d_point``'s memoization guard checked
  ``2d_t_critical`` for ``NaN``-ness to decide whether a cell had already
  been evaluated, but the coarse-to-fine ``RectBivariateSpline``
  interpolation step fills in ``2d_t_critical`` for *every* cell (real or
  not) before the refinement pass runs -- so the guard always fired, and
  every non-coarse cell's ``2d_t_data`` stayed ``NaN`` forever. Since
  ``accepted = t_data <= t_critical`` is always ``False`` against
  ``NaN``, this silently excluded the vast majority of the grid from the
  accepted region regardless of its true classification (confirmed by
  direct reproduction: 220/256 cells force-excluded on a 16x16 grid,
  accepted count 8/256 instead of the true, much larger region). Fixed by
  keying the guard off ``2d_t_data`` instead, which is only ever set by
  an actual conditional fit and untouched by interpolation. New
  regression tests in ``tests/test_sparsify_grid.py``. **Known remaining
  limitation:** boundary refinement is still a single, non-iterative pass
  with a 1-cell-wide halo around ~5-per-axis coarse nodes, so for grids
  much larger than ~20x20 per axis (the scale this feature is meant for)
  it still under-counts the true accepted region -- see the caveat now
  documented in ``compute_fc_intervals``'s docstring and README's
  "Contour Edge Tracing" section. Tracked as follow-up work.
* **``compute_fc_intervals``'s own Python keyword defaults disagreed with
  README.md's documented defaults for six parameters**: ``n_toys``
  (``2000`` vs. documented ``500``), ``save_log`` (``False`` vs.
  ``True``), ``smooth_1d``/``smooth_2d`` (``False``/``False`` vs.
  ``True``/``True``), and ``cl`` (resolved to ``[0.90]`` when omitted,
  vs. documented ``[0.68, 0.90]``). Anyone calling ``compute_fc_intervals``
  directly without a config dict -- again, the README's own Quick Start
  Guide style -- silently got different behavior than what the docs
  describe (e.g. 4x fewer toys, no persistent log, unsmoothed plots, only
  a single CL). Unified all six to match README.md. ``num_cores``,
  ``output_file``, and ``param_names`` are deliberately left as their
  existing ``None``-sentinel defaults (auto-detect hardware threads, skip
  writing a results file, and auto-generate ``param{i}`` names matching
  the model's actual ``n_params``, respectively) rather than hardcoded to
  README's example-config values, since those three sentinels encode real
  auto-detect/opt-out behavior that a literal copy would remove or, for
  ``param_names``, actively break on any model where ``n_params != 3``;
  this is now spelled out explicitly in the docstring for each. Five
  example notebooks that omitted ``save_log``/``smooth_1d``/``smooth_2d``
  (relying on the old implicit defaults) were re-executed to refresh
  their embedded (now-smoothed) plot images:
  ``pyfc_algorithmic_features_tutorial.ipynb``,
  ``pyfc_high_dimensional_tutorial.ipynb``,
  ``pyfc_joint_constraints_tutorial.ipynb``,
  ``pyfc_non_contiguous_data_tutorial.ipynb``,
  ``pyfc_quickstart_tutorial.ipynb``, and
  ``pyfc_strategy_comparison_tutorial.ipynb``.
* **A razor-thin (~1e-12-wide) NLL discontinuity in the unphysical-region
  smoothing** (``binned.calc_nll``, ``unbinned.calc_nll_unbinned``), left
  over from the smoothing fix documented above. For a bin/event with real
  observed data, the physical branch's Poisson term correctly diverges to
  ``+inf`` as ``mu_i``/``p_events`` -> 0+, but the unphysical branch's
  base term is evaluated at a *fixed* floor (``mu_floor``/``p_floor`` =
  ``1e-12``) rather than at the actual value, and was triggered only at
  ``mu_i <= 0``/``p_events <= 0``. That left the tiny physical sliver
  ``(0, 1e-12)`` on the diverging physical branch, so crossing from
  ``+epsilon`` to ``-epsilon`` made the NLL *drop* right at the boundary
  instead of continuing to climb -- the opposite of the smoothing fix's
  whole purpose. Very unlikely to be hit in practice (no realistic
  optimizer step lands within ``1e-12`` of exactly zero), but fixed
  defensively anyway: the trigger is now ``mu_i <= mu_floor`` /
  ``p_events <= p_floor``, so the transition point is single and
  consistent, approached from both directions. ``mu_floor``/``p_floor``
  are astronomically smaller than any physically meaningful value, so
  this has no practical effect on real fits (verified: existing
  backward-compatibility tests for the physical-branch formula,
  ``test_binned_calc_nll_identical_for_physical_bins``/
  ``test_unbinned_calc_nll_identical_for_physical_events``, are
  unaffected). New fine-resolution monotonicity-sweep tests in
  ``tests/test_smoothing.py`` covering the exact old discontinuity point.
* **README.md's and ``docs/source/methodology.rst``'s displayed finite-MC
  correction formula did not match ``calc_nll``'s actual implementation at
  all** (not just a missing citation): the docs' formula had no
  ``lgamma``/``Gamma`` terms whatsoever, while the code implements an
  exact Negative Binomial log-likelihood built from ``lgamma``.
  Numerically evaluating both at the same ``(mu, sigma2, n)`` gives
  substantially different values -- the docs' formula was simply wrong,
  not an approximation of the code. Checked the code's
  ``alpha = mu^2/sigma^2 + 1`` (previously flagged as possibly a bug,
  since a naive mean-and-variance-matching derivation gives
  ``alpha = mu^2/sigma^2`` with no ``+1``) against Argüelles, Schneider &
  Yuan, "A binned likelihood for stochastic models", JHEP 06 (2019) 030
  [arXiv:1901.04645]: ``calc_nll``'s formula is *exactly* the paper's
  ``L_Eff`` (Eq. 3.16, their main, recommended result) -- the ``+1`` is a
  deliberate, load-bearing part of that specific (best-coverage)
  parameterization, not a bug, and no code change was needed. New
  hand-computed-reference tests in ``tests/test_finite_mc_likelihood.py``
  verify ``calc_nll``'s finite-MC branch against the paper's Eq. 3.16
  directly. Added an explicit citation and provenance note to
  ``calc_nll``'s docstring (this paper was already cited in
  ``docs/source/methodology.rst``/
  ``docs/source/refs.bib`` and README.md's own "Methodology References"
  section, just not from the code itself), and corrected both docs'
  displayed formula to match the paper/code exactly. Also corrected
  README.md's feature-list description, which called this the
  "Beeston-Barlow technique" -- a different, *profiled* (not marginalized)
  likelihood that the paper itself distinguishes from ``L_Eff`` (and which
  this codebase does not implement).
* **``adaptive_toys`` and ``toy_batch_size`` were documented, threaded
  through ``config.py``/``generate_config.py``'s CLI/JSON/wizard
  machinery, and part of ``compute_fc_intervals``'s own signature -- but
  neither was ever read anywhere to make an actual decision.** README's
  own OOM-troubleshooting advice ("reduce ``toy_batch_size`` from 200 to
  50") would have had zero effect. Made both real, for ``strategy`` in
  ``"scipy"``/``"ultranest"``/``"hybrid"``
  (``toys.generate_and_fit_toys_python``); no effect for
  ``strategy="grid"``, whose toy generators are a single vectorized
  ``numba`` ``@njit(parallel=True)`` call per point rather than a pool of
  per-toy tasks, not a natural fit for either batching or a sequential
  early-exit check.

  * ``toy_batch_size``: toys are now submitted/collected from the
    ``ThreadPoolExecutor``/``ProcessPoolExecutor`` in batches of this
    size instead of one ``executor.map`` call for all ``n_toys``,
    bounding how many toy datasets / in-flight results are alive at once
    (matters most for ``likelihood_type="unbinned"``, matching README's
    OOM scenario). Same total ``n_toys`` are always generated -- purely a
    dispatch/memory change, zero effect on results (proved directly: for
    binned toys, the same seed produces bit-for-bit identical output
    regardless of ``toy_batch_size``, since the underlying
    ``np.random.poisson`` call that generates the toy data happens once
    up front either way).
  * ``adaptive_toys``: after each ``toy_batch_size`` batch, computes the
    running fraction of toys with ``t_toy >= t_data`` (a p-value estimate
    for that point's accept/reject decision) and a strict 99.9% Wilson
    score confidence interval around it. If that interval lies entirely
    on one side of the target significance ``alpha = 1 - CL`` (using the
    *largest* requested ``cl`` when several are given, so stopping there
    guarantees every less-stringent ``cl``'s decision is also settled),
    the verdict cannot plausibly flip with more toys and generation stops
    early for that point. Never considered before 100 toys
    (``toys.ADAPTIVE_MIN_TOYS``), regardless of how extreme the running
    estimate looks -- small-sample binomial confidence intervals are
    unreliable. Meaningfully saves time only for points far from the
    accept/reject boundary; points near it correctly run the full
    ``n_toys``. ``compute_fc_intervals``'s own quantile-index computation
    (``t_stats[int(cl * n_toys)]``) now uses the *actual* number of toys
    generated for that point instead of the original ``n_toys``, since
    ``adaptive_toys`` can make that shorter.
  * **Validated, not just implemented** (per the risk this fix was
    flagged with -- a wrong stopping rule would silently bias confidence
    intervals, worse than the previous do-nothing state): new tests in
    ``tests/test_adaptive_toys.py`` cover the Wilson-interval/stopping-
    decision primitives directly, confirm well-inside/well-outside points
    stop at exactly the minimum toy count while a point near the true
    critical value runs the full ``n_toys``, and -- the real test of "did
    this introduce bias" -- confirm ``adaptive_toys=True`` reaches the
    *same* accept/reject verdict as ``adaptive_toys=False`` with the same
    ``n_toys`` cap, both for isolated toy-generation calls across many
    independent trials and for a full end-to-end
    ``compute_fc_intervals`` run. README's ``adaptive_toys``/
    ``toy_batch_size`` table rows, the OOM-troubleshooting paragraph, and
    :doc:`configuration`'s table are updated to describe what actually
    happens now, plus a new README "Adaptive Toy Generation" section
    (mirroring the existing "Contour Edge Tracing" section's depth).
* **``generate_config.py``'s interactive wizard still offered four
  default answers that disagreed with ``compute_fc_intervals``'s own
  (already README-aligned) defaults**, missed when those defaults were
  unified earlier in this release: ``n_toys`` (wizard default ``2000``
  vs. ``500``), ``smooth_1d``/``smooth_2d`` (wizard default ``n`` vs.
  ``True``), and ``save_log`` (wizard default ``n`` vs. ``True``). Anyone
  who ran the wizard and accepted every default (the documented,
  intended workflow) would get a config that silently diverged from what
  README.md and ``pyfc/config.py``'s own base config dict both call "the"
  default. Unified all four; the wizard's defaults now match everywhere
  else exactly, so pressing Enter on every question reproduces
  ``compute_fc_intervals``'s own defaults.
* **README's "Optimizer Strategy" section claimed PyFC discards a
  non-converged toy and automatically generates a replacement to preserve
  exact ``N_toys`` statistics -- no such mechanism exists anywhere in the
  codebase.** Confirmed by exhaustive grep across ``toys.py``/``binned.py``/
  ``unbinned.py``: toy-generation code never inspects a toy's own fit
  convergence, and ``optimizers.py``'s ``_minimize_with_restarts`` (which
  does check ``res.success``, retry once, and warn on non-convergence)
  applies identically to data fits and toy fits alike -- it does not
  discard or regenerate anything. Whatever ``t_stat`` comes out of a
  non-converged toy fit is used exactly like any other toy's result.
  Corrected the README bullet to describe actual behavior (a
  ``warnings.warn`` message on persistent non-convergence, no
  discard/regeneration, no ``verbose`` gating on the warning either --
  also inaccurately claimed). Actually implementing discard-and-replace
  was considered and deliberately deferred as a separate,
  carefully-scoped follow-up if ever wanted, since it would be a genuine
  new statistical feature (changing which toys contribute to the
  empirical critical-value distribution) rather than a documentation fix.
* **README.md's and ``docs/source/methodology.rst``'s displayed "Binned
  Likelihood" formula did not match ``calc_nll``'s actual
  implementation.** The docs showed the simple, non-saturated, single
  (not doubled) Poisson NLL (``mu_i - n_i*ln(mu_i)``), while ``calc_nll``'s
  own docstring and its actual ``n_obs > 0`` branch compute the
  saturated, doubled Baker-Cousins form
  (``2*(mu_i - n_i + n_i*ln(n_i/mu_i))``). Verified numerically: for a
  single-bin toy ``(mu_i=5, n_i=3)``, the old displayed formula gives
  ``0.172`` while ``calc_nll`` actually returns ``0.935``, matching the
  corrected formula exactly. Because ``t_data = NLL_cond - NLL_uncond`` is
  a difference evaluated at the same data in both fits, the missing
  saturation/doubling terms cancel exactly -- **this was a
  documentation-only bug; the FC interval construction itself was never
  affected** -- but ``results["data_uncond_nll"]`` and any other absolute
  NLL value PyFC reports would not match a reader's by-hand calculation
  from the old formula. Corrected both docs to the saturated, doubled
  formula, with an added note on the factor-of-2 convention (``calc_nll``
  computes ``-2 ln L`` directly, matching the Baker-Cousins/Wilks
  convention used for ``t`` itself, so no separate doubling is needed
  when forming ``t``).
* **``n_restarts`` and ``neighbor_seeding`` (``compute_fc_intervals``
  parameters, tested since ``tests/test_restarts.py``) were never wired
  into the CLI/JSON config system** -- the only two parameters (excluding
  inherently non-serializable ones) missing from ``config.py``'s
  argparse/base-config dicts, ``generate_config.py``'s interactive
  wizard, README's and :doc:`configuration`'s parameter tables, and
  ``config/example_fc_config.json``, unlike every other tunable knob.
  Reachable only via a direct Python call, not through the documented
  CLI/JSON workflow. Wired both through end to end, matching the existing
  pattern used for ``warm_start``/``toy_batch_size``; the new wizard
  questions default to ``n_restarts=1``/``neighbor_seeding=True``,
  matching ``compute_fc_intervals``'s own defaults.
* **``calc_nll``'s finite-MC (Poisson-Gamma) branch lost catastrophic
  floating-point precision for well-simulated templates (large
  ``alpha = mu^2/sigma^2 + 1``).** Evaluating
  ``ln(Gamma(n+alpha)) - ln(Gamma(alpha))`` as a literal difference of two
  ``lgamma`` calls loses precision once both terms are individually huge
  (``~alpha*ln(alpha)``) while their true difference is only
  ``O(n*ln(alpha))`` -- double precision's ~16 significant digits get
  eaten by the leading digits both terms share. Verified against a
  50-digit ``mpmath`` reference run through the actual compiled
  ``calc_nll``: at ``mu=1000, sigma^2=1e-9`` (a realistic
  well-simulated-template regime), the old formula was off by 7.5 in
  absolute NLL, large enough to distort ``t = NLL_cond - NLL_uncond`` and
  bias interval boundaries. Fixed via two algebraically exact (not
  approximated) reformulations: the lgamma difference is now computed as
  the exact integer-recurrence sum ``SUM_{k=0}^{n-1} ln(alpha + k)``, and
  ``alpha*ln(beta) - (n+alpha)*ln(1+beta)`` is rewritten using ``log1p``
  to avoid an analogous (smaller but real) cancellation. Re-verified
  against the same 50-digit reference across 500 randomized
  ``(mu, sigma2, n)`` trials spanning ordinary-to-extreme ``alpha``:
  worst-case error now 2.6e-8, down from an unbounded-growing error
  before. New regression test in ``tests/test_finite_mc_likelihood.py``
  pins the exact bug-report case and asserts the naive formula is still
  measurably wrong there. The ``sigma2 <= 1e-10`` Poisson fallback for
  near-zero simulation variance is unaffected by this fix and was
  confirmed to be a deliberate, mathematically-correct limit (NB
  collapses onto Poisson as ``sigma^2 -> 0``), not a numerical
  workaround -- documented as such in ``calc_nll``'s own docstring.
* **``toys.generate_and_fit_toys_python`` crashed with
  ``ValueError: range() arg 3 must not be zero`` when called with
  ``n_toys=0`` and the default ``toy_batch_size=None``.**
  ``compute_fc_intervals``'s own entry point is shielded (its
  ``toy_batch_size`` default is ``200``, always positive), but this is
  public API with no ``n_toys`` validation of its own. Fixed by flooring
  the fallback batch size at 1 (``max(n_toys, 1)``), so ``n_toys=0`` now
  correctly returns an empty result array instead of crashing. New
  regression test in ``tests/test_adaptive_toys.py``.
* **``constraints`` were silently unenforced whenever a 1D/2D scan fixed
  every parameter, leaving zero free parameters to optimize over** (a
  natural simplification of this codebase's own flavor-fraction
  constraint example, down to exactly 2 parameters with no extra
  nuisance parameters). All four ``conditional_fit_*_scipy``/
  ``conditional_fit_*_ultranest`` functions have a
  ``len(free_bounds) == 0`` early-return path that bypasses their normal
  optimizer machinery entirely (scipy's ``_minimize_with_restarts``/
  UltraNest's ``log_likelihood`` closure, which is where ``constraints``
  is otherwise projected/checked) -- so a scan point that individually
  violated a joint constraint silently returned an ordinary finite NLL
  instead of being rejected like every other constraint-violating point.
  Fixed by checking ``_constraints_satisfied`` directly against the
  fully-assembled point in all four early-return branches, returning the
  same ``1e10`` flat penalty already used elsewhere in this module for a
  violated constraint. New regression tests in ``tests/test_constraints.py``
  covering all four functions (the two UltraNest ones gated on
  ``ultranest`` being installed).
* **``orchestrator.py``'s own ``__main__`` demo block never forwarded
  ``n_restarts``/``neighbor_seeding``** to either of its two
  ``compute_fc_intervals(...)`` calls, even though ``config.py`` correctly
  parses both from the CLI (added earlier in this release) -- the
  residual gap in that change. ``python -m pyfc.orchestrator --n_restarts
  5 --neighbor_seeding false`` parsed without error but silently had zero
  effect end to end. Now both calls pass ``config.get("n_restarts", 1)``/
  ``config.get("neighbor_seeding", True)``, matching every other config
  key already forwarded there.
* **``scipy_method`` (fully wired into ``config.py`` and both README's/
  :doc:`configuration`'s parameter tables) had no ``generate_config.py``
  wizard question and no key in ``config/example_fc_config.json``** --
  the same bug class as the ``n_restarts``/``neighbor_seeding`` gap
  above, just older (predates this release) and missed by earlier
  sweeps. Added a wizard question (``"auto"`` sentinel mapping back to
  ``None``, matching the ``num_cores`` ``0``-to-``None`` pattern already
  used there) and the corresponding JSON key.
* **NaN ``mu_i`` (binned) / ``p_events`` (unbinned) silently bypassed the
  unphysical-region barrier** instead of triggering it:
  ``mu_i <= mu_floor``/``p_events <= p_floor`` are both ``False`` for
  NaN, so a NaN arising from a bug in the user's ``compute_rates_func``
  (division by zero, sqrt/log of a negative number) fell through into
  ``math.log(NaN)``/``np.log(NaN)``, silently producing a NaN NLL with
  no warning and no gradient signal for the optimizer to recover from (a
  quadratic barrier needs a finite distance-from-floor to mean anything;
  NaN has none). Both now raise an explicit ``ValueError`` identifying
  the likely cause.

  .. warning::
     ``calc_nll`` is ``@njit(fastmath=True)``, and numba's ``fastmath``
     compiles under LLVM's "assume no NaN" flag, which makes
     ``math.isnan(x)``, ``np.isnan(x)``, and even the ``x != x``
     self-inequality trick all silently return ``False`` for a genuine
     NaN under ``fastmath`` -- verified directly before landing this fix.
     The actual check reinterprets the float64's raw bits as an int64 and
     tests the IEEE-754 NaN bit pattern directly (exponent all 1s,
     mantissa nonzero), since integer bitwise ops aren't affected by
     ``fastmath``'s floating-point assumptions. ``calc_nll_unbinned`` is
     plain Python/NumPy (not ``@njit``), so its fix uses ordinary
     ``np.isnan`` with no such caveat.

  New regression tests in ``tests/test_smoothing.py``, including one
  that pins the fastmath/isnan interaction directly so a future
  "simplification" back to a standard NaN check would be caught.
* **Three inaccurate behavioral claims in the docs, found by
  re-verifying every specific claim against the actual current code:**
  (1) README's "press Enter on every question to reproduce
  [``compute_fc_intervals``'s defaults] exactly" was false for
  ``num_cores``/``output_file``/``param_names``, which the wizard
  correctly maps to ``None``-sentinels while the adjacent example JSON
  block shows illustrative literal values (``8``, ``"fc_results"``,
  etc.) -- reworded to carve out this exception explicitly rather than
  overclaim exactness, confirmed by actually running the wizard end to
  end. (2) README's and :doc:`outputs`'s checkpoint-resume descriptions
  claimed resuming happens "from the exact point of interruption"; the
  real granularity (confirmed via ``_save_fc_archive``'s call sites) is
  per-1D-parameter and per-2D-pair, matching what the index page and
  ``examples/07_pyfc_checkpointing_tutorial.ipynb`` already said
  correctly -- an internal inconsistency, not just loose phrasing.
  (3) ``plotting.generate_corner_plot``'s docstring claimed it returns
  ``None`` and closes the matplotlib figure to free memory; it actually
  returns the live ``Figure`` and never closes it (an intentional prior
  change that the docstring was never updated to match) -- corrected,
  and covered by a new regression test in ``tests/test_plotting.py`` (a
  previously fully untested function).

Added
~~~~~
* **``generate_corner_plot`` now importable as ``from pyfc import generate_corner_plot``**
  (previously only reachable as ``from pyfc.plotting import generate_corner_plot``),
  matching ``compute_fc_intervals``'s existing top-level export. README.md,
  :doc:`quickstart`, and all example notebooks now import both functions this
  way -- code copied from the docs/tutorials no longer needs to know PyFC's
  internal module layout (``orchestrator.py``/``plotting.py``).
* **``bounds_func`` hook** on ``unconditional_fit_scipy``/``unconditional_fit_ultranest``
  and ``conditional_fit_1d/2d_scipy``/``conditional_fit_1d/2d_ultranest``
  (optimizers.py), plus a ``bounds_func=None`` kwarg on
  ``compute_fc_intervals`` (orchestrator.py): lets the box bounds of the
  profiled (free) parameters depend on whatever parameter(s) the scan
  currently has fixed, so a joint constraint like ``f_mu <= 1 - f_e_test``
  can be expressed without a user-side penalty function. Default ``None``
  preserves prior behavior exactly. See :ref:`joint-simplex-constraints`.
* **``constraints`` support** on the same scipy/ultranest fit functions, plus
  ``constraints=None``/``scipy_method=None`` kwargs on ``compute_fc_intervals``:
  accepts a list of ``scipy.optimize.LinearConstraint``/``NonlinearConstraint``
  objects (expressed in the full parameter-vector space) for constraints
  among multiple *simultaneously-free* nuisance parameters, which
  ``bounds_func`` cannot express. Automatically switches the scipy method
  from ``L-BFGS-B`` to ``SLSQP`` when constraints are supplied (overridable
  via ``scipy_method``, e.g. ``"trust-constr"``); a new ``--scipy_method``
  CLI/JSON config option is also available. A shared
  ``_project_linear_constraint`` helper projects full-space constraints down
  to the free-parameter subspace for the conditional fit functions. Default
  ``None``/empty preserves prior behavior and method exactly.
* **Optimizer restarts and ``res.success`` checking** (optimizers.py's new
  ``_minimize_with_restarts`` helper, used by all three scipy fit
  functions): ``res.success``/``res.status`` from ``scipy.optimize.minimize``
  were previously read nowhere in the codebase -- a silently non-converged
  fit's result was used exactly like a converged one. Now, a non-converged
  result triggers one cheap perturbed-restart retry and a warning if that
  also fails. A new ``n_restarts`` parameter (default ``1``, unchanged
  behavior aside from the check above) allows trying multiple starting
  points and keeping the best-converged one.
* **Neighbor warm-starting for the data fit** (orchestrator.py): Phase 1's
  1D scan and Phase 2's 2D scan previously started every single grid point's
  data fit fresh from the bounds midpoint, with zero information sharing
  between adjacent, already-solved grid points, even though these loops
  already iterate the grid in order. (MC toy fits were already well-seeded
  from the conditional MLE and are unaffected.) A new
  ``neighbor_seeding=True`` kwarg on ``compute_fc_intervals`` seeds each grid
  point's data fit from an adjacent, already-evaluated point's profiled
  parameters; whenever used, ``n_restarts`` is forced to at least 2 (the
  neighbor-seeded start plus a fresh bounds-midpoint start) so a bad
  neighbor optimum can't silently cascade forward. Set ``False`` to restore
  the always-fresh-start behavior.
* README section "Handling Joint/Simplex-Constrained Parameters" (also
  mirrored, with worked code examples, in :ref:`joint-simplex-constraints`):
  a worked end-to-end tutorial for the ``f_e + f_mu <= 1`` example, covering
  both ``bounds_func`` and ``constraints`` and when to use each.
  ``configuration.rst`` and ``quickstart.rst`` cross-reference it, and this
  page itself is new.
* New example notebook, ``examples/pyfc_joint_constraints_tutorial.ipynb``:
  a hands-on walkthrough of both mechanisms using a toy 3-parameter
  neutrino flavor-fraction model (``f_e``, ``f_mu``, ``norm``) constrained
  to the standard 2-simplex. Reproduces the flat-gradient plateau trap with
  a hard-cutoff rate function and no constraint handling, then fixes it
  with ``bounds_func``; demonstrates ``constraints`` for the case
  ``bounds_func`` cannot express (two simultaneously-free parameters); runs
  a real ``compute_fc_intervals`` pipeline and plots the resulting 2D
  confidence region confined to the simplex; and includes a bonus
  ``NonlinearConstraint`` (unit-disk) example.
* ``tests/test_nd_binned.py``: flatten-invariance checks (a genuine 2D
  histogram and its 1D-flattened equivalent must give byte-identical NLL)
  plus a full ``compute_fc_intervals`` smoke test on 2D-shaped data.
* ``tests/test_pdf_components.py``: correctness checks for the new
  ``pdf_components`` mechanism with both 2 and 3+ components, including a
  hand-computed reference NLL for the 3-component case (no old-API
  equivalent to diff against).
* ``xbranch_compare/``: a reusable cross-branch regression harness (kept
  in the repo for future breaking API changes, not a one-off validation)
  that empirically proves this refactor changed signatures only, never
  numerics, by running equivalent models against both ``dev`` (old API,
  via a temporary ``git worktree``) and ``dev-no-templates`` (new API) and
  diffing the results.
* New example notebook, ``examples/pyfc_non_contiguous_data_tutorial.ipynb``:
  demonstrates the non-contiguous-array fix above -- passes a transposed
  histogram straight into ``compute_fc_intervals`` and confirms the result
  matches a contiguous copy of the identical data, after first illustrating
  the underlying ``numba``/``numpy`` contiguity restriction in isolation.
* ``.github/workflows/changelog-sync.yml`` and
  ``scripts/check_changelog_sync.py``: a CI check that fails the build if
  this page and ``CHANGELOG.md`` structurally drift out of sync (version/
  subsection headings, bullet counts, and each bullet's opening wording),
  since the two are hand-maintained in parallel and RST-only enhancements
  (admonitions, cross-references) mean a byte-for-byte diff or a generic
  Markdown-to-RST converter isn't viable. See the script's module docstring
  for exactly what is and isn't checked.
* New example notebook, ``examples/07_pyfc_checkpointing_tutorial.ipynb``:
  demonstrates ``warm_start`` checkpointing/resume against a genuinely
  interrupted run, not a simulated one. Launches a real
  ``compute_fc_intervals`` call in a ``multiprocessing.Process``, lets it
  run long enough for one parameter's 1D scan to checkpoint, sends it
  ``SIGTERM`` (via ``.terminate()``, the same signal a Slurm walltime kill
  sends), inspects the resulting ``checkpoint_fc.npz`` to confirm a
  genuine partial state, then resumes with the identical call and
  ``warm_start=True``. Verifies correctness (not just "it finished") two
  ways: the resumed run completes measurably faster than a fresh
  reference run (the checkpointed parameter wasn't recomputed), and the
  checkpointed parameter's data test statistic (a deterministic quantity,
  no MC randomness involved) matches the reference run's value exactly.

0.9.2 and earlier
-----------------
See the `git history <https://github.com/mbustama/FeldmanCousins/commits/main/>`_.
