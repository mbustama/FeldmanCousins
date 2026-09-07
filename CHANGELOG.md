# Changelog

<!--
This file is the single source of truth for the version history.
docs/source/changelog.rst renders it directly via myst-parser, so editing
this file is all that is needed -- there is no RST mirror to keep in step,
and nothing to run afterwards.
-->

All notable changes to PyFC are documented in this file.

## [0.20.0]

A quality-and-packaging release rather than a feature one: no public API
changes, and no change to any number PyFC computes. It adds test-coverage
measurement, acts on the three real gaps that first measurement exposed, and
fixes two defects and one packaging error that the work surfaced along the way.

The coverage figure is the visible outcome (64% at the start of the work to 81%
locally, higher in CI), but the more useful result is that the figure is now
honest. Three separate mechanisms were causing code that runs on every test to
be reported as untested -- Numba's `@njit` compilation, `ProcessPoolExecutor`
child processes, and `ThreadPoolExecutor` threads -- and each, taken at face
value, would have sent someone to write tests for code that was already covered.
The measurement is now arranged so that none of those lie, and the reasoning is
recorded in `pyproject.toml` and `scripts/run_coverage.sh` so it does not have
to be rediscovered.

Of the new tests, the ones worth knowing about are the self-discovering
structural sweeps over the configuration layer, which check that every analysis
parameter lines up across all four places it is written down. Every new test in
this release was verified to fail against a deliberately broken version of the
code before being kept.

> **Upgrading from 0.10.0 — one manual cleanup step.**
>
> 0.10.0, the version currently on PyPI, installed `tests`, `scripts`, `docs`
> and `xbranch_compare` into `site-packages` as top-level packages (see the
> packaging entry under *Fixed*). 0.20.0 installs only `pyfc`, but **upgrading
> does not fully undo the damage, and neither does `pip uninstall`.** Measured
> on a clean virtualenv: pip removes the files it recorded and prunes three of
> the four directories, but leaves an empty `docs/` behind — and an empty
> directory is still an importable namespace package, so `import docs` in that
> environment continues to resolve into `site-packages` and can shadow a real
> `docs` package belonging to another project.
>
> To check whether an environment is affected:
>
> ```bash
> python -c "import docs; print(docs.__path__)"
> ```
>
> If that succeeds and points into `site-packages` with nothing underneath, the
> directory is PyFC's leftover and is safe to delete. If it points anywhere else,
> or contains files, it belongs to something else — leave it alone.
>
> Environments that never installed 0.10.0 are unaffected, and nothing inside
> PyFC itself depends on these names.

### Added

- **Test-coverage measurement.** `pytest-cov` joins the `test` and `dev`
  extras, the settings live in `[tool.coverage.run]` in `pyproject.toml` so a
  local run measures exactly what CI measures, and a separate Coverage job in
  `pytest.yml` reports the figure on each run's summary page and uploads
  `coverage.xml` as an artifact. Branch coverage is enabled: a line-coverage
  number flatters this codebase, because the orchestrator is a dispatch machine
  whose failure modes are branches only ever taken one way -- `strategy="grid"`
  vs `"adaptive"`, a resumed checkpoint vs a fresh run, the
  disconnected-interval reporting path, the `if ultranest is None` guards --
  and line coverage counts all of those as covered the moment the function runs
  at all. The job reports rather than gates: there is deliberately no
  `--cov-fail-under`, since a threshold invented before the first measurement
  either sits below the real figure and never fires, or above it and blocks
  unrelated work. It also carries a Codecov upload step gated on a
  `CODECOV_TOKEN` secret, skipped entirely when that secret is absent so the
  workflow stays safe to run in a fork; the secret now exists, so uploads are
  live, the coverage badge resolves, and Codecov comments per-PR diff coverage.
- **`scripts/run_coverage.sh`, which splits the suite around Numba's JIT.**
  `coverage.py` traces Python bytecode, and Numba executes none: it compiles
  `@njit` functions to machine code, so every line inside `binned.py` -- six
  compiled functions including `calc_nll`, the NLL everything else is built on
  -- is reported as missed no matter how hard the suite hits it. Measured over
  the three test files that target `calc_nll` directly, `binned.py` reports
  **10%** with the JIT on and **90%** under `NUMBA_DISABLE_JIT=1`; publishing
  the former would be actively misleading about the part of PyFC that most
  needs to be trustworthy. Disabling the JIT wholesale is not an option either,
  since `strategy="grid"`'s `@njit(parallel=True)` toy generators degrade into
  plain Python loops and the suite then fails to finish in any reasonable time.
  The script therefore runs the pathological files with the JIT on, everything
  else with it off, and lets `coverage` combine the halves. The classification
  lives in the script rather than in the workflow so that there is exactly one
  place it can go stale, and its failure mode is benign: a newly added test file
  falls into the JIT-disabled half automatically, so it is measured correctly
  and at worst runs slowly, never silently skipped.
- **Structural and behavioural tests for the configuration layer**, closing what
  that first coverage run found: `config.py` sat at 10% and
  `generate_config.py` at 6%, meaning essentially nothing tested the CLI/JSON
  config layer -- a layer edited on almost every release, whose correctness had
  until now been established by running the CLI by hand. They take those two
  modules to 100% and 98% respectively, and the package as a whole from 66% to
  75%. The structural sweeps
  check that every analysis parameter lines up across all four places it is
  written down: the hardcoded defaults, the `argparse` flags, the interactive
  wizard's questions, and `compute_fc_intervals`' own signature and defaults.
  They are self-discovering -- each derives its subjects from the code rather
  than from a hardcoded list -- so a parameter added later is swept without
  anyone remembering to extend a list, which is the only reason they would have
  caught the two real v0.10.0 bugs they are modeled on (`scipy_method` reaching
  `parse_arguments` but never being added to the wizard, and a wizard default
  disagreeing with the one `compute_fc_intervals` actually uses). Alongside
  them, the layer's stated `Defaults -> JSON -> CLI` precedence chain is now
  tested in both directions, as is `get_input`'s rejection of a bad cast, a
  value outside `choices`, and a failed validator. Every one of these tests was
  verified to fail against a deliberately broken version of the code before
  being kept.
- **Tests for the unbinned grid-search path** (`tests/test_unbinned_grid.py`),
  the second gap the measurement exposed. `unbinned.py` sat at 34%, and reading
  which lines were cold showed the shortfall was not scattered: four whole
  functions -- `conditional_fit_grid_unbinned_1d`,
  `conditional_fit_grid_unbinned_2d`, `generate_and_fit_toys_grid_unbinned_1d`
  and `generate_and_fit_toys_grid_unbinned_2d` -- were executed by nothing at
  all, despite every one being imported and dispatched to by
  `compute_fc_intervals`. Across the entire suite `likelihood_type="unbinned"`
  appeared exactly once, and that test used `strategy="scipy"`, so the unbinned
  grid path shipped and was reachable by users while never being run. That
  combination is the one place in PyFC where profiling is exhaustive rather
  than numerical, which makes a silent error there especially hard to notice: a
  scan that quietly profiles over the wrong axis still returns plausible,
  finite, monotonic-looking numbers. The tests therefore compare against
  oracles computed independently in the test body -- an explicit loop over the
  same grid calling `calc_nll_unbinned` directly -- and against the invariants
  that define a profile likelihood ratio: the conditional NLL never undercuts
  the unconditional minimum, and the two coincide exactly at the best-fit
  point. The end-to-end run reconstructs the expected `t_data` array rather
  than asserting finiteness, because `t_data` is clamped at zero and a
  mis-wired dispatch would otherwise hide behind a valid-looking array of
  zeros. Also closes the unbinned twins of two guards `binned.py` has had
  tested since v0.10.0: the NaN per-event-density rejection and the quadratic
  barrier for unphysical densities. Takes `unbinned.py` from 34% to 100%; all
  13 mutations tried against these tests were detected.
- **Tests for the toy pool's `ThreadPoolExecutor` fallback**
  (`tests/test_toy_pool_fallback.py`). The unbinned toy path dispatches each toy
  to a separate process, which imposes a requirement users are never told about
  at the call site: every callable they pass must be picklable, and a lambda or
  a closure is not. `toys.py` already recovers from that by warning and retrying
  the batch on threads, but nothing exercised the recovery, since it only runs
  once the pool has already failed. Both halves of its contract are now covered:
  that unpicklable closures still yield a complete set of test statistics, and
  that a genuine bug in user code surfaces *uncaught* on the thread retry rather
  than being converted into a warning plus a plausible-looking array of numbers.
  **Known issue, found by writing them:** on Python 3.9 that recovery does not
  recover -- it deadlocks. Breaking the pool while Numba's threads are live in
  the forked parent leaves a worker that never exits, and the implicit
  `shutdown(wait=True)` on the way out of the failed block waits on it forever.
  Observed as a two-hour CI hang on the 3.9 runner, with orphaned workers
  reported at cleanup, while 3.10 and 3.11 passed in minutes. The tests are
  skipped below 3.10 rather than removed, since what they document is a real
  exposure for users on 3.9 and not a defect in the tests. `requires-python` is
  still `>=3.8`; narrowing it, or making the fallback non-blocking, is a
  deliberate decision that has not been taken here.

- **`extra_nll`, a hook for soft (Gaussian) constraints on nuisance parameters.**
  `bounds_func` and `constraints` express HARD geometry on the parameter box;
  nothing could express the SOFT statement that a parameter is `1.0 +- 4.6%`
  because it was measured, which is how an external constraint on a systematic
  enters a physics likelihood. `extra_nll(params) -> float` is added to the NLL
  at every evaluation -- a callable rather than a table of `(centre, width)`
  triples, because triples cannot express a one-sided bound (a cosmological
  limit) nor a constraint on a quantity derived from several parameters. It
  receives the FULL parameter vector in `grids` order with scan-fixed values
  substituted, the convention `constraints` already uses and the only one under
  which one function serves the unconditional fit, the 1D scan and the 2D scan
  alike.

  It applies to both likelihoods, all four strategies, and unconditional,
  conditional AND toy fits. The toys are the point: a term reaching the data fit
  but not the toys would have the observed statistic and its critical value
  computed under different likelihoods, silently destroying the coverage the
  method exists to provide. Three tests target that specific failure, because
  end-to-end numbers cannot see it -- toys are generated at `true_params` taken
  from the conditional fit, so removing the penalty from the toy dispatch still
  moves the results and the run still looks plausible.

  **Units:** PyFC's NLL is `-2 ln L`, so a Gaussian of width `s` contributes
  `((x - c)/s)**2`, NOT `0.5 * ((x - c)/s)**2`. The wrong convention makes every
  constraint weaker by `sqrt(2)` -- a stated 4.6% prior acting as 6.5% -- with a
  converged fit and no symptom. A closed-form test pins it: a lone parameter
  constrained only by this term gives a test statistic of exactly 1.0 at `c + s`.

  With `strategy="grid"` the callable must be `@njit`-decorated, since the grid
  scan runs inside compiled code; a plain callable raises `TypeError` naming the
  decorator, rather than the 316-character numba `TypingError` that mentions
  neither `njit` nor the offending argument. Callers who do not use the feature
  pay ~5%, not the ~170% a jitted no-op would have cost, because numba compiles
  the `is not None` branch away per specialisation.

- **`examples/08_pyfc_soft_constraints_tutorial.ipynb`**, the runnable
  counterpart to the entry above. It opens with the units check as a *numerical*
  demonstration rather than an assertion -- a lone parameter whose predicted rate
  ignores it, so the likelihood is the prior alone and the test statistic has the
  closed form `((x - c)/s)**2`; the correct penalty lands on 1.0, 4.0, 9.0 at one,
  two and three sigma, and the `-ln L` form lands on exactly half of each, which
  is what the `sqrt(2)` weakening looks like from the other side. It then runs the
  full construction twice on identical data, once with a signal/background
  degeneracy left free and once with a 3% prior on the background, and shows the
  observed statistic against the toy-derived critical value for both -- the
  critical values differ between the runs, which is the visible consequence of the
  penalty reaching the toys. The remaining sections cover the vectorised
  many-priors form, correlated priors built with `pyfc.priors` (see the entry
  below), a one-sided penalty on a derived sum, and the guards firing -- both the
  `strategy="grid"` one on a plain Python callable and the builders' own on a
  singular covariance, a correlation matrix passed where a covariance was wanted,
  mismatched shapes and a repeated index.

  The notebook is named for the mechanism rather than for one shape of it. It
  covers a plain Gaussian, correlated blocks, a one-sided bound and a constraint
  on a derived quantity, only the first of which is Gaussian, so "soft
  constraints" is what the file and its title now say. Nothing was released under
  the old name. Two notes on what it does *not* claim:
  the toy is deliberately small, so the effect shown is the upper limit on the
  signal strength halving rather than the total loss of a limit described above,
  and the notebook says so; and neither run excludes zero signal, which is correct
  for data generated at the no-signal expectation. `docs/source/tutorials.rst` and
  the README table gain a row for it, paired with notebook `06` as the soft and
  hard halves of the same constraint story.

- **`pyfc.priors`, builders for correlated and composite priors.** `extra_nll`
  could always express a pairwise or higher-dimensional constraint -- it receives
  the full parameter vector, so a correlated Gaussian is a quadratic form with
  off-diagonal terms, verified here against an exact oracle at a correlation of
  0.9 and over a dense 13x13 covariance (worst |t-1| = 2.8e-09), including inside
  the `@njit` grid scan. What it could not do was stop three silent mistakes, and
  that is what these builders are for.

  `gaussian_block(indices, centres, cov)` takes a COVARIANCE and inverts it once,
  at build time, in `-2 ln L` units. `gaussian_block_from_correlation` takes the
  form results are actually published in -- central values, per-parameter sigmas,
  a correlation matrix -- because the outer product is easy to omit and omitting
  it still leaves a symmetric positive-definite matrix that nothing complains
  about. `combine_priors` sums any number of them; composition is additive, nests
  cheaply (3.03 ns for one 2x2 block, 4.85 ns for three nested), and stays jitted,
  so adding a second constraint does not silently cost you `strategy="grid"`.

  **Marginal, not conditional.** Profiling a Gaussian returns the marginal, so the
  1-sigma point a scan reports for one parameter of a correlated block is
  `sqrt(Sigma_ii)` -- a COVARIANCE element -- and not `1/sqrt((Sigma^-1)_ii)`,
  which is what reading the inverse's diagonal gives. At a correlation of 0.9
  those are 0.100 and 0.0436: a factor of 2.3, in the direction of claiming a
  constraint twice as tight as the measurement supports. The whole test file is
  built on that identity, since one assertion catches a dropped off-diagonal, a
  wrong units convention, and an inverse applied one time too many or too few.

  **Why the builders and not a hand-written function.** Three measured reasons.
  Numba freezes a global or closed-over matrix BY VALUE when it compiles: editing
  the covariance afterwards is ignored, silently, for `M[i,j] = x`, `M *= x`,
  `M[:] = new` and rebinding the name alike -- so these are factories, and
  changing a constraint means rebuilding, visibly. `d @ inv @ d` costs ~154 ns
  almost regardless of size, which is BLAS dispatch overhead rather than
  arithmetic, against ~1.2 ns for an explicit expression over a 2x2 block; since
  the prior runs at every NLL evaluation of every fit of every toy, and one
  `calc_nll` is ~628 ns, the matmul form would add ~25% to a pairwise-constrained
  run for nothing. The builders switch formulation at a measured crossover of 35
  parameters, exposed as `MATMUL_THRESHOLD`. And a covariance that is not positive
  definite is accepted by `np.linalg.inv` while making the penalty unbounded below,
  so the "constraint" pushes the fit away and the run completes looking ordinary;
  that is rejected up front, naming the smallest eigenvalue, the tolerance and the
  condition number. The test is by eigenvalue with a relative tolerance rather than
  by `np.linalg.cholesky`, which is unreliable for exactly this case: at
  `sigma = (0.2, 0.3)` with correlation exactly 1 the determinant rounds to `+1e-18`
  and cholesky accepts the matrix -- measured, it accepted 1 of 8 exactly-singular
  covariances, so a single test fixture passes or fails on rounding luck. The guard
  and its test are now parametrised over several widths, and a companion test pins
  that merely ill-conditioned matrices (condition number up to 1e12) still build.

  `gaussian_block`, `gaussian_block_from_correlation` and `combine_priors` are
  re-exported from `pyfc`. Section 4 of notebook `08` works through all of it, and
  all 16 mutations of the module -- dropped off-diagonal factor, missing inverse,
  ignored threshold, each validation guard, the symmetrisation of the inverse --
  are caught by `tests/test_priors.py`.
- **`pyfc.__version__`**, read from the installed distribution's metadata rather
  than written down a second time. `pyproject.toml` is the single source: the
  package exposes it, and `docs/source/conf.py` imports that value instead of
  repeating the `importlib.metadata` lookup, so there is one implementation and
  one place where the distribution-vs-import name distinction can be got wrong.
  `tests/test_version.py` pins the whole chain -- that the attribute exists and
  is exported, that it is not the uninstalled fallback, that it agrees with
  `pyproject.toml`, that the docs take their version from the package, and that
  no module docstring reintroduces a hand-written version string. All five links
  fail silently when broken, which is how the docs came to advertise 0.1.0 for
  nine releases.
- **Tests that actually run the three UltraNest fits**
  (`tests/test_ultranest_fits.py`). Installing `ultranest` moved `optimizers.py`
  from 59% only to 63%, and the cold lines said why: nothing ran the fits.
  `test_ultranest_retry.py` exercises the retry wrapper against a mocked
  sampler, and `test_constraints.py` reaches the two conditional fits only in
  the zero-free-parameters case, which returns before sampling begins. So
  `unconditional_fit_ultranest`'s body was entirely unrun and both conditional
  fits were entered but never sampled -- roughly the last third of the module,
  shipped and reachable via `strategy="ultranest"`/`"hybrid"`. The new tests fit
  an Asimov dataset, where the generating parameters are an oracle rather than
  something read back out of the fit, and check MLE recovery, that the tested
  parameters are pinned exactly, the profile-likelihood invariants in both
  directions, the `bounds_func` hook, the unbinned likelihood branch, the toy
  generators' UltraNest paths through both pools, and agreement with SciPy on
  the same optimum -- the last being the check that would catch the two backends
  being wired to different likelihoods, which no single-backend test can see.
  Tolerances come from measurement: the sampler lands within 0.004 of the truth,
  and the test statistic reads about -1e-4 at the true value, so the invariant
  assertions carry that much slack rather than demanding exactly zero.
  `optimizers.py` 59% -> 92%, `toys.py` 78% -> 88%.
- **Tests for the JSON results export** (`tests/test_json_export.py`).
  `orchestrator.py`'s cold lines were not in the statistics but in the
  serialisation layer: `NumpyEncoder`'s type conversions and the entire 2D
  branch of the payload were unrun. That layer is what a user actually reads,
  and a `TypeError` there arrives at the very end of a long run, after all the
  expensive work is done. Each encoder branch is now checked individually, along
  with NaN/infinity handling and the fallback that must keep rejecting genuinely
  unserialisable objects, and a real 2D run is read back from disk and
  re-dumped through a stock `json.dump` to prove nothing NumPy-typed survived.
  `orchestrator.py` 84% -> 89%.
- **Tests for the last two cold paths in `orchestrator.py`**
  (`tests/test_2d_resume_and_sparsify.py`): the 2D half of the checkpoint-resume
  block, and `sparsify_grid`'s nearest-neighbour interpolation fallback. Both
  were unrun because of a condition no existing test satisfied --
  `test_checkpoint_resume.py` passes `compute_2D_intervals=False` on every run,
  and the fallback is the `else` of `if SCIPY_AVAILABLE`, which never fires
  because SciPy is a hard dependency of the test extra. 2D contours are the
  expensive half of a run, so they are exactly what a user restarting after a
  walltime kill most needs back, and silently recomputing them would look
  identical to working. Both tests needed a second attempt to become
  meaningful: the resume test first used two parameters and asserted no 2D fit
  was recomputed, which is the wrong claim -- checkpoints are written per
  completed *pair*, so a crash mid-pair legitimately loses that pair -- and now
  uses three parameters and asserts on the amount of work skipped. The sparsify
  test first used a 5x5 grid, at which size the edge-refinement pass
  re-evaluates every cell and overwrites the interpolation entirely, so
  deleting the fallback changed nothing; at 15x15 the interior survives, and
  removing it leaves 79 of 225 cells at NaN. `orchestrator.py` 89% -> 94%,
  package total 92% -> 93%.
- **A Downloads badge**, in `README.md` and on the documentation homepage,
  matching the placement used in the author's Magnus project: between the
  Python-version and ruff badges. It reports PyPI installs of
  `PyFeldmanCousins` via pepy.tech. Note the figure it shows counts mirrors and
  CI traffic alongside real users, so it is an upper bound rather than a
  readership number, and until 0.20.0 ships it reflects 0.10.0 alone.

### Changed

- **`orchestrator.py`'s 13 `E701` lint findings are fixed**, so `ruff check`
  now reports zero. They were `if condition: statement` one-liners in the
  logging helper, the `S_sumw2`/`B_sumw2` defaulting, and the checkpoint-restore
  block -- style rather than defects, but they were also the only findings
  standing between the project and a clean lint signal, and `lint.yml` runs
  `ruff check` with `continue-on-error: true`, so they could never fail a build
  and were never going to be fixed by accident. Verified purely cosmetic rather
  than assumed: the file's abstract syntax tree is byte-identical before and
  after, the suite passes, and the cross-branch numerical harness reports zero
  mismatches. `ruff format --check` is untouched and still reports 49 files, as
  it did before -- that is a separate, much larger decision about reformatting a
  codebase that predates the formatter.
- **Releases are gated on the test suite.** `publish.yml` triggers on
  `release: published` and went straight to building and uploading; `pytest.yml`
  runs on pushes and pull requests, not on releases, so a release cut from a
  commit with a failing suite reached PyPI exactly as readily as a green one.
  A `test` job now runs the full matrix on the commit being published and
  `build-and-publish` carries `needs: test`. It repeats the matrix rather than
  linking to `pytest.yml` via `workflow_run`, which would tie publication to
  whatever that workflow last did on some branch instead of to the commit
  actually going out. This matters more than a normal CI gate because the action
  is irreversible: a PyPI release can be yanked but its version number can never
  be reused, so a bad upload is corrected only by burning the number.
- **Seven README links now point at absolute GitHub URLs.** The example-notebook
  links were repository-relative, which renders correctly on GitHub and 404s on
  PyPI, where there is no file tree beside the description. Verified against the
  built wheel's metadata rather than the source file: the published description
  now contains no repository-relative links at all. Badges and the logo were
  already absolute.
- **PyPI classifiers list the Python versions actually tested.** The package
  advertised only a generic `Programming Language :: Python :: 3`, so it did not
  appear under any version filter on PyPI despite CI running three. Added 3.9,
  3.10 and 3.11 -- and deliberately not 3.12, which works locally but that
  nothing tests; advertising an untested interpreter is the mistake
  `requires-python = ">=3.8"` already made once. Also added
  `Operating System :: OS Independent`.
- **`docs/source/changelog.rst` now renders this file directly instead of
  mirroring it by hand.** It was a 1037-line RST duplicate of these 988 lines,
  kept in step manually and policed by `scripts/check_changelog_sync.py` and a
  dedicated CI workflow. The check worked -- it caught real drift twice while
  0.20.0 was being written -- but it could only ever *detect* divergence, never
  prevent it, and every entry had to be written twice. The page is now eight
  lines that `.. include::` the Markdown through `myst-parser`, so this file is
  the single source and there is nothing to keep in step and nothing to run
  afterwards. `scripts/check_changelog_sync.py` and
  `.github/workflows/changelog-sync.yml` are removed as obsolete, and
  `myst-parser` joins the `docs` extra.
  The trade is presentational and was checked entry by entry before making it:
  six `.. warning::` admonitions become inline bold text and fifteen
  `:doc:`/`:ref:` cross-references become plain prose. No *information* is lost
  -- every one of those passages already existed in this file, since the sync
  check only ever compared each bullet's first four words and the RST's extra
  wording was rendering rather than content. The docs build emits exactly the
  same 28 warnings as before, none of them from the changelog page.
- **The `Last modified: vX.Y.Z` line is gone from every module docstring.** It
  was hand-maintained and nothing could derive it, so it drifted exactly as a
  duplicated number always does: when 0.20.0 was prepared, seven of the nine
  modules still read `v0.10.0`, and no reader could tell whether that meant
  "genuinely unchanged since then" or "somebody forgot to update it" -- which
  makes the annotation worse than absent, because it invites a conclusion it
  cannot support. Git already records when a file last changed, accurately and
  without anyone having to remember. `pyfc.__version__` is the one version a
  caller needs, and a test now fails if a hand-written version string reappears
  in a module docstring. The `Created:` lines stay: those are fixed historical
  facts that cannot go stale.
- **`requires-python` narrowed from `>=3.8` to `>=3.9`.** The 3.8 claim was
  never checked by anything: CI has only ever run 3.9/3.10/3.11, and this
  release is the first time a real 3.8 environment was built and the suite run
  against it. The result is that 141 of 148 tests pass on 3.8, but `toys.py`'s
  `ProcessPoolExecutor` recovery path deadlocks -- it hangs indefinitely instead
  of falling back to threads. That path is reachable by any user who passes a
  lambda or a closure, so it is not an obscure corner. It is also not fixable
  from here: CPython rewrote `concurrent.futures.process` in 3.9
  (`_ExecutorManagerThread`, plus the `cancel_futures` argument to `shutdown`),
  and 3.8's legacy queue-management implementation is what hangs -- the same
  non-blocking shutdown that fixes 3.9 does not rescue it. With 3.8 also
  end-of-life since October 2024, the declaration now matches what is actually
  tested. Users on 3.8 keep resolving to 0.10.0.
- **CI now runs on `dev` and `dev-*` branches, not only `main`.** The previous
  filter meant a topic branch got no CI signal at all until a pull request was
  opened, which is the point at which a failure is most expensive to discover.
- **The Coverage job installs `.[test,optimizers]`** while the Python-version
  matrix deliberately stays on `.[test]` alone. Without `ultranest` the three
  `*_ultranest` fit functions count as missed and the reported figure for
  `optimizers.py` understates its real coverage, which invites someone to go and
  "fix" coverage that is not actually missing; but that `ultranest` is optional
  is a promise PyFC makes to its users, and the matrix is where that promise is
  checked.

### Fixed

- **`unconditional_fit_ultranest` now returns an `ndarray` like every other fit
  function.** It passed UltraNest's result dict straight through, so it alone
  among the six returned a plain Python list. `compute_fc_intervals` assigns
  whichever fit it called directly into `results["best_fit"]`, which it
  initialises as an ndarray, so the type of that entry silently depended on
  which strategy ran. Nothing broke, because the value is only indexed and
  serialised -- but any NumPy operation added to it later would have worked
  under three strategies and failed under the fourth, at the end of a long run.
  Found by writing the first test that actually calls the function.
- **The wheel no longer installs `tests`, `scripts`, `docs` and
  `xbranch_compare` as top-level packages.** `[tool.setuptools.packages.find]`
  had `where = ["."]` with no include filter, so setuptools treated every
  top-level directory in the repository as something to install: the 0.10.0
  wheel declares eight top-level names and ships four of them with real
  content. The practical consequence is that `pip install PyFeldmanCousins`
  dropped a top-level `tests` package into the user's `site-packages`, where it
  shadows or is shadowed by any other project's `tests` -- a collision that is
  hard to diagnose precisely because nothing about it points back at this
  package. Now `include = ["pyfc*"]`, verified by building the wheel before and
  after and installing it into a clean virtualenv: `import pyfc` resolves to
  `site-packages` and none of the other names are importable. The `pyfc-config`
  entry point and the bundled licence are unaffected, and the sdist still
  carries `tests/` so a source checkout can run the suite.
- **Coverage now traces inside `ProcessPoolExecutor` workers and threads.**
  `toys.py` reported 62% with `_worker_unbinned_toy` counted as entirely missed.
  It was not missed: the end-to-end unbinned test dispatches every toy through
  it, but coverage does not follow child processes by default. This is the same
  class of error as the Numba problem `scripts/run_coverage.sh` exists for, and
  the more dangerous of the two: the Numba figure was absurd enough to force a
  second look, whereas a plausible 62% instead invites someone to write tests
  for code that is already covered. Note that `concurrency` *replaces*
  coverage's default rather than adding to it, so both values are needed --
  naming only `multiprocessing` stops tracing threads, and `toys.py`'s binned
  path runs every toy through a `ThreadPoolExecutor`. Measured over
  `test_adaptive_toys.py` + `test_unbinned_toys.py`, which between them exercise
  both pools: 58% with the default, 61% with `multiprocessing` alone (a
  different blind spot, not a smaller one), 70% with both. No new test was
  written to produce any of that; the lines were always running.
- **`pyfc-config` no longer spins forever when input runs out.**
  `get_input`'s catch-all `except Exception` swallowed the `EOFError` that
  `input()` raises on a closed or exhausted stdin, and `while True` retried
  immediately. Since stdin stays at EOF the retry fails identically with no
  delay, so the loop span at full CPU re-printing its prompt: ~7.4 million
  iterations in 5 seconds, and one such process was found still running after
  nearly three hours. That made the wizard unusable anywhere without a terminal
  -- CI, cron, a container, or behind a pipe whose input ran out -- and
  `yes "" | pyfc-config` hid it completely, because `yes` never stops. EOF is
  now caught ahead of the catch-all and ends the run with exit status 1, naming
  the question that ran out, which is the useful detail when a script piped too
  few answers. It is deliberately *not* treated as "accept the default": that
  would write a configuration the user never saw, and a wizard that invents
  answers when nobody is listening fails more quietly than one that spins.
- **The toy pool's fallback no longer deadlocks on Python 3.9.** When the
  `ProcessPoolExecutor` broke, the `with` block called `shutdown(wait=True)` on
  the way out -- waiting on exactly the workers that were wedged, rather than
  falling through to the `ThreadPoolExecutor` retry a few lines below. On 3.9
  that never returned: a CI run sat for two hours before being cancelled, with
  orphaned workers reported at cleanup, while 3.10 and 3.11 passed in minutes.
  The pool's lifetime is now managed explicitly and the two exits differ: the
  success path still waits, since those results are the return value, while the
  failure path abandons the pool with `shutdown(wait=False)`. Verified against a
  real 3.9 + Numba environment, where the reproduction went from a 120-second
  timeout to passing in 0.95 s; `tests/test_toy_pool_fallback.py` is no longer
  skipped below 3.10. Note the failure mode was a hang rather than a red test,
  and it required the full pytest context -- neither a stdlib reduction of the
  same `try`/`with`/`map`/`except` shape nor a standalone script calling the
  same code path reproduces it -- so what actually guards against a regression
  is the `timeout-minutes` now set on the CI matrix job.

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
- **README.md's "Salient Features" list, and its (previously absent)
  counterpart on the Sphinx docs homepage, audited for accuracy against
  the current codebase and augmented.** Fixes: "Advanced optimizers" now
  names the `"hybrid"` strategy explicitly (it was already documented
  elsewhere and used throughout the example notebooks, just missing from
  this list and from both files' shared intro paragraph, which said only
  "SciPy, UltraNest, Grid"); "Dynamic 2D sparsification" no longer claims
  an unqualified "radically reducing computational overhead" now that
  `sparsify_grid` is known to default to `False` and not scale reliably
  past ~20x20 grid nodes per axis (both established earlier in this
  release). Additions: "Unified binned and unbinned analysis" now
  mentions N-dimensional/non-contiguous binned data support; "Exact
  coverage" now mentions disconnected-interval reporting; a new "Joint/
  simplex-constrained parameters" bullet covers `bounds_func`/
  `constraints`, previously undocumented at this level despite having its
  own dedicated tutorial notebook; "Massive parallelization" now mentions
  the validated `adaptive_toys`/`toy_batch_size` behavior from this same
  release. `docs/source/index.rst` gained its own "Salient Features"
  section mirroring README.md's (it previously had none), plus an
  "Important Links" pointer to :doc:`tutorials`.

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
- **README's "Optimizer Strategy" section claimed PyFC discards a
  non-converged toy and automatically generates a replacement to preserve
  exact `N_toys` statistics -- no such mechanism exists anywhere in the
  codebase.** Confirmed by exhaustive grep across `toys.py`/`binned.py`/
  `unbinned.py`: toy-generation code never inspects a toy's own fit
  convergence, and `optimizers.py`'s `_minimize_with_restarts` (which does
  check `res.success`, retry once, and warn on non-convergence) applies
  identically to data fits and toy fits alike -- it does not discard or
  regenerate anything. Whatever `t_stat` comes out of a non-converged toy
  fit is used exactly like any other toy's result. Corrected the README
  bullet to describe actual behavior (a `warnings.warn` message on
  persistent non-convergence, no discard/regeneration, no `verbose`
  gating on the warning either -- also inaccurately claimed). Actually
  implementing discard-and-replace was considered and deliberately
  deferred as a separate, carefully-scoped follow-up if ever wanted, since
  it would be a genuine new statistical feature (changing which toys
  contribute to the empirical critical-value distribution) rather than a
  documentation fix.
- **README.md's and `docs/source/methodology.rst`'s displayed "Binned
  Likelihood" formula did not match `calc_nll`'s actual implementation.**
  The docs showed the simple, non-saturated, single (not doubled) Poisson
  NLL (`mu_i - n_i*ln(mu_i)`), while `calc_nll`'s own docstring and its
  actual `n_obs > 0` branch compute the saturated, doubled Baker-Cousins
  form (`2*(mu_i - n_i + n_i*ln(n_i/mu_i))`). Verified numerically: for a
  single-bin toy `(mu_i=5, n_i=3)`, the old displayed formula gives
  `0.172` while `calc_nll` actually returns `0.935`, matching the
  corrected formula exactly. Because `t_data = NLL_cond - NLL_uncond` is a
  difference evaluated at the same data in both fits, the missing
  saturation/doubling terms cancel exactly -- **this was a
  documentation-only bug; the FC interval construction itself was never
  affected** -- but `results["data_uncond_nll"]` and any other absolute
  NLL value PyFC reports would not match a reader's by-hand calculation
  from the old formula. Corrected both docs to the saturated, doubled
  formula, with an added note on the factor-of-2 convention (`calc_nll`
  computes `-2 ln L` directly, matching the Baker-Cousins/Wilks convention
  used for `t` itself, so no separate doubling is needed when forming
  `t`).
- **`n_restarts` and `neighbor_seeding` (`compute_fc_intervals` parameters,
  tested since `tests/test_restarts.py`) were never wired into the
  CLI/JSON config system** -- the only two parameters (excluding
  inherently non-serializable ones) missing from `config.py`'s
  argparse/base-config dicts, `generate_config.py`'s interactive wizard,
  README's and `docs/source/configuration.rst`'s parameter tables, and
  `config/example_fc_config.json`, unlike every other tunable knob.
  Reachable only via a direct Python call, not through the documented
  CLI/JSON workflow. Wired both through end to end, matching the existing
  pattern used for `warm_start`/`toy_batch_size`; the new wizard questions
  default to `n_restarts=1`/`neighbor_seeding=True`, matching
  `compute_fc_intervals`'s own defaults.
- **`calc_nll`'s finite-MC (Poisson-Gamma) branch lost catastrophic
  floating-point precision for well-simulated templates (large `alpha =
  mu^2/sigma^2 + 1`).** Evaluating `ln(Gamma(n+alpha)) - ln(Gamma(alpha))`
  as a literal difference of two `lgamma` calls loses precision once both
  terms are individually huge (`~alpha*ln(alpha)`) while their true
  difference is only `O(n*ln(alpha))` -- double precision's ~16
  significant digits get eaten by the leading digits both terms share.
  Verified against a 50-digit `mpmath` reference run through the actual
  compiled `calc_nll`: at `mu=1000, sigma^2=1e-9` (a realistic
  well-simulated-template regime), the old formula was off by 7.5 in
  absolute NLL, large enough to distort `t = NLL_cond - NLL_uncond` and
  bias interval boundaries. Fixed via two algebraically exact (not
  approximated) reformulations: the lgamma difference is now computed as
  the exact integer-recurrence sum `SUM_{k=0}^{n-1} ln(alpha + k)`, and
  `alpha*ln(beta) - (n+alpha)*ln(1+beta)` is rewritten using `log1p` to
  avoid an analogous (smaller but real) cancellation. Re-verified against
  the same 50-digit reference across 500 randomized `(mu, sigma2, n)`
  trials spanning ordinary-to-extreme `alpha`: worst-case error now
  2.6e-8, down from an unbounded-growing error before. New regression
  test in `tests/test_finite_mc_likelihood.py` pins the exact bug-report
  case and asserts the naive formula is still measurably wrong there
  (guards against this test silently losing its teeth if the regime
  changes). The `sigma2 <= 1e-10` Poisson fallback for near-zero
  simulation variance is unaffected by this fix and was confirmed to be
  a deliberate, mathematically-correct limit (NB collapses onto Poisson
  as `sigma^2 -> 0`), not a numerical workaround -- documented as such in
  `calc_nll`'s own docstring.
- **`toys.generate_and_fit_toys_python` crashed with `ValueError: range()
  arg 3 must not be zero` when called with `n_toys=0` and the default
  `toy_batch_size=None`.** `compute_fc_intervals`'s own entry point is
  shielded (its `toy_batch_size` default is `200`, always positive), but
  this is public API with no `n_toys` validation of its own. Fixed by
  flooring the fallback batch size at 1 (`max(n_toys, 1)`), so `n_toys=0`
  now correctly returns an empty result array instead of crashing. New
  regression test in `tests/test_adaptive_toys.py`.
- **`constraints` were silently unenforced whenever a 1D/2D scan fixed
  every parameter, leaving zero free parameters to optimize over** (a
  natural simplification of this codebase's own flavor-fraction
  constraint example, down to exactly 2 parameters with no extra nuisance
  parameters). All four `conditional_fit_*_scipy`/`conditional_fit_*_
  ultranest` functions have a `len(free_bounds) == 0` early-return path
  that bypasses their normal optimizer machinery entirely (scipy's
  `_minimize_with_restarts`/UltraNest's `log_likelihood` closure, which is
  where `constraints` is otherwise projected/checked) -- so a scan point
  that individually violated a joint constraint silently returned an
  ordinary finite NLL instead of being rejected like every other
  constraint-violating point. Fixed by checking `_constraints_satisfied`
  directly against the fully-assembled point in all four early-return
  branches, returning the same `1e10` flat penalty already used elsewhere
  in this module for a violated constraint. New regression tests in
  `tests/test_constraints.py` covering all four functions (the two
  UltraNest ones gated on `ultranest` being installed).
- **`orchestrator.py`'s own `__main__` demo block never forwarded
  `n_restarts`/`neighbor_seeding`** to either of its two
  `compute_fc_intervals(...)` calls, even though `config.py` correctly
  parses both from the CLI (added earlier in this release) -- the
  residual gap in that change. `python -m pyfc.orchestrator --n_restarts 5
  --neighbor_seeding false` parsed without error but silently had zero
  effect end to end. Now both calls pass `config.get("n_restarts", 1)`/
  `config.get("neighbor_seeding", True)`, matching every other config key
  already forwarded there.
- **`scipy_method` (fully wired into `config.py` and both README's/
  `configuration.rst`'s parameter tables) had no `generate_config.py`
  wizard question and no key in `config/example_fc_config.json`** -- the
  same bug class as the `n_restarts`/`neighbor_seeding` gap above, just
  older (predates this release) and missed by earlier sweeps. Added
  wizard question 5 (`"auto"` sentinel mapping back to `None`, matching
  the `num_cores` `0`-to-`None` pattern already used there) and the
  corresponding JSON key.
- **NaN `mu_i` (binned) / `p_events` (unbinned) silently bypassed the
  unphysical-region barrier** instead of triggering it: `mu_i <=
  mu_floor`/`p_events <= p_floor` are both `False` for NaN, so a NaN
  arising from a bug in the user's `compute_rates_func` (division by
  zero, sqrt/log of a negative number) fell through into
  `math.log(NaN)`/`np.log(NaN)`, silently producing a NaN NLL with no
  warning and no gradient signal for the optimizer to recover from (a
  quadratic barrier needs a finite distance-from-floor to mean anything;
  NaN has none). Both now raise an explicit `ValueError` identifying the
  likely cause. **Subtlety worth flagging for future work on this
  module:** `calc_nll` is `@njit(fastmath=True)`, and numba's `fastmath`
  compiles under LLVM's "assume no NaN" flag, which makes
  `math.isnan(x)`, `np.isnan(x)`, and even the `x != x` self-inequality
  trick all silently return `False` for a genuine NaN under `fastmath` --
  verified directly before landing this fix. The actual check reinterprets
  the float64's raw bits as an int64 and tests the IEEE-754 NaN bit
  pattern directly (exponent all 1s, mantissa nonzero), since integer
  bitwise ops aren't affected by `fastmath`'s floating-point assumptions.
  `calc_nll_unbinned` is plain Python/NumPy (not `@njit`), so its fix uses
  ordinary `np.isnan` with no such caveat. New regression tests in
  `tests/test_smoothing.py`, including one that pins the fastmath/isnan
  interaction directly so a future "simplification" back to a standard
  NaN check would be caught.
- **Three inaccurate behavioral claims in the docs, found by re-verifying
  every specific claim against the actual current code:** (1) README's
  "press Enter on every question to reproduce \[`compute_fc_intervals`'s
  defaults\] exactly" was false for `num_cores`/`output_file`/
  `param_names`, which the wizard correctly maps to `None`-sentinels while
  the adjacent example JSON block shows illustrative literal values (`8`,
  `"fc_results"`, etc.) -- reworded to carve out this exception explicitly
  rather than overclaim exactness, confirmed by actually running the
  wizard end to end. (2) README's and `outputs.rst`'s checkpoint-resume
  descriptions claimed resuming happens "from the exact point of
  interruption"; the real granularity (confirmed via `_save_fc_archive`'s
  call sites) is per-1D-parameter and per-2D-pair, matching what
  `index.rst` and `examples/07_pyfc_checkpointing_tutorial.ipynb` already
  said correctly -- an internal inconsistency, not just loose phrasing.
  (3) `plotting.generate_corner_plot`'s docstring claimed it returns
  `None` and closes the matplotlib figure to free memory; it actually
  returns the live `Figure` and never closes it (an intentional prior
  change that the docstring was never updated to match) -- corrected, and
  covered by a new regression test in `tests/test_plotting.py` (a
  previously fully untested function).

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
- New example notebook, `examples/07_pyfc_checkpointing_tutorial.ipynb`:
  demonstrates `warm_start` checkpointing/resume against a genuinely
  interrupted run, not a simulated one. Launches a real
  `compute_fc_intervals` call in a `multiprocessing.Process`, lets it run
  long enough for one parameter's 1D scan to checkpoint, sends it
  `SIGTERM` (via `.terminate()`, the same signal a Slurm walltime kill
  sends), inspects the resulting `checkpoint_fc.npz` to confirm a genuine
  partial state, then resumes with the identical call and `warm_start=True`.
  Verifies correctness (not just "it finished") two ways: the resumed run
  completes measurably faster than a fresh reference run (the checkpointed
  parameter wasn't recomputed), and the checkpointed parameter's data test
  statistic (a deterministic quantity, no MC randomness involved) matches
  the reference run's value exactly.

## [0.9.2] and earlier

See git history.
