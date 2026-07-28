# Brief: Fix findings from the full-codebase audit (post fixes #1-#6)

**Purpose of this document**: handoff brief for a fresh chat session, to be pasted or
`@`-referenced at the start of that session. Written by an assistant that had just completed
a "full codebase deep dive" (code, comments, docstrings, README, tests, docs/rst) across
PyFC's `dev-no-templates` branch, at the repo owner's request, immediately before running low
on context budget from a long session. The repo owner's own words: *"I want to fix everything
you found. However, I see that your context window for this chat session is getting filled
up, so I would rather you produce a very detailed brief..."* -- so the repo owner has already
agreed, in principle, to fixing everything below, but several items still involve a genuine
judgment call (flagged explicitly) that a rushed, wrong implementation would make *worse* than
leaving as-is -- read those sections carefully before touching code, same discipline as the
prior `FIXES_BRIEF_dev-no-templates.md` round (see docs/dev/ for that precedent).

Repository: `/home/mbustamante/Research/FeldmanCousins` (PyFC, GPLv3, sole author is the repo
owner). Branch: **`dev-no-templates`** (already exists, already has everything described below
as "already done"). Continue working directly on this branch. Do **not** touch `main` or `dev`.
This branch has not been pushed to GitHub yet -- this is still pre-push cleanup.

---

## 0. Context: what already happened on this branch (do NOT redo this)

Chronologically, on top of the large `S_model`/`B_model`-removal refactor and an earlier
full-codebase review (11 findings, #1-9 fixed, #10-11 explicitly deferred -- see below), this
branch has been through two further rounds:

1. **"Deep pre-push inspection" fixes #1-#6** (tracked in `docs/dev/FIXES_BRIEF_dev-no-templates.md`,
   already archived): `sparsify_grid` defaulted to `False` and its broken 2D boundary-refinement
   guard fixed; `compute_fc_intervals`'s defaults aligned with README's documented defaults;
   a razor-thin NLL discontinuity at the unphysical-region floor fixed; the finite-MC
   `alpha = mu^2/sigma^2 + 1` formula verified correct against its source paper (arXiv:1901.04645);
   disconnected 1D accepted intervals now reported as separate `[lo, hi]` pairs instead of
   merged; and `adaptive_toys`/`toy_batch_size` made genuinely functional (Wilson-score-based
   early stopping, validated against a non-adaptive reference).
2. **Tutorial notebooks reorganized and expanded**: all six example notebooks renumbered
   `01`-`06` in suggested reading order (with every cross-reference updated); a new
   `07_pyfc_checkpointing_tutorial.ipynb` added, demonstrating a genuinely-interrupted
   (`SIGTERM`-killed) `compute_fc_intervals` run resumed correctly via `warm_start`; a new
   README "Tutorial Notebooks" section and mirroring `docs/source/tutorials.rst` added; and
   README's "Salient Features" list (plus a newly-added matching section on
   `docs/source/index.rst`, which previously had none) audited and brought up to date --
   `"hybrid"` strategy now named explicitly, `sparsify_grid`'s overstated "radically reduces
   overhead" claim corrected, and several existing-but-undocumented capabilities (N-D data,
   disconnected intervals, joint/simplex constraints, validated adaptive toys) added.
3. **This audit** (this document's findings): a systematic pass over every `.py` file in
   `pyfc/` and `tests/`, every `.rst` file in `docs/source/`, all of README.md, CHANGELOG.md,
   `pyproject.toml`, `.gitignore`, and the CI workflow files, cross-checked against `git log`,
   `pyflakes`, and `ruff`, specifically hunting for (a) missing/incomplete documentation and
   (b) leftover content from earlier versions that's no longer accurate. Nothing below has
   been fixed yet -- this document exists specifically because that audit, plus this whole
   session's fix work, filled up the context budget.

**Deliberately out of scope, already established by the repo owner in an earlier session**
(see `docs/dev/FIXES_BRIEF_dev-no-templates.md`'s own context section): finding #10
(`pyproject.toml`'s `requires-python = ">=3.8"` claim, unverified since CI only tests
3.9/3.10/3.11) and finding #11 (repo-wide `ruff` non-compliance). These reappear in this
audit's own findings 15-16 below purely for completeness/quantification (nothing has changed
about them) -- **do not fix them without first confirming with the repo owner that "fix
everything" is meant to override that earlier, explicit deferral.** Ask, don't assume.

**Current state**: full test suite green (75 tests with `ultranest` installed, 72 + 3 skipped
otherwise), cross-branch harness (`xbranch_compare/run_comparison.sh`) passes with zero
mismatches, `git worktree list` is clean, `git status --short` is clean except this brief file
itself (untracked -- same situation as every prior handoff brief; per the established
precedent, ask the repo owner whether to archive it into `docs/dev/` once this round is done).

---

## 1. Misleading claim: README describes a toy-replacement mechanism that does not exist

### The problem (root-caused, not yet resolved -- read the whole section before deciding how to fix this)

README.md, "Optimizer Strategy (`strategy`)" section, "Error Handling" bullet (grep for
`Error Handling:` in README.md to relocate if line numbers have shifted):

```
*   **Error Handling:** If an optimizer fails to converge for a specific toy or grid point,
    PyFC logs a warning to the console (if `verbose > 0`), discards the failed toy, and
    automatically attempts to generate a replacement to preserve exact $N_{\text{toys}}$
    statistics.
```

**No such mechanism exists anywhere in the codebase.** Confirmed by exhaustive grep:

```bash
grep -rn "replacement\|discard\|regenerat" pyfc/*.py   # only one unrelated hit, in binned.py's
                                                        # docstring about NLL accumulation, not toys
grep -n "res.success\|\.success\b" pyfc/toys.py pyfc/binned.py pyfc/unbinned.py   # zero hits
grep -n "_minimize_with_restarts\|n_restarts" pyfc/toys.py   # zero hits
```

`toys.py`'s `generate_and_fit_toys_python` (and `binned.py`/`unbinned.py`'s grid-strategy toy
generators) never check whether a toy's own fit converged. `optimizers.py`'s
`_minimize_with_restarts` *does* warn on non-convergence (`scipy.optimize.minimize did not
converge...`), but that warning fires per-*fit*-call generically (data fits and toy fits
alike) -- it is not toy-generation-aware, does not discard anything, and does not trigger
regeneration of a replacement toy. Whatever `t_stat` comes out of a non-converged toy fit
(however poor) is used exactly like any other toy's result.

### What to do -- **this needs a decision from the repo owner before writing code**

Two very different resolutions, with very different risk profiles:

- **(a) Correct the documentation** to describe actual behavior (a console warning only,
  no discarding/regeneration) -- cheap, zero risk, purely honest.
- **(b) Actually implement toy discard-and-replace on non-convergence** -- this is a genuine
  new feature, in the same risk class as `adaptive_toys` from the prior fix round: it changes
  which toys contribute to the empirical critical-value distribution. Getting it wrong (e.g.
  silently biasing which toys survive) would be worse than not having it. If the repo owner
  wants (b), at minimum: thread `res.success` out of the toy-fit path in `toys.py` (currently
  discarded entirely -- `_worker_unbinned_toy` and `fit_single_toy` both call
  `unconditional_fit_scipy`/`conditional_fit_*_scipy` and only keep `.fun`, never inspecting
  success), decide what "regenerate" means (redraw the toy's random data and refit, up to some
  retry cap?), and validate with a dedicated comparison test (converged-only reference vs.
  discard-and-replace) analogous to `tests/test_adaptive_toys.py`'s verdict-matching tests,
  *before* considering it done.

**Recommendation**: do (a) first (cheap, immediate, honest), and treat (b) as a separate,
carefully-scoped follow-up if the repo owner actually wants the feature -- don't bundle a
risky new statistical feature into what should be a quick documentation fix. But ask first;
don't decide unilaterally, per the same discipline the prior brief used for `adaptive_toys`.

### Verification
- If (a): re-read the corrected paragraph aloud against `_minimize_with_restarts`'s actual
  behavior (one retry with a perturbed start, then a warning) to make sure the new wording is
  itself accurate.
- If (b): new tests proving no bias introduced (verdict-matching against a reference, per the
  pattern in `tests/test_adaptive_toys.py`), full `pytest tests/ -v`, `xbranch_compare` green.

---

## 2. Documentation bug: the "Binned Likelihood" formula shown doesn't match the code

### The problem (root-caused)

README.md's "Statistical Methodology & Mathematics" -> "Binned Likelihood" section, and the
identical duplicate in `docs/source/methodology.rst`'s "Binned Likelihood" section, both show:

$$-\ln \mathcal{L}_{\text{Poisson}} = \sum_{i=1}^{N} \left( \mu_i - n_i \ln \mu_i \right)$$

i.e. the simple, non-saturated, non-doubled Poisson NLL (dropping only the constant
`ln(n_i!)` term).

`binned.py`'s `calc_nll` **docstring itself** (grep `Baker-Cousins` in `pyfc/binned.py`) says:

```
1. Standard Poisson Likelihood (use_finite_mc = False):
   ...
   It uses the Baker-Cousins chi-square equivalent form (based on the likelihood
   ratio with a saturated model).
   Expression: NLL = 2 * SUM [ mu_i - n_i + n_i * ln(n_i / mu_i) ]
```

...and the actual code (the `else:` branch of `calc_nll`'s per-bin loop, `n_obs > 0` case)
computes exactly `2.0 * (mu_i - n_obs + n_obs * math.log(n_obs / mu_i))` -- confirmed by
direct reading, not just the docstring.

**These are genuinely different formulas** (saturated-and-doubled vs. simple-and-single),
differing by the additive, `mu`-independent term `2*n_i*ln(n_i) - 2*n_i` per bin. Because
`t_data = NLL_cond - NLL_uncond` is a **difference** evaluated at the same observed data `n_i`
in both the conditional and unconditional fits, that additive constant cancels exactly --
**so the FC interval construction itself is unaffected; this is a documentation-only bug, not
a statistical-correctness bug.** But `results["data_uncond_nll"]` (and any other absolute NLL
value PyFC reports) will not match what a reader computes by hand from the README's displayed
formula.

(For contrast: `unbinned.py`'s EUML formula in README/`methodology.rst` -- `-ln L_EUML =
N_expected - sum(ln(lambda(x_j)))` -- *does* match `calc_nll_unbinned`'s own docstring exactly.
Only the binned-Poisson formula has this problem.)

### What to do

Update both README.md and `docs/source/methodology.rst`'s "Binned Likelihood" formula to the
saturated, doubled form, matching `calc_nll`'s docstring exactly:

$$-\ln \mathcal{L}_{\text{Poisson}} = \sum_{i=1}^{N} \left( \mu_i - n_i + n_i \ln\frac{n_i}{\mu_i} \right)$$

with a brief added note that PyFC's implementation carries an extra overall factor of 2 (i.e.
computes $-2\ln\mathcal{L}$ directly, matching the Baker-Cousins/Wilks convention used for the
PLR test statistic $t$ itself, so no separate doubling is needed when forming $t = \text{NLL}_{\text{cond}} - \text{NLL}_{\text{uncond}}$)
-- follow the citation/precision style already established for the finite-MC correction
formula in the same section (added in the prior fix round) as a model for how much derivation
detail is appropriate.

### Verification
- Numerically spot-check: pick a toy `(mu_i, n_i)` pair, compute both the old and new
  formula's value by hand, and confirm the new one matches `calc_nll`'s actual return value
  for a single-bin model (a quick REPL check is enough, no new automated test strictly
  required since this is a docs-only change with no behavior change -- though a doctest-style
  comment showing the arithmetic, matching this repo's established "show your work" habit for
  formula-related fixes, would be a nice touch).
- `python scripts/check_changelog_sync.py` only relevant if a CHANGELOG entry is added (see
  the CHANGELOG guidance in the execution-order section below).
- Docs build clean (`cd docs && make html`, check no new warnings, `rm -rf build` after).

---

## 3. Missing config-system wiring: `n_restarts` and `neighbor_seeding`

### The problem (root-caused)

`compute_fc_intervals`'s signature includes `n_restarts=1` and `neighbor_seeding=True` --
both real, tested (`tests/test_restarts.py`), docstring-documented parameters (added in an
earlier "FIX 4" round predating even the deep-pre-push-inspection fixes). Confirmed via
exhaustive grep that **neither appears anywhere in the CLI/JSON config system**:

```bash
grep -n "n_restarts\|neighbor_seeding" pyfc/config.py pyfc/generate_config.py \
    README.md docs/source/configuration.rst config/example_fc_config.json
# zero hits in every one of those files
```

Contrast with every other tunable knob (`adaptive_toys`, `toy_batch_size`, `sparsify_grid`,
`warm_start`, `smooth_1d`, etc.), all of which are exposed via: an `argparse` entry in
`config.py`'s `parse_arguments()`, an entry in both of `config.py`'s base config dicts, a
question in `generate_config.py`'s interactive wizard, a row in README's "Configuration
Parameters (CLI / JSON)" table, a row in `docs/source/configuration.rst`'s mirroring table,
and a key in `config/example_fc_config.json`. `n_restarts`/`neighbor_seeding` are the only two
`compute_fc_intervals` parameters (excluding inherently-non-serializable ones like
`compute_rates_func`, `bounds_func`, `constraints`, `pdf_components`, `data`, `grids`) missing
from every one of those six places -- reachable only via a direct Python call.

### What to do

Wire both through, matching the existing pattern exactly for a boolean/int pair like
`warm_start`/`toy_batch_size`:
1. `pyfc/config.py`: add `--n_restarts` (int, default 1, validate `> 0`) and
   `--neighbor_seeding` (bool via the existing `lambda x: str(x).lower() in [...]` pattern,
   default `True`) to `parse_arguments()`'s `argparse` calls, and add both keys (with those
   same defaults) to **both** occurrences of the base config dict in that file (matching the
   two-dict pattern already used for every other parameter there).
2. `pyfc/generate_config.py`: add two new numbered wizard questions (after the existing
   `sparsify_grid`/`toy_batch_size`-adjacent questions, or wherever reads naturally next to
   the other "algorithmic enhancement" questions), with defaults matching step 1.
3. README.md's "Configuration Parameters (CLI / JSON)" table: two new rows, styled like the
   `adaptive_toys`/`toy_batch_size` rows added in the prior fix round (including a short
   description of what each actually does and any caveats -- e.g. `neighbor_seeding` only
   affects the `strategy="scipy"` DATA fit, not toy fits, not other strategies; see its own
   existing docstring in `orchestrator.py` for the precise wording to draw from).
4. `docs/source/configuration.rst`: same two rows, mirroring README's.
5. `config/example_fc_config.json`: add both keys with their defaults, consistent with every
   other key already there.

### Verification
- `pytest tests/ -v` green (existing `test_restarts.py` tests should be unaffected -- they
  call `compute_fc_intervals`/`_minimize_with_restarts` directly, not through the config
  layer, so this is purely additive).
- Manually run `python -m pyfc.generate_config`, accept every default, confirm the written
  JSON includes both new keys with the expected default values.
- Confirm `python -m pyfc.orchestrator --n_restarts 3 --neighbor_seeding false` (or similar)
  round-trips correctly through `config.py`'s CLI parsing.

---

## 4. Missing documentation: `fc_results.npz`'s actual key set

### The problem (root-caused, precise)

The actual keys written by `_save_fc_archive` in `pyfc/orchestrator.py` (grep
`save_dict\[` in that file to enumerate exactly):

```
best_fit, data_uncond_nll, grid_p{i}, 1d_test_p{i}, 1d_t_data_p{i}, 1d_prof_params_p{i},
1d_t_critical_p{i}_{cl}, 1d_accepted_p{i}_{cl}, 2d_test_p{i}_{pair}, 2d_test_p{j}_{pair},
2d_t_data_{pair}, 2d_t_critical_{pair}_{cl}, 2d_accepted_{pair}_{cl}
```

README.md's "Available `.npz` Keys" table documents only: `grid_p{i}`, `1d_t_data_p{i}`,
`1d_t_critical_p{i}_{cl}`, `1d_accepted_p{i}_{cl}`, `2d_t_data_p{i}p{j}`,
`2d_t_critical_p{i}p{j}_{cl}`, `2d_accepted_p{i}p{j}_{cl}` -- **missing** `best_fit`,
`data_uncond_nll`, `1d_test_p{i}`, `1d_prof_params_p{i}`, and both `2d_test_p{i}_{pair}`/
`2d_test_p{j}_{pair}`.

`docs/source/outputs.rst`'s mirroring table is **even less complete**: only `grid_p{i}`,
`1d_t_data_p{i}`, `1d_accepted_p{i}_{cl}`, `2d_t_critical_p{i}p{j}_{cl}`,
`2d_accepted_p{i}p{j}_{cl}` -- missing everything README is missing, *plus*
`1d_t_critical_p{i}_{cl}` and `2d_t_data_p{i}p{j}`, both of which README does document.

Notably, `best_fit` and `data_uncond_nll` are the two scalars README's own prose (in the
"Stored Results & Custom Plotting" section, describing `fc_results.json`) calls "required for
cross-model ratio comparisons" -- yet neither appears in either `.npz` key table.

### What to do

Add the missing rows to both tables (README.md and `docs/source/outputs.rst`), making
`outputs.rst`'s table a complete mirror of README's (matching the pattern -- `outputs.rst` is
meant to be the Sphinx-rendered version of the same information). For `1d_test_p{i}`: note
that it duplicates `grid_p{i}`'s content exactly (see `orchestrator.py`:
`results[f"1d_test_p{p_idx+1}"] = grid_test` where `grid_test = grids[p_idx]`) -- worth a
one-line note explaining the redundancy rather than presenting it as a distinct array. Add
`best_fit` (shape `(n_params,)`, the global unconditional MLE) and `data_uncond_nll` (scalar)
as their own rows given their importance. Add `1d_prof_params_p{i}` (shape `(N, n_params)`,
the full profiled parameter vector at each 1D scan point) and the 2D test-point arrays.

### Verification
- Cross-check the final table against a fresh `grep -oE 'save_dict\[f?"[^]]*"\]'
  pyfc/orchestrator.py` to confirm every key is now listed exactly once in both tables.
- Docs build clean, no new warnings.

---

## 5. Stale documentation: README's CI section describes ~6 of 75 tests

### The problem

README.md's "Continuous Integration (CI)" section has a bulleted "The test suite executes the
following verifications" list with exactly six bullets: core imports & dependencies, I/O &
checkpointing integrity, multiprocessing serialization, JIT compilation hooks, optimizer
boundary clamping, statistical Asimov convergence. These map **exactly** to the six original
test files: `test_core.py`, `test_io.py`, `test_multiprocessing.py`,
`test_numba_compilation.py`, `test_optimizers.py`, `test_statistics.py`.

The suite has since grown to **17 files, 75 tests** (`ls tests/*.py | wc -l`; `pytest
tests/ --collect-only -q`, with `ultranest` installed). None of the following files -- which
between them cover most of this branch's actual statistical-correctness guarantees -- get any
mention: `test_adaptive_toys.py`, `test_bounds_func.py`, `test_constraints.py`,
`test_disconnected_intervals.py`, `test_finite_mc_likelihood.py`, `test_nd_binned.py`,
`test_pdf_components.py`, `test_restarts.py`, `test_smoothing.py`, `test_sparsify_grid.py`,
`test_ultranest_retry.py`.

### What to do

Rewrite the section to accurately reflect current coverage breadth without necessarily
listing all 17 files 1:1 (that would be a maintenance burden going forward) -- group by what
they establish, e.g.: core statistical correctness (smoothing/discontinuity handling, the
finite-MC formula, N-D/non-contiguous data, disconnected-interval reporting), optimizer
robustness (restarts, neighbor warm-starting, bounds_func/constraints, boundary clamping,
Asimov convergence), algorithmic features (sparsify_grid, adaptive_toys), infrastructure
(I/O/checkpointing, multiprocessing, JIT compilation, UltraNest retry). Consider whether a
per-category count ("N tests across M files") is more maintainable than an exhaustive list,
so this doesn't go stale again the next time a test file is added -- flag this maintainability
concern to the repo owner rather than silently picking an approach, since it's a real editorial
choice about how much detail this section should commit to keeping current.

### Verification
- Docs build clean.
- Sanity-read the rewritten section against the actual current test file list one more time
  before committing.

---

## 6. Stale File Tree listings: missing test files

### The problem

`ls tests/*.py` (17 files) vs. what's listed in each File Tree:

- README.md's File Tree is missing: `test_disconnected_intervals.py`,
  `test_finite_mc_likelihood.py`, `test_sparsify_grid.py`, `test_ultranest_retry.py`.
- `docs/source/installation.rst`'s File Tree is missing those same four, **plus**
  `test_adaptive_toys.py` (which README's tree does already have).

### What to do

Add the missing entries to both File Trees, following the exact one-line-comment style
already used for every other test file listed there (e.g. `test_smoothing.py            #
Tests for the smoothed unphysical-rate NLL penalty`). Suggested descriptions:
- `test_adaptive_toys.py` -- Tests for the validated adaptive_toys/toy_batch_size early-stopping behavior
- `test_disconnected_intervals.py` -- Tests for the contiguous-run helper and disconnected 1D interval reporting
- `test_finite_mc_likelihood.py` -- Hand-computed-reference tests for the finite-MC likelihood formula
- `test_sparsify_grid.py` -- Regression tests for the sparsify_grid boundary-refinement guard fix
- `test_ultranest_retry.py` -- Tests for the UltraNest internal-bug retry wrapper

While in there: this is also a good moment to double check both File Trees against `git
ls-files` one more time end to end (not just for `tests/`), since this audit did not
exhaustively diff every single line of both File Trees against the actual repo contents --
only the `tests/` subtree was cross-checked in detail.

### Verification
- `git ls-files tests/` compared line-by-line against both File Tree listings, zero
  discrepancies remaining.

---

## 7. Leftover dead code: three commented-out lines in `optimizers.py`

### The problem

`pyfc/optimizers.py`, lines 362, 476, and 592 (as of this audit -- re-grep
`# x0 = seed if seed is not None` to relocate if line numbers have shifted), inside
`unconditional_fit_scipy`, `conditional_fit_1d_scipy`, and `conditional_fit_2d_scipy`
respectively:

```python
    # x0 = seed if seed is not None else [(b[0] + b[1]) / 2.0 for b in resolved_bounds]
    if seed is not None:
        eps = 1e-8
        lb = np.array([b[0] for b in resolved_bounds]) + eps
        ...
```

Each is the old, pre-clipping `x0` assignment, left in as a comment directly above the code
that replaced it (the `np.clip`-based version, added to fix the boundary-clamping issue
covered by `tests/test_optimizers.py::test_scipy_boundary_clamping`).

### What to do

Delete all three commented-out lines. No behavior change -- purely dead-code removal.

### Verification
- `pytest tests/ -v` green (no behavior touched).
- `git diff` shows only three line deletions, nothing else.

---

## 8. Stale version headers: `generate_config.py` and `plotting.py`

### The problem

Every `pyfc/*.py` module has a `Created: v0.1.0 (July 24, 2026)` / `Last modified: vX.Y.Z`
docstring header. Confirmed via `grep -rn "Last modified:" pyfc/*.py`:

- `pyfc/generate_config.py` and `pyfc/plotting.py` both say `Last modified: v0.9.0`; every
  other module (`binned.py`, `unbinned.py`, `toys.py`, `optimizers.py`, `orchestrator.py`,
  `config.py`, `__init__.py`) says `v0.10.0`.
- `git log --oneline -- pyfc/generate_config.py` shows real edits during this branch's
  (unreleased) v0.10.0 development cycle (most recently this session's wizard-default fixes
  for `n_toys`/`smooth_1d`/`smooth_2d`/`save_log`, plus earlier `sparsify_grid` and
  `save_directory` changes) -- confirms this header is genuinely stale, not just cosmetically
  inconsistent.
- `pyfc/plotting.py`: `git merge-base --is-ancestor <0.10.0-bump-commit> <dead-code-removal-commit>`
  confirms a dead-code-removal edit to this file also happened chronologically during the
  v0.10.0 cycle. Lower-confidence/lower-priority than `generate_config.py`'s case, since that
  specific edit was a trivial cleanup -- repo owner's call whether trivial edits warrant a
  version bump in this header's convention.

### What to do

Bump `pyfc/generate_config.py`'s header to `Last modified: v0.10.0` (clear-cut). For
`pyfc/plotting.py`, ask the repo owner what "Last modified" is meant to track (any edit at
all, vs. only substantive logic changes) before deciding -- it's a minor, low-stakes
convention question, not worth guessing at.

### Verification
- Trivial -- just confirm no other module's header needs the same check (`grep -rn "Last
  modified:" pyfc/*.py` one more time after the edit, confirm consistency).

---

## 9. Leftover from this session: unused `pytest` import

### The problem

`tests/test_disconnected_intervals.py` (added during the prior fix round's Fix #6) imports
`pytest` at the top but never uses it anywhere in the file. Confirmed by both `pyflakes` and
`ruff check`:

```
pyflakes: tests/test_disconnected_intervals.py:17:1: 'pytest' imported but unused
ruff:     F401 [*] `pytest` imported but unused --> tests/test_disconnected_intervals.py:17:8 (auto-fixable)
```

### What to do

Remove the unused `import pytest` line. Trivial, `ruff check --fix` handles this
automatically if you want to use the tool rather than a manual edit (verify the diff only
touches this one import line).

### Verification
- `pytest tests/test_disconnected_intervals.py -v` still green.
- `ruff check tests/test_disconnected_intervals.py` no longer flags F401.

---

## 10. Cosmetic: orphaned "FIX 2"/"FIX 3"/"FIX 4" labels in test docstrings

### The problem

`tests/test_bounds_func.py` ("Tests for FIX 2: ..."), `tests/test_constraints.py` ("Tests for
FIX 3: ..."), and `tests/test_restarts.py` ("Tests for FIX 4: ...") all reference an internal
numbering scheme from a prior, unlogged development round (predating even the deep-pre-push-
inspection fixes #1-6, which used their own independent "Fix #1-#6" numbering in this same
branch -- the two numbering schemes are unrelated and both now coexist in the repo,
potentially confusing). Nothing in the current repo maps "FIX 2/3/4" to anything a reader can
trace.

### What to do (low priority, repo owner's call)

Either leave as-is (harmless, doesn't affect correctness) or reword the docstrings to describe
the feature being tested without the dangling numeric reference (e.g. "Tests for the
`bounds_func` hook..." dropping "FIX 2:"). If the repo owner wants this touched, it's a pure
docstring wording change with zero behavior impact -- safe to batch with other cosmetic fixes
in this brief.

### Verification
- None needed beyond a visual diff review if changed.

---

## 11-12. Minor code-quality notes flagged by `pyflakes` (correct code, non-obvious to a reader)

### 11. `orchestrator.py`'s `_find_contiguous_intervals`

`pyflakes` flags: `orchestrator.py:193:42: undefined name 'prev_val'`. This is a **false
positive** -- `prev_val` is assigned unconditionally at the end of every loop iteration
(`prev_val = val`, last line of the loop body), so by the time any code path reads it (the
`elif` branch, which requires `run_start is not None`, itself only set by an earlier
iteration), it is always defined. Verified correct by manual trace and by the 6 existing unit
tests in `tests/test_disconnected_intervals.py` (including empty/all-accepted/single-point/
edge-touching-run cases). Not a bug -- but worth a low-priority readability refactor (e.g.
initialize `prev_val = None` before the loop, or rewrite using `enumerate`/index-based logic)
so neither `pyflakes` nor a future human reader has to re-derive the same non-obvious
"assigned every iteration" invariant from scratch.

### 12. `optimizers.py`'s `_project_linear_constraint`

The `jac_free = None` followed by a conditional `def jac_free(free_p): ...` redefinition
(inside the `NonlinearConstraint` branch) is correct but an unusual enough pattern that
`pyflakes` flags "redefinition of unused 'jac_free' from line 154". No functional issue --
optional style cleanup only if the repo owner wants it (e.g. restructure as an `if/else`
assigning to two different local names, or a small helper function).

### What to do
Both are optional, low-priority readability improvements, not bugs. Fix only if the repo
owner wants the codebase to be clean under `pyflakes`/`ruff` for these specific two warnings
specifically (as distinct from the repo-wide ruff non-compliance question in finding 16,
which remains explicitly out of scope pending confirmation).

### Verification
- If touched: `pytest tests/ -v` green, `pyflakes pyfc/orchestrator.py pyfc/optimizers.py`
  shows neither warning anymore.

---

## 13. Test hygiene: `test_restarts.py` uses hardcoded `/tmp` paths

### The problem

`tests/test_restarts.py` writes to hardcoded absolute paths (`/tmp/pyfc_test_neighbor_seed_1d`,
`/tmp/pyfc_test_neighbor_seed_disabled`, `/tmp/pyfc_test_e2e_{neighbor_seeding}`) instead of
pytest's `tmp_path` fixture, unlike every newer test file in the suite (`test_sparsify_grid.py`,
`test_disconnected_intervals.py`, `test_adaptive_toys.py`, `test_nd_binned.py`, etc., all of
which use `tmp_path`). Not a correctness bug (each test passes `warm_start=False`, so stale
files from a prior run can't cause a false pass/fail), but inconsistent, not parallel-test-safe
(would collide under `pytest-xdist`), and doesn't get automatically cleaned up.

### What to do

Add a `tmp_path` fixture parameter to the three affected test functions
(`test_orchestrator_neighbor_seeding_passes_previous_point_as_seed`,
`test_orchestrator_neighbor_seeding_disabled_via_flag`,
`test_end_to_end_with_and_without_neighbor_seeding_no_plateau_or_regression`) and replace the
hardcoded path strings with `str(tmp_path / "...")`, matching the pattern already established
in the rest of the suite.

### Verification
- `pytest tests/test_restarts.py -v` green.
- Confirm no leftover `/tmp/pyfc_test_*` directories get created by a fresh test run
  (`ls /tmp/pyfc_test_* 2>/dev/null` should show nothing new after running the updated tests
  from a clean state).

---

## 14. Already-known, deliberately deferred (confirm scope before touching)

These were explicitly deferred by the repo owner in an earlier session (see
`docs/dev/FIXES_BRIEF_dev-no-templates.md`'s own context section, findings #10-11 of the
review that preceded it) and are repeated here **only for precise, up-to-date quantification**,
not as new asks. **Ask the repo owner explicitly whether "I want to fix everything you found"
is meant to override that specific, earlier, explicit deferral before touching either of
these** -- a blanket instruction in a later session shouldn't silently override a specific,
deliberate decision from an earlier one without confirming that's really what's intended.

- **Python version floor**: `pyproject.toml` claims `requires-python = ">=3.8"`, but
  `.github/workflows/pytest.yml`'s CI matrix only tests `["3.9", "3.10", "3.11"]` -- 3.8
  support is asserted but unverified.
- **Repo-wide `ruff` non-compliance**: `.github/workflows/lint.yml` runs both `ruff check` and
  `ruff format --check`, but **both steps have `continue-on-error: true`**, so the CI job is
  green regardless of lint status -- ruff compliance is currently informational only, not
  gated. Precise current snapshot (for reference, not necessarily meaningful by the time this
  is read, if other fixes in this brief are applied first): `ruff check pyfc/ tests/ --ignore
  BLE001,C414,SIM102,RUF059,PERF402,TRY201` finds **15 errors** (1 auto-fixable -- the same
  unused `pytest` import in finding 9 above); `ruff format --check .` reports **40 files**
  would be reformatted repo-wide (including all of `xbranch_compare/` and several test files).

---

## 15. Suggested execution order

Roughly cheapest/lowest-risk first, matching this repo's established one-commit-per-logical-
change discipline:

1. **Findings 7, 9** (dead code removal, unused import) -- trivial, zero risk, do first.
2. **Finding 8** (stale version headers) -- trivial, but ask the one `plotting.py` question
   inline early so it doesn't block anything.
3. **Finding 6** (File Tree test listings) -- mechanical, and worth doing before finding 5
   since an accurate File Tree is a natural reference while rewriting the CI section.
4. **Finding 5** (CI section rewrite) -- ask the one editorial question (exhaustive list vs.
   category counts) early.
5. **Finding 4** (`.npz` key tables) -- mechanical, self-contained.
6. **Finding 2** (binned-likelihood formula) -- self-contained, moderate care needed to get the
   corrected formula/wording exactly right.
7. **Finding 3** (`n_restarts`/`neighbor_seeding` config wiring) -- moderate effort, touches
   five files, but entirely mechanical (matches an existing, well-established pattern).
8. **Finding 13** (`test_restarts.py` tmp_path) -- mechanical, self-contained.
9. **Findings 10, 11, 12** (cosmetic/optional) -- batch together if the repo owner wants them
   at all; skip freely otherwise.
10. **Finding 1** (toy-replacement claim) -- do this **last**, since it's the one item with a
    real open design question (document vs. implement) that needs a repo owner decision before
    any code is written, and the "document" resolution alone is cheap/fast once decided.
11. **Finding 14** (Python floor / ruff) -- do **not** touch without first explicitly
    confirming scope with the repo owner, per the note in that section.

After all applicable findings: full final verification pass -- `pytest tests/ -v` (both the
base environment and `PYTHONPATH=/home/mbustamante/Research/FeldmanCousins
/home/mbustamante/anaconda3/envs/py310/bin/python -m pytest tests/ -v`, since only the `py310`
env has `ultranest` installed), `bash xbranch_compare/run_comparison.sh` (zero mismatches),
`python scripts/check_changelog_sync.py`, a docs build (`cd docs && make html`, check no new
warnings, `rm -rf build` after), `git worktree list` (clean), and a final `git status --short`
review. Add CHANGELOG.md / `docs/source/changelog.rst` entries for anything that changes
user-visible behavior or documentation of substance (the config-wiring finding #3 and the
formula fix #2 both plausibly warrant an entry each, under "Fixed"; pure dead-code/typo-level
fixes like #6-#9 probably don't need their own entries, but use judgment against this
release's existing entries as precedent).

## 16. Final acceptance checklist

- [ ] Finding 1: toy-replacement claim resolved (documented honestly, or implemented and
      validated) -- repo owner's decision obtained first.
- [ ] Finding 2: binned-likelihood formula in README.md and `methodology.rst` corrected to
      match `calc_nll`'s actual (saturated, doubled) implementation.
- [ ] Finding 3: `n_restarts`/`neighbor_seeding` wired through `config.py`,
      `generate_config.py`, README's parameter table, `configuration.rst`, and
      `example_fc_config.json`.
- [ ] Finding 4: `fc_results.npz`'s full key set documented in both README.md and
      `docs/source/outputs.rst`.
- [ ] Finding 5: README's CI section rewritten to reflect actual (75-test, 17-file) coverage.
- [ ] Finding 6: File Tree test listings complete in both README.md and
      `docs/source/installation.rst`.
- [ ] Finding 7: dead commented-out code removed from `optimizers.py`.
- [ ] Finding 8: `generate_config.py`'s version header bumped to v0.10.0; `plotting.py`'s
      resolved per repo owner's answer.
- [ ] Finding 9: unused `pytest` import removed from `test_disconnected_intervals.py`.
- [ ] Findings 10-12: resolved per repo owner's preference (fix or explicitly skip).
- [ ] Finding 13: `test_restarts.py` uses `tmp_path` instead of hardcoded `/tmp` paths.
- [ ] Finding 14: scope explicitly confirmed with repo owner before any action; untouched
      otherwise.
- [ ] `pytest tests/ -v` green in both environments.
- [ ] `bash xbranch_compare/run_comparison.sh` shows zero mismatches.
- [ ] `python scripts/check_changelog_sync.py` passes.
- [ ] Docs build clean, no new warnings.
- [ ] `git worktree list` clean.
- [ ] `git status --short` reviewed, nothing unexpected left uncommitted or untracked.
- [ ] CHANGELOG.md / `docs/source/changelog.rst` updated for anything substantive.
- [ ] Ask the repo owner whether to archive this brief into `docs/dev/` once done, per the
      established precedent (see section 0 above).
