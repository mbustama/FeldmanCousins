# Brief: Implement fixes #1-#6 from the deep pre-push inspection

**Purpose of this document**: handoff brief for a fresh chat session, to be pasted or
`@`-referenced at the start of that session. Written by an assistant that had just
completed a "final deep inspection" of PyFC's `dev-no-templates` branch (statistical
correctness, not just code hygiene) immediately before running low on context budget.
The repo owner (Mauricio Bustamante) reviewed that inspection's findings and gave exact
instructions for fixing all six. **None of the six fixes below have been implemented
yet** -- this document exists specifically because the prior session judged its
remaining context (~21%) too risky to spend on six fixes of this depth, several of
which (#2, #5, #6) involve real statistical/design judgment where a rushed, wrong
implementation would be *worse* than the current state (silently-wrong output vs.
obviously-incomplete output).

Repository: `/home/mbustamante/Research/FeldmanCousins` (PyFC, GPLv3, sole author is the
repo owner). Branch: **`dev-no-templates`** (already exists, already has everything
described below as "already done"). Continue working directly on this branch. Do
**not** touch `main` or `dev`. This branch has not been pushed to GitHub yet -- this is
still pre-push cleanup.

---

## 0. Context: what already happened on this branch (do NOT redo this)

This branch already contains (chronologically):

1. A large refactor removing the `S_model`/`B_model` templating mechanism (replaced by
   closure-captured templates + a general `pdf_components` list), adding N-D binned data
   support, and bumping `pyproject.toml` to `version = "0.10.0"` (unreleased).
2. A full-codebase review (repo owner requested) that produced 11 findings; #1-9 were
   fixed (a non-contiguous-array crash fix in `calc_nll`, a docs/gitignore/comment
   cleanup pass, docstring `Date:` -> `Created:`/`Last modified:` conversion, etc.);
   #10-11 (Python-version-floor claim, repo-wide ruff non-compliance) were explicitly
   deferred by the repo owner and remain **out of scope** -- do not touch them.
3. A `scripts/check_changelog_sync.py` + `.github/workflows/changelog-sync.yml` CI check
   that fails the build if `CHANGELOG.md` and `docs/source/changelog.rst` structurally
   drift (see that script's own module docstring for exactly what it checks). **Any
   change to either changelog file in this session must keep them in sync** -- run
   `python scripts/check_changelog_sync.py` after editing either, before committing.
4. A dedicated example notebook (`examples/pyfc_non_contiguous_data_tutorial.ipynb`)
   plus end-to-end regression tests for the non-contiguous-array fix.
5. A "final checks" round (12 checks, A1-E12) that found and fixed: a Python-3.12-only
   f-string in `pyfc_high_dimensional_tutorial.ipynb` (broke on the project's own
   3.9-3.11 CI matrix), missing entries in README's/`installation.rst`'s File Tree
   sections, and **a real, previously-unknown crash**: `strategy="ultranest"` for 2D
   intervals hit a still-open bug inside `ultranest.ReactiveNestedSampler`'s own
   `_find_strategy` method (confirmed present in v4.5.0 and current GitHub master, so
   upgrading doesn't help). Fixed with a retry-with-fresh-sampler wrapper
   (`_run_ultranest_with_retry` in `pyfc/optimizers.py`, tests in
   `tests/test_ultranest_retry.py`). Also found `pyfc_strategy_comparison_tutorial.ipynb`
   had apparently never been run end-to-end with `ultranest` actually installed (its own
   committed output used to read "ultranest not installed -- skipping"); its grid/n_toys
   were scaled down so it actually completes.
6. Also from that round: `save_directory`'s default was unified across three previously-
   inconsistent locations (`compute_fc_intervals`'s own kwarg default, `config.py`'s
   CLI/JSON default, `generate_config.py`'s wizard default) to
   `"output/example_fc_output"`. **This established a pattern** -- fixes #1 and #3 below
   are the same class of bug, found by systematically checking *every* parameter this
   way afterward.
7. A "final deep inspection" (this is the one that produced findings #1-#6 below) that
   went past docs/tests hygiene into re-deriving the actual statistical formulas and
   tracing the actual control flow of the orchestration/optimizer code. This is what
   found the six issues below.

**Current state**: full test suite green (`pytest tests/ -v` -- 50 tests when run under
an environment with `ultranest` installed, 47 + 3 skipped otherwise), cross-branch
harness passes with zero mismatches, `git worktree list` is clean, `git status --short`
is clean except this brief file itself (untracked, same situation the repo owner's own
`FOLLOWUP_BRIEF_dev-no-templates.md` was in when *that* handoff started this whole
process -- see `docs/dev/` for the precedent of what happened to that file, in case it's
relevant to what should happen to this one too. Ask, don't assume, per that precedent).

---

## 1. Fix #1: Make `sparsify_grid=False` the default

### The bug this fixes (already root-caused, not yet fixed)

`pyfc/orchestrator.py`'s `eval_2d_point` (~line 649, inside Phase 2's 2D-interval loop)
has a boundary-refinement pass that's supposed to exactly re-evaluate (via real toys)
grid cells near the coarse-interpolated contour boundary. **It never actually runs.**
Its guard is:
```python
if not np.isnan(results[f"2d_t_critical_{pair_name}"][cl[0]][i, j]):
    return
```
...meant to skip cells already exactly evaluated. But a few lines later, the
coarse-to-fine interpolation step does:
```python
results[f"2d_t_critical_{pair_name}"][c] = interp(gridA, gridB)
```
...which fills in **every** cell (not just coarse ones) with a non-NaN value. So by the
time the refinement pass calls `eval_2d_point` on the boundary cells it identified,
every one already has a non-NaN `2d_t_critical` (from interpolation, not real
evaluation) -- the guard fires, and refinement is a silent no-op.

**Confirmed by direct reproduction** (3-param binned model, 16-point grid, matching the
`pyfc_algorithmic_features_tutorial.ipynb` notebook's own sparsify_grid demo): of a
16x16=256-cell grid, only the 36 coarse points ever get `2d_t_data` computed; the other
220 stay `NaN`. Since `accepted = (t_data <= t_critical)` and any comparison against
`NaN` is `False` in numpy, **220/256 cells are automatically excluded regardless of
their true classification** -- accepted count was 8/256 instead of the true (much
larger) accepted region.

`compute_fc_intervals`'s own signature default is currently `sparsify_grid=True` *and*
`compute_2D_intervals=True` -- so anyone computing 2D intervals without explicitly
passing `sparsify_grid=False` silently gets wrong output, by default. Zero test
coverage exists for `sparsify_grid=True` (every test in the suite explicitly passes
`False`).

### What the repo owner asked for (verbatim, this is the scope)

> make sparsify_grid=False the default. For everywhere in the docstrings, comments,
> README, and docs where needed.

This is a **default-value change + doc sweep**, not (necessarily) a fix to the
underlying `eval_2d_point` bug itself. Read that distinction carefully:

- `pyfc/config.py`'s CLI/JSON default is **already** `False` (confirmed, both
  occurrences of the base config dict).
- README.md's parameter table (~line 359) and `docs/source/configuration.rst`'s table
  (~line 68-71) **already** document the default as `False`.
- The **only** place the wrong value (`True`) lives is `compute_fc_intervals`'s own
  Python keyword default in `pyfc/orchestrator.py` (~line 246):
  `sparsify_grid=True, warm_start=True,` -> change to `sparsify_grid=False,`.

So this fix may be as small as **one line**, since the docs already say the right thing.
Still do a full sweep to be sure nothing else claims `True`:
```bash
grep -rn "sparsify_grid" pyfc/ README.md docs/source/*.rst examples/*.ipynb
```
Check in particular: (a) `orchestrator.py`'s own docstring line documenting this
parameter (~line 312, currently just says "bool/int" generically -- fine as-is, but
re-verify), (b) whether any example notebook relies on the *implicit* default (i.e.
calls `compute_fc_intervals` with `compute_2D_intervals=True` but omits `sparsify_grid`
entirely, expecting the old `True` behavior) -- if so, that notebook's behavior/timing
will change and it should be re-executed to confirm it still produces sensible,
non-degenerate output (and its embedded outputs re-committed if they change materially,
following the same pattern used for the ultranest-strategy-comparison notebook fix
above -- re-execute with
`PYTHONPATH=/home/mbustamante/Research/FeldmanCousins jupyter nbconvert --to notebook
--execute --inplace <notebook>`, using the `/home/mbustamante/anaconda3/envs/py310/bin/`
interpreter if the notebook exercises `ultranest`, since the registered Jupyter kernel
on this machine only has `ultranest` installed there, not in the base env).

### Open question -- ask the repo owner, don't decide unilaterally

The repo owner's instruction is scoped to the *default value*. It does **not** say to
fix the underlying `eval_2d_point` refinement bug itself, which would still silently
produce wrong output for anyone who *explicitly* opts into `sparsify_grid=True`.
Leaving a known-broken opt-in feature in the codebase (even if off by default) is not
great, but it's explicitly a bigger, separate piece of work (redesigning the
memoization guard to key off `2d_t_data`'s NaN-ness instead of `2d_t_critical`'s, since
`2d_t_data` is never touched by interpolation and accurately reflects "has this point
actually been toy-evaluated"). **Ask the repo owner whether they also want the
underlying bug fixed in this pass, or whether flipping the default is sufficient
mitigation for now** (with maybe a code comment / docstring warning that
`sparsify_grid=True` is currently known-broken, referencing this brief or a tracking
note, if they want to defer the deeper fix).

### Verification

- `pytest tests/ -v` green.
- `python scripts/check_changelog_sync.py` if a CHANGELOG entry is added (recommended --
  this is a real, if narrow, default-behavior fix; follow the "Fixed" section pattern
  already established in `CHANGELOG.md`'s `[0.10.0]` entry for the `save_directory`
  unification, which is the most similar precedent).
- Re-execute any notebook whose output changes as a result.

---

## 2. Fix #2: Make `adaptive_toys` and `toy_batch_size` functional

### The bug this fixes (already root-caused, not yet fixed)

Both are real parameters in `compute_fc_intervals`'s signature, documented in
README.md's parameter table, and threaded through `config.py`/`generate_config.py`'s
CLI/JSON/interactive-wizard machinery -- but grep for either name across all of `pyfc/`
turns up exactly 4 occurrences each, all just signature-default/docstring-mention/
pass-through. **Neither is ever read to make an actual decision anywhere.**

What the docs claim vs. reality:
- README.md: *"`adaptive_toys` -- Dynamically stops toy generation early if a grid
  point is definitively excluded, saving compute time."* -- does not happen.
- README.md: *"`toy_batch_size` -- Chunk size for batched array generation (optimizes
  memory/speed)."* -- there is no batching anywhere in `toys.py`/`binned.py`/
  `unbinned.py`.
- **README.md ~line 483**, specifically: *"DO monitor memory scaling for unbinned
  analyses... If this occurs, reduce `toy_batch_size` from 200 to 50."* This is
  concrete troubleshooting advice for OOM crashes on shared HPC/Slurm clusters. It
  would have **zero effect**.

### What the repo owner asked for (verbatim)

> please make adaptive_toys and toy_batch_size functional.

### This is the highest-design-risk fix of the six. Read this whole section before touching code.

A **wrong** implementation of adaptive early-stopping is worse than the current
do-nothing state: it would silently bias the empirical critical-value distribution
(and therefore the confidence intervals themselves) rather than just being slow. Do not
ship a stopping rule you haven't validated numerically against a non-adaptive reference.

**Where the toy-generation loops actually live** (all three need `toy_batch_size`
plumbed through; `adaptive_toys` conceptually applies to all three too, though the
`numba`-`@njit`/`parallel=True` grid-strategy generators are the least natural fit for
early-exit logic since `prange` loops parallelize the whole batch at once):
1. `pyfc/binned.py`: `generate_and_fit_toys_grid_1d`/`generate_and_fit_toys_grid_2d`
   (the `strategy="grid"` toy generators, `@njit(parallel=True)`, loop via `prange`).
2. `pyfc/unbinned.py`: `generate_and_fit_toys_grid_unbinned_1d`/`_2d` (same, but plain
   Python, sequential `for` loop, not `prange` -- easier to add early-exit logic to).
3. `pyfc/toys.py`: `generate_and_fit_toys_python` (the `strategy in
   ("scipy","ultranest","hybrid")` path; binned toys via `ThreadPoolExecutor`, unbinned
   toys via `ProcessPoolExecutor` through the module-level `_worker_unbinned_toy`).

**Suggested design for `toy_batch_size`** (do this part first -- it's lower-risk, no
statistics change, same total `n_toys` always run, just processed/held in memory in
chunks of `toy_batch_size` instead of all at once):
- In `toys.py`'s `generate_and_fit_toys_python`: instead of building the full
  `args_list`/dispatching all `n_toys` tasks to the executor in one `executor.map`
  call, loop over batches of size `toy_batch_size`, submitting/collecting each batch
  before starting the next. This directly addresses the README's OOM-avoidance claim
  (bounds how many toy event arrays / `ProcessPoolExecutor` result objects are alive
  at once for `unbinned` analyses, which is exactly the scenario README's OOM note
  describes).
- In `binned.py`'s grid-strategy generators (`@njit(parallel=True)`): batching inside a
  `numba`-compiled `prange` loop is awkward -- consider whether `toy_batch_size` should
  simply be a no-op there (binned toy arrays are small scalars per bin, not large
  kinematic arrays, so the OOM concern README describes is specifically about
  *unbinned* analyses) and document that explicitly, rather than forcing a batching
  scheme onto a code path that doesn't have the memory problem it's meant to solve.

**Suggested design for `adaptive_toys`** (do this second, and validate carefully):
- A statistically defensible sequential-stopping approach: after each batch of
  `toy_batch_size` toys, compute the running fraction of toys with
  `t_toy >= t_data` (this is the running p-value estimate for that grid point's
  accept/reject decision at the target CL). Compute a binomial confidence interval on
  this proportion (e.g. Wilson score interval) around the target `alpha = 1 - CL`
  threshold, at a *strict* confidence level (e.g. 99.9%, not 95% -- the cost of a wrong
  early stop is a biased confidence interval, so bias the stopping rule itself toward
  caution). If that confidence interval lies entirely on one side of `alpha`, the
  accept/reject verdict cannot plausibly flip with more toys -- stop early. Otherwise,
  keep going up to `n_toys`.
- Enforce a **minimum toy count** before early-stopping is even considered (e.g. never
  stop before ~100 toys, regardless of how extreme the running proportion looks early
  on -- small-sample binomial CIs are unreliable).
- This only meaningfully saves time for grid points *far* from the boundary (deep
  inside or deep outside the confidence region) -- points near the boundary will and
  should run the full `n_toys`, which is exactly the desired behavior (matches
  README's own framing: "if a grid point is *definitively* excluded").

**Validation before considering this done** (do not skip):
- Add a test comparing `adaptive_toys=True` vs `adaptive_toys=False` on the same
  well-inside and well-outside grid points (fixed seed), confirming both reach the
  *same* accept/reject verdict.
- Add an end-to-end comparison: run a full `compute_fc_intervals` with a large
  `n_toys` reference (`adaptive_toys=False`) vs. `adaptive_toys=True` with the same
  `n_toys` cap, and confirm the resulting confidence intervals match closely (this is
  the real test of "did the stopping rule introduce bias").
- If you're not confident the stopping rule is statistically sound after implementing
  and testing it, **stop and ask the repo owner** rather than shipping something that
  could silently produce wrong scientific results -- that's a strictly worse outcome
  than leaving these two parameters as documented-but-unimplemented, which is at least
  honestly "doesn't do anything" rather than "does the wrong thing."

### Verification

- New tests as described above.
- Full `pytest tests/ -v` still green.
- `xbranch_compare/run_comparison.sh` still shows zero mismatches (this proves the
  *default* behavior, i.e. `adaptive_toys`/`toy_batch_size` at whatever default you
  land on, doesn't change numerics vs. the reference implementation for the toy
  scenarios the harness covers -- adjust the harness's scenarios if needed to actually
  exercise the new code paths meaningfully, or add a new scenario).
- Update README's `adaptive_toys`/`toy_batch_size` table rows and the OOM-troubleshooting
  paragraph (~line 483) to accurately describe what actually happens now.
- CHANGELOG entry required (this is real new functionality, not just a fix -- consider
  whether it belongs under "Fixed" since the docs already claimed this happened, or
  "Added" since it's newly-real functionality; probably "Fixed" reads more honestly
  given the framing of this whole brief).

---

## 3. Fix #3: Make `compute_fc_intervals`'s defaults match README.md's documented defaults

### The mismatches (already enumerated, not yet fixed)

Checked every default in `compute_fc_intervals`'s signature (`pyfc/orchestrator.py`,
~line 242-252) against `pyfc/config.py`'s base config dict (which is what README.md's
and `configuration.rst`'s parameter tables both document as "the" default):

| Parameter | `compute_fc_intervals`'s own default | README-documented default |
|---|---|---|
| `n_toys` | `2000` | `500` |
| `sparsify_grid` | `True` | `False` (this is fix #1, above) |
| `save_log` | `False` | `True` |
| `smooth_1d` | `False` | `True` |
| `smooth_2d` | `False` | `True` |
| `cl` (`None` resolves to, ~line 406-407: `if cl is None: cl = [0.90]`) | `[0.90]` | `[0.68, 0.90]` |

### What the repo owner asked for (verbatim)

> the default vlaues of `compute_fc_intervals` should be the same as the ones in
> README.md. Please fi[x].

### Judgment call flagged for you (recommend resolving, don't just silently pick one)

Three more parameters resolve differently when left as their sentinel value, but these
read as *intentional* "auto-detect / opt-out" behavior, not straightforward value bugs:
- `num_cores=None` -> passed straight to `ThreadPoolExecutor`/`ProcessPoolExecutor`,
  which interpret `None` as "use `os.cpu_count()`" (auto-detect). README's table says
  the default is `8`. Hardcoding `8` would *remove* the auto-detect behavior and could
  under/over-subscribe whatever machine the code actually runs on.
- `output_file=None` -> (verify exact behavior in the function body, but this almost
  certainly means "don't write a JSON file") vs. README's `"fc_results"` (always
  write). Hardcoding this would change save-to-disk behavior for anyone currently
  relying on `None` to mean "just give me the in-memory `results` dict."
- `param_names=None` -> auto-generates `param1`, `param2`, ... matching `n_params` vs.
  README's literal `["param1", "param2", "param3"]`. **Hardcoding this one would
  actively break any model with `n_params != 3`** -- this is not a safe default to
  copy verbatim; README's table value is specifically a config-file *example*, not a
  sane universal default.

**Recommendation**: fix the 6 concrete-value mismatches in the table above (mechanical,
low-risk), and leave `num_cores`/`output_file`/`param_names` as `None`-sentinels,
documenting clearly in the docstring what `None` resolves to for each (this itself
partially existed already for `save_directory` before that fix -- follow the same
docstring-clarification pattern). If you want to honor the repo owner's literal
wording more strictly, **ask them to confirm** this carve-out before deciding
unilaterally, since it's a real interpretation gap in their instruction.

### Verification

- `pytest tests/ -v` green (double-check no test relies on the *old* wrong defaults by
  omitting these params and expecting the current behavior -- grep
  `compute_fc_intervals(` calls across `tests/*.py` for any that omit `n_toys`,
  `save_log`, `smooth_1d`, `smooth_2d`, or `cl` and would be affected).
- Any example notebook that omits these parameters (relying on the default) needs
  re-execution to confirm sensible output and updated embedded outputs if they change.
- CHANGELOG entry (fold into the same entry as fix #1 if convenient, since both are
  "defaults didn't match documentation" fixes discovered the same way).

---

## 4. Fix #4: Protect against the ~1e-12-wide discontinuity in the unphysical-region smoothing

### The issue this fixes (already root-caused, not yet fixed)

In `binned.py`'s `calc_nll` and `unbinned.py`'s `calc_nll_unbinned`, for a bin/event
with real observed data (`n_obs > 0` / event present), the *physical*-branch NLL
formula diverges to `+infinity` as `mu_i -> 0+` (correct -- a near-zero expected rate
with data actually observed should be heavily disfavored). But the *unphysical* branch
(triggered the instant `mu_i <= 0`) evaluates the base NLL term at a **fixed** floor
(`mu_floor = 1e-12`), not at the actual `mu_i` -- so it returns a comparatively small,
roughly-constant value regardless of how close to zero `mu_i` actually is. There's a
razor-thin region (empirically, for `n_obs=5`, roughly `mu_i` in `(1e-12, ~3e-12)`)
where crossing from `mu_i = +epsilon` to `mu_i = -epsilon` causes the NLL to *drop*
rather than continue climbing -- the opposite of the smoothing fix's whole purpose.

In practice this is very unlikely to be hit (no realistic optimizer step lands within
`1e-12` of exactly zero), which is why this was filed as "medium," not "critical." The
repo owner wants it fixed anyway, defensively.

### What the repo owner asked for (verbatim)

> add protection against this, just in case, even if is unlikely to be triggered.

### Recommended fix (designed, not yet implemented or numerically verified)

Change the branch condition from `if mu_i <= 0:` to `if mu_i <= mu_floor:` (i.e.
extend the "unphysical treatment" to also cover the tiny physical sliver
`(0, mu_floor)`), so there's a single, consistent transition point at `mu_i = mu_floor`
approached from *both* directions:
- For `mu_i > mu_floor`: real physical Poisson formula (unchanged).
- For `mu_i <= mu_floor` (whether a tiny-but-still-positive sliver, exactly zero, or
  negative): floor+barrier formula (unchanged formula, just a wider trigger range).

At `mu_i = mu_floor` exactly, both branches already agree (barrier term is `0` there
by construction) -- extending the trigger condition to `mu_i <= mu_floor` removes the
sliver where the diverging physical formula and the floor-based formula would otherwise
disagree, without changing either formula's actual expression. `mu_floor = 1e-12` is
astronomically smaller than any physically meaningful `mu_i`, so this should have zero
practical effect on real fits -- but verify this claim numerically (see below) before
considering it done, don't just trust the derivation.

Apply the analogous change in `unbinned.py`'s `calc_nll_unbinned`: the trigger is
currently `unphysical = p_events <= 0`; change to `unphysical = p_events <= p_floor`
(where `p_floor = 1e-12`, already defined a few lines below -- move its definition up
or restructure so it's available at the comparison).

Double-check the `n_obs == 0` (binned) / no-event (unbinned) edge case still behaves
sensibly after widening the trigger range -- for `n_obs == 0` and `mu_i` in the tiny
sliver `(0, mu_floor)`, the physical branch currently contributes `2.0 * mu_i`
(a negligible ~`2e-12`); after this change it would instead hit the "continue" no-op
(contributes `0`). This difference is utterly negligible (verify) but worth a one-line
comment noting the deliberate tradeoff.

### Verification (required, not optional)

- Write a new test (extend `tests/test_smoothing.py`) that explicitly sweeps `mu_i`
  (or `p_events`) across the transition at fine resolution (e.g. `mu_floor * 0.5` to
  `mu_floor * 2.0`, or a coarser sweep from `+1e-9` to `-1e-9`) and asserts the NLL
  sequence is monotonically non-decreasing throughout -- i.e. no drop anywhere,
  including right at the old discontinuity point. This directly tests the bug this fix
  targets, not just the pre-existing smoothness test that already covers the *rest* of
  the unphysical region.
- Re-run the existing `test_binned_calc_nll_identical_for_physical_bins` /
  `test_unbinned_calc_nll_identical_for_physical_events` (the "byte-for-byte...
  practically identical" backward-compat tests) -- confirm they still pass unchanged
  (they test `mu_i`/`p_events` values well away from the floor, so should be
  unaffected, but verify).
- Full `pytest tests/ -v` and `xbranch_compare/run_comparison.sh` both green.
- CHANGELOG entry under "Fixed."

---

## 5. Fix #5: Verify/correct the finite-MC `alpha = mu^2/sigma^2 + 1` formula against the reference paper

### The question this resolves (flagged, not yet resolved -- this needs the paper, which the prior session did not read)

`binned.py`'s `calc_nll`, finite-MC (Poisson-Gamma mixture) branch:
```python
alpha = (mu_i**2) / sigma2 + 1.0
beta = max(mu_i / sigma2, 1e-300)
lnL = (alpha * math.log(beta)
       + math.lgamma(n_obs + alpha)
       - (n_obs + alpha) * math.log(1.0 + beta)
       - math.lgamma(alpha))
nll += -2.0 * lnL
```
This is internally consistent with its own docstring (both say `alpha = mu^2/sigma^2 +
1`), so it's not a code-vs-docstring bug -- the question is whether this is the
*statistically correct* formula. A standard mean-and-variance-matching derivation
(`E[lambda]=mu`, `Var[lambda]=sigma^2` for a Gamma-distributed rate `lambda`) gives
`alpha = mu^2/sigma^2` with **no** `+1` term. The prior session could not verify
whether the `+1` is a deliberate, named convention (this formula has been unchanged
since `v0.1.0`, i.e. it predates the current refactor entirely) without a primary
source, and did not have time to research it further.

### What the repo owner asked for (verbatim)

> I will be giving you the paper from where the finite-MC estimate was extracted, so
> you can use it to fix the code if needed.

**The paper**: `/home/mbustamante/Downloads/1901.04645v2.pdf` (19 pages). This file
should still be at that path in the new session -- read it directly
(`pages: "1-19"` fits under the 20-page-per-request limit for large PDFs).

### What to do

1. Read the paper. Find the section deriving the Poisson-Gamma mixture / Negative
   Binomial correction for finite Monte Carlo statistics in a template/histogram
   likelihood (this is a well-known class of correction in HEP -- Barlow-Beeston,
   Conway, and others have published variants; identify which one this paper presents
   and what its exact `alpha`/`beta` (or equivalent shape/rate) parameterization is in
   terms of the expected count `mu` and its MC variance `sigma^2`).
2. Compare directly against the code above. If the paper's formula matches exactly
   (including the `+1`, if that's genuinely part of this paper's specific convention),
   **no code change is needed** -- but *do* add a citation to the paper (e.g. in
   `calc_nll`'s docstring, near the "Finite Monte Carlo Likelihood" section) so this
   provenance is preserved and the next person auditing this code doesn't have to
   re-derive it from scratch. Consider adding the paper's citation to
   `docs/source/refs.bib`/`references.rst` too, matching how other methodology
   citations are handled in this project (check `docs/source/methodology.rst` for the
   existing citation style/pattern first).
3. If the paper's formula **differs** from the code, this is a genuine behavior-
   changing bug fix:
   - Update `calc_nll`'s implementation (the `alpha`/`beta` computation) to match the
     paper.
   - Update `calc_nll`'s docstring math block (the "Finite Monte Carlo Likelihood"
     section, ~line 57-65) to state the corrected formula, with the citation.
   - Check whether README.md's math section duplicates this formula (it has a
     "Poisson-Gamma" or similar section somewhere in its statistical-methodology
     content -- grep for `alpha` or `Gamma` in README.md) and `docs/source/
     methodology.rst` likewise; update both if so.
   - Search `tests/test_smoothing.py`, `tests/test_statistics.py`, and any other test
     using `use_finite_mc=True` for hard-coded expected NLL values computed from the
     *old* formula -- these will need updating to match the corrected formula (or
     ideally, add a *new* test with a hand-computed reference NLL using the paper's
     exact formula on a small worked example, matching the pattern already established
     in `tests/test_pdf_components.py`'s "hand-computed reference NLL" tests).
   - This is a **behavior-changing fix to already-shipped-in-this-branch functionality**
     (the finite-MC correction has existed since v0.1.0) -- it needs a clear
     "BREAKING CHANGES" or "Fixed" CHANGELOG entry (probably "Fixed," since it's
     correcting a bug, not intentionally changing an API) explicitly warning that
     **NLL values from the finite-MC-corrected likelihood will differ from all prior
     runs** -- mirror the wording style already used for the smoothing-fix's own
     "NLL values in the previously-unphysical region will differ from prior runs"
     warning in the same CHANGELOG entry, including the `.. warning::` RST admonition
     in `docs/source/changelog.rst` (see that file for the existing pattern to copy).

### Verification

- New/updated tests with a hand-computed reference NLL matching the paper's formula
  exactly (show your work in a code comment -- this is exactly the kind of formula
  that's easy to get subtly wrong twice in a row without an explicit numeric check).
- Full `pytest tests/ -v` and `xbranch_compare/run_comparison.sh` green (the harness's
  own reference scenarios may need updating if they exercise `use_finite_mc=True` with
  the old formula's expected values baked in -- check `xbranch_compare/xbranch_old_api.py`
  /`xbranch_new_api.py`/`shared_constants.py`).

---

## 6. Fix #6: Report disconnected accepted intervals separately, not merged

### The issue this fixes (already root-caused, not yet fixed)

`pyfc/orchestrator.py`'s `_save_fc_json` (~line 205-213) currently computes 1D interval
bounds as:
```python
accepted_points = [val for val, is_acc in zip(test_points, accepted) if is_acc]
if accepted_points:
    interval_bounds = [min(accepted_points), max(accepted_points)]
```
This assumes the accepted region is a single contiguous run. If the true accepted set
were disconnected (a real possibility under the Feldman-Cousins unified approach near
certain boundaries), this silently reports a single interval spanning from the true
region's minimum to maximum, **including whatever rejected gap sits in between** --
i.e. it merges what should be reported as separate disjoint intervals into one too-wide
interval.

### What the repo owner asked for (verbatim)

> please report all separate disconnected allowed intervals, don't report them merged.
> Document this.

### What to do

1. Replace the min/max logic with a **contiguous-run finder**: scan `test_points`
   (already in grid/scan order -- confirm this is always ascending; `grids` are built
   from `np.linspace`, which is ascending, so this should hold, but verify at the point
   of use since `test_points` here is `results.get(f"1d_test_p{p_idx+1}")`, which should
   trace back to the same `grid_test`/`grids[p_idx]` array) and find every maximal run
   of consecutive `accepted=True` indices. Each run becomes its own `[lo, hi]` pair.
   Consider factoring this into a small, independently-testable helper, e.g.:
   ```python
   def _find_contiguous_intervals(test_points, accepted):
       """Returns a list of [lo, hi] pairs, one per maximal contiguous run of
       accepted=True in `accepted`, in scan order. Empty list if none accepted."""
   ```
   (naming/placement your call -- module-level private helper in `orchestrator.py`
   alongside `_save_fc_archive`/`_save_fc_json` seems natural).
2. **Schema change**: `interval_bounds` in the output JSON changes from a flat 2-element
   list (`[lo, hi]`) to a **list of 2-element lists** (`[[lo1, hi1], [lo2, hi2], ...]`),
   even in the single-contiguous-interval case (i.e. always a list-of-pairs now, for a
   consistent schema regardless of how many disjoint pieces there are -- don't special-
   case "if only one interval, flatten it," that just reintroduces ambiguity for
   downstream parsers). This is a **breaking change** to the JSON output format for
   anyone currently parsing `interval_bounds` as `[lo, hi]` directly.
3. Check whether the **2D** case (`_save_fc_json`'s `2d_intervals` section) has an
   analogous concept worth extending similarly, or whether 2D regions are already
   reported as raw accept/reject grids (no precomputed bounds) making this a non-issue
   there -- from the prior session's reading, 2D output does *not* precompute
   `interval_bounds`-style scalars (it stores the full accepted boolean grid), so this
   fix is likely 1D-only. Confirm this by re-reading `_save_fc_json`'s 2D branch before
   assuming.
4. Check `_save_fc_archive` (the `.npz` writer) -- from the prior session's reading,
   it stores the *raw* per-point `accepted`/`test_points` arrays, not precomputed
   bounds, so it likely needs **no** schema change (a consumer of the `.npz` could
   already reconstruct disconnected intervals themselves from the raw arrays -- this
   fix is specifically about the JSON's *convenience* precomputed field). Confirm this
   assumption before assuming no `.npz` changes are needed.
5. Check `pyfc/plotting.py`'s `generate_corner_plot` -- does it consume
   `interval_bounds` anywhere, or does it draw directly from the raw `accepted`
   boolean array (e.g. for shading the 1D profile plot)? From the prior session's
   reading, the 1D plotting path likely uses the raw accepted array directly (not the
   JSON's precomputed bounds), so no plotting code change is *expected* -- but verify,
   since if it does consume merged bounds anywhere, the plot would need updating too
   to correctly shade/highlight multiple disjoint accepted regions.
6. **"Document this"** -- the repo owner explicitly asked for documentation, not just
   the code fix:
   - README.md's description of `fc_results.json`'s structure (the "Stored Results &
     Custom Plotting" section) needs updating to describe the new list-of-intervals
     schema, ideally with a short concrete example showing a two-interval case.
   - `docs/source/outputs.rst` likewise (it explains the output file structure).
   - Add a short note explaining *why* (the Feldman-Cousins unified construction can in
     principle produce a disconnected accepted region, and silently merging them would
     misrepresent the actual confidence region) -- this is worth a sentence or two of
     genuine statistical context, not just "the format changed."
   - Consider whether this deserves a mention in one of the example notebooks (not
     required by the repo owner's instruction, but if an easy, natural example exists
     already or can be added cheaply to `pyfc_algorithmic_features_tutorial.ipynb` or
     similar, it would make the behavior concrete and visible -- optional, use
     judgment, don't force a contrived example just to demonstrate this).

### Verification

- New unit test(s) for the `_find_contiguous_intervals` helper directly: a hand-built
  `accepted` boolean array with two separate `True` runs (e.g.
  `[F,F,T,T,T,F,F,T,T,F]`), asserting it returns two `[lo,hi]` pairs matching the
  correct `test_points` values at each run's start/end, not one merged pair.
- An end-to-end test: either find or construct a real (or deliberately pathological/
  mock) `compute_rates_func` that produces a genuinely disconnected accepted region for
  some parameter (this may take some searching/construction -- a rate function with a
  sharp, non-monotonic feature could do it), and confirm `compute_fc_intervals`'s JSON
  output reports the disjoint pieces separately. If constructing a genuinely-physical
  disconnected-region example proves difficult in the time available, a test that
  directly calls `_save_fc_json`-adjacent logic with a hand-crafted `results` dict
  (mocking `accepted`/`test_points`) is an acceptable substitute for proving the JSON-
  writing logic itself is correct, even without a fully organic disconnected-region
  physics example.
- Full `pytest tests/ -v` green, including any *existing* tests that read
  `interval_bounds` and assumed the old `[lo, hi]` flat-list shape (grep for
  `interval_bounds` across `tests/*.py` and update accordingly).
- `xbranch_compare/run_comparison.sh` -- check whether its comparator
  (`xbranch_compare/comparator.py`) reads `interval_bounds` from JSON output anywhere
  and needs updating for the new schema (it primarily diffs `.npz` arrays per the
  earlier reading, so likely unaffected, but confirm).
- CHANGELOG entry under **"BREAKING CHANGES"** (not just "Fixed"), since this changes
  the JSON output schema for existing callers -- follow the established BREAKING
  CHANGES entry format/verbosity from this same `[0.10.0]` release (see the top of
  `CHANGELOG.md` for the tone/detail level already established, e.g. the
  `S_model`/`B_model` removal entry's "Migration:" pointer style).

---

## 7. Suggested execution order

Mirroring this repo's established one-commit-per-logical-change discipline:

1. **Fix #1** (sparsify_grid default) -- cheapest, do first, ask the one open question
   (fix the underlying bug too, or just the default?) early so the answer doesn't block
   later work.
2. **Fix #3** (defaults matching README) -- also cheap, same class of fix as #1, do
   together or back-to-back. Ask the one open question (the 3 sentinel parameters)
   here too.
3. **Fix #4** (discontinuity protection) -- self-contained, moderate effort, clear
   acceptance criteria (the monotonicity sweep test).
4. **Fix #5** (paper-based formula check) -- read the paper first, since the answer
   (does the code change or not) determines how much work this actually is.
5. **Fix #6** (disconnected intervals) -- moderate-to-large, but well-scoped once the
   contiguous-run helper is written and tested in isolation.
6. **Fix #2** (adaptive_toys/toy_batch_size) -- do this **last**, since it's the
   highest-risk/highest-design-effort fix and benefits from having every other fix's
   test/CHANGELOG/doc-update rhythm already warmed up. Do the `toy_batch_size` (memory
   chunking) half first, independently -- it's lower-risk and can land as its own
   commit even if the `adaptive_toys` (statistical early-stopping) half needs more time
   or a check-in with the repo owner before finalizing the exact stopping rule.

After all six: full final verification pass --
`pytest tests/ -v` (both the base environment and
`PYTHONPATH=/home/mbustamante/Research/FeldmanCousins
/home/mbustamante/anaconda3/envs/py310/bin/python -m pytest tests/ -v`, since only the
`py310` env has `ultranest` installed and some new tests may need it),
`bash xbranch_compare/run_comparison.sh` (zero mismatches), `python
scripts/check_changelog_sync.py`, a docs build (`cd docs && make html`, check no new
warnings, `rm -rf build` after), `git worktree list` (clean), and a final `git status
--short` review before considering this done.

## 8. Final acceptance checklist

- [ ] Fix #1: `sparsify_grid` defaults to `False` in `compute_fc_intervals`'s own
      signature; docs/comments swept for stale mentions; open question about the
      underlying `eval_2d_point` bug resolved with the repo owner (fixed or explicitly
      deferred, not silently ignored).
- [ ] Fix #2: `adaptive_toys`/`toy_batch_size` actually change behavior; validated not
      to bias results (comparison tests against a non-adaptive reference); README's
      description (including the OOM-troubleshooting paragraph) matches reality.
- [ ] Fix #3: `compute_fc_intervals`'s defaults for `n_toys`/`save_log`/`smooth_1d`/
      `smooth_2d`/`cl` match README.md; `num_cores`/`output_file`/`param_names` handled
      per the flagged judgment call (resolved with the repo owner if going against the
      brief's recommendation).
- [ ] Fix #4: unphysical-region smoothing is provably monotonic across the mu=0/
      p_events=0 boundary (new sweep test), both binned and unbinned.
- [ ] Fix #5: paper read; `alpha = mu^2/sigma^2 + 1` formula confirmed correct (with
      citation added) or corrected (with tests, docs, and a CHANGELOG warning updated
      to match).
- [ ] Fix #6: disconnected 1D accepted intervals reported as separate `[lo,hi]` pairs,
      never merged; documented in README.md and `docs/source/outputs.rst`; new tests
      for the contiguous-run helper.
- [ ] `pytest tests/ -v` green in both environments.
- [ ] `bash xbranch_compare/run_comparison.sh` shows zero mismatches.
- [ ] `python scripts/check_changelog_sync.py` passes.
- [ ] Docs build clean, no new warnings.
- [ ] `git worktree list` clean.
- [ ] `git status --short` reviewed, nothing unexpected left uncommitted or untracked.
- [ ] CHANGELOG.md / docs/source/changelog.rst updated for all six fixes (some may
      share an entry where that reads more naturally, per the guidance above).
