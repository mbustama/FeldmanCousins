# Brief: Add test-coverage measurement to PyFC, modeled on the Magnus project

**Purpose of this document**: handoff brief for a fresh chat session, to be pasted or
`@`-referenced at the start of that session. The task: add `pytest-cov`-based coverage
measurement to PyFC (`/home/mbustamante/Research/FeldmanCousins`), following the pattern
already established in the repo owner's other project, Magnus
(`/home/mbustamante/Research/magnus`) — then act on what the first measurement finds.

**Branch**: `dev-coverage` already exists locally, created from `main`, currently sitting
at `main`'s HEAD with zero commits of its own. Work there. Do not touch `main`.

Written by an assistant that had just finished a long PyFC session (two full-codebase
audit rounds, the v0.10.0 release, PyPI publish, and a badges PR) and then investigated
the coverage question empirically before running low on context. **Everything in
section 1 below was measured on this machine, not inferred** — those numbers are the
main reason this brief exists, because the single most important fact about coverage in
PyFC does not apply to Magnus at all and would be easy to get wrong.

---

## 1. READ THIS FIRST: numba `@njit` makes naive coverage numbers a lie

PyFC's mathematical core (`pyfc/binned.py`, 6 `@njit(fastmath=True, nogil=True)`
functions including `calc_nll`, the NLL everything else is built on) is compiled by
numba at runtime. **coverage.py cannot trace inside numba-compiled functions** — the
Python bytecode is never executed, so those lines are reported as missed even when the
tests hammer them.

Measured on this machine, same three test files (`test_smoothing.py`,
`test_finite_mc_likelihood.py`, `test_nd_binned.py`), branch coverage on:

| Run | `pyfc/binned.py` coverage | wall time |
|---|---|---|
| default (JIT on) | **10%** | 27.2 s |
| `NUMBA_DISABLE_JIT=1` | **90%** | 6.3 s |

10% is not a real number — `calc_nll` is one of the most heavily tested functions in the
package (`tests/test_smoothing.py`, `tests/test_finite_mc_likelihood.py`,
`tests/test_nd_binned.py` all target it directly, including the high-`alpha` precision
regression and the NaN-guard tests added in the last release). Publishing 10% for it
would be actively misleading. `NUMBA_DISABLE_JIT=1` makes numba run the decorated
functions as ordinary Python, which coverage can trace.

### The catch, also measured

`NUMBA_DISABLE_JIT=1` is *not* uniformly free. Most test files are actually **faster**
without JIT (no compilation overhead), but two are pathological, because
`strategy="grid"`'s toy generators are `@njit(parallel=True)` vectorized loops that
become plain Python loops:

| test file | time under `NUMBA_DISABLE_JIT=1` |
|---|---|
| `test_smoothing.py` | 0.5 s |
| `test_finite_mc_likelihood.py` | 0.4 s |
| `test_numba_compilation.py` | 0.2 s |
| `test_pdf_components.py` | 0.7 s |
| `test_statistics.py` | 0.7 s |
| `test_unbinned_toys.py` | 1.6 s |
| `test_nd_binned.py` | 2.3 s |
| `test_disconnected_intervals.py` | 3.2 s |
| `test_adaptive_toys.py` | **72.5 s** |
| `test_sparsify_grid.py` | **> 90 s (timed out)** |

Running the *whole* suite under `NUMBA_DISABLE_JIT=1` was abandoned after **20+ minutes**
without finishing (the same suite takes ~75 s normally). Do not do that.

### The recipe that works (validated end to end on this machine)

Split into two runs and let coverage combine them via `--cov-append`:

```bash
python -m coverage erase

# Run A: JIT disabled -> binned.py's njit internals become traceable.
#        Everything except the one pathological file.
NUMBA_DISABLE_JIT=1 python -m pytest tests/ --ignore=tests/test_sparsify_grid.py -q \
    --cov=pyfc --cov-branch --cov-append --cov-report=

# Run B: JIT enabled -> the pathological file runs at normal speed.
python -m pytest tests/test_sparsify_grid.py -q \
    --cov=pyfc --cov-branch --cov-append --cov-report=

python -m coverage report
```

Measured result: Run A **235 s** (92 passed, 5 skipped), Run B **14 s** (2 passed) —
99 tests total, all accounted for, ~4 minutes end to end. That is a perfectly acceptable
CI job.

`test_adaptive_toys.py` at 72.5 s is the bulk of Run A's time but is tolerable; it is
left in Run A deliberately, since excluding it would lose real coverage of
`toys.py`'s adaptive-stopping logic. If Run A's time becomes a problem later, it is the
next candidate to move to Run B — but note that anything moved to Run B loses njit
traceability, which only matters for code that actually runs inside `@njit` functions.

**Sanity check to run early**: confirm the full suite still *passes* under
`NUMBA_DISABLE_JIT=1` (Run A above: 92 passed, 5 skipped, 0 failed — it did on this
machine). `NUMBA_DISABLE_JIT=1` changes `prange` to `range` and skips compilation
entirely, so a test asserting on JIT-specific behavior could in principle behave
differently. `tests/test_numba_compilation.py` passed fine.

---

## 2. The measured baseline (branch coverage on, via the recipe above)

This is where PyFC starts, before any new tests are written:

| Module | Stmts | Miss | Branch | BrPart | Cover |
|---|---|---|---|---|---|
| `pyfc/__init__.py` | 4 | 0 | 0 | 0 | 100% |
| `pyfc/binned.py` | 133 | 14 | 48 | 3 | 90% |
| `pyfc/config.py` | 50 | 44 | 10 | 0 | **10%** |
| `pyfc/generate_config.py` | 90 | 82 | 26 | 1 | **8%** |
| `pyfc/optimizers.py` | 318 | 126 | 128 | 4 | 59% |
| `pyfc/orchestrator.py` | 401 | 98 | 232 | 32 | 73% |
| `pyfc/plotting.py` | 104 | 6 | 50 | 5 | 93% |
| `pyfc/toys.py` | 93 | 28 | 44 | 6 | 62% |
| `pyfc/unbinned.py` | 81 | 52 | 30 | 1 | 34% |
| **TOTAL** | **1274** | **450** | **568** | **52** | **64%** |

Caveat: this run had `ultranest` **not** installed, so the 5 UltraNest tests skipped and
all three `*_ultranest` fit functions in `optimizers.py` (roughly the last third of that
968-line file) counted as missed. Installing the `optimizers` extra will move
`optimizers.py` up substantially — see decision D2 below.

---

## 3. What Magnus does (the pattern to replicate)

Read these files directly; they are the reference implementation and are heavily
commented with the *rationale*, which matters more than the syntax:

- **`/home/mbustamante/Research/magnus/pyproject.toml`**, `[tool.coverage.run]` (lines
  ~86-102) and `[tool.coverage.report]` (~104-113). Note `branch = true`, `omit` for the
  `__main__.py` shim, `show_missing = true`, `exclude_also` for
  `if __name__ == .__main__.:`, and the explicit comment explaining why there is
  deliberately **no `fail_under`**.
- **`/home/mbustamante/Research/magnus/.github/workflows/tests.yml`**, the `coverage:`
  job. Study the comments: why it is a separate job rather than a matrix axis, the
  job-level `env: CODECOV_TOKEN` trick (a step's own `env:` is not visible to that
  step's `if:`), the `GITHUB_STEP_SUMMARY` publish step, the **dormant** Codecov step,
  and the `coverage.xml` artifact upload.
- **`/home/mbustamante/Research/magnus/README.md`** — "Continuous Integration" section
  (the Coverage job paragraph) and "Requirements" (local usage + why branch coverage).
- **`/home/mbustamante/Research/magnus/docs/source/installation.rst`** — the
  "Measuring test coverage" subsection (~lines 89-114).
- **`/home/mbustamante/Research/magnus/CHANGELOG.md`** ~lines 72-99 — both the plumbing
  entry *and* the follow-up entry describing what the first coverage run exposed.

Magnus's `test` extra is `["pytest", "pytest-cov"]`; `dev` also carries `pytest-cov`.

---

## 4. PyFC's current state (what has to change)

- **`pyproject.toml`**: `test = ["pytest"]`, `dev = ["pytest", "twine", "build"]` — no
  `pytest-cov`. **No `[tool.coverage.*]` section at all.** No `[tool.pytest.ini_options]`
  either.
- **`.gitignore`**: no coverage entries (`.coverage`, `coverage.xml`, `htmlcov/` all
  missing) — must be added, or a coverage run leaves untracked files behind.
- **`.github/workflows/pytest.yml`**: single `test:` job, matrix 3.9/3.10/3.11,
  `pip install -e ".[test]"`, `pytest tests/ -v`. No coverage job.
  **Gotcha**: triggers are `push: branches: ["main"]` and `pull_request: branches:
  ["main"]` — so **pushes to `dev-coverage` will not run CI at all**. Magnus widened its
  filter to `[main, dev, "dev-*"]` for exactly this reason. Widen PyFC's too (at minimum
  add `"dev-*"`), or the next session gets no CI signal on its own branch and only finds
  out at PR time.
- **`pyfc/orchestrator.py`** has a **181-line** `if __name__ == "__main__":` demo block
  (line 931 to EOF, 1111 lines total). Under coverage that is a permanent, unclosable
  miss — ~16% of the file. Magnus's `exclude_also = ["if __name__ == .__main__.:"]`
  handles this: when the matched line introduces a block, coverage excludes the **entire
  block**, not just the `if` line. This matters far more for PyFC (181 lines) than it did
  for Magnus (a 2-line shim). `pyfc/generate_config.py`'s equivalent block is 1 line.
- **Docs sections to update** (mirroring Magnus): README `### Verifying the Installation`
  (line ~85), `### Continuous Integration (CI)` (~97), `### Developer Installation`
  (~109); and `docs/source/installation.rst`'s `Verifying the Installation` /
  `Developer Installation` sections (~lines 40-59). Both README and `installation.rst`
  also carry a **File Tree** whose `.github/workflows/` entries describe each workflow —
  update `pytest.yml`'s one-line description to mention coverage.
- **Badges**: PyFC just gained a badge block in both `README.md` and
  `docs/source/index.rst` (merged in PR #2). Magnus has no coverage badge (its Codecov
  step is dormant). If Codecov is ever switched on for PyFC, a coverage badge belongs in
  both places — but per D3 below, default to leaving it dormant.

---

## 5. Decisions that need the repo owner (ask before implementing)

**D1 — JIT-disable strategy.** The two-run split in section 1 is the recommendation, and
it is validated. But it is a genuine design choice with a maintenance cost: the CI job
hardcodes which test files go in which run, and that list can go stale when tests are
added. Alternatives: (a) accept a meaningless ~10% for `binned.py` and run everything
normally (simplest, but publishes a misleading number for the package's mathematical
core); (b) the two-run split (recommended); (c) two-run split plus a guard test that
fails if a new test file is added without being classified. Recommend (b), and mention
(c) as a possible follow-up rather than doing it up front.

**D2 — install the `optimizers` extra in the coverage job?** CI currently installs only
`.[test]`, so `ultranest` is absent, 5 tests skip, and roughly the last third of
`optimizers.py` (the three `*_ultranest` fit functions) counts as missed. Installing
`.[test,optimizers]` in the *coverage job only* would make `optimizers.py`'s number
honest. Cost: a heavier install and slightly longer job; also `ultranest` pulls in
`cython`/`corner`. Recommend **yes** — an artificially depressed number invites someone
to "fix" coverage that is not actually missing. Note the matrix `test:` job should
probably stay on `.[test]` alone, since testing that PyFC works *without* its optional
dependency is itself valuable.

**D3 — Codecov.** Magnus's Codecov step is deliberately dormant (skipped unless a
`CODECOV_TOKEN` secret exists) because that repo is private. **PyFC is public** and
already on PyPI, so switching Codecov on is genuinely available here in a way it was not
for Magnus. Still recommend shipping it dormant, identically to Magnus, and treating
"create the secret" as a separate, deliberate decision — but flag to the owner that for
PyFC it is a live option, not a theoretical one, and that per-PR diff coverage comments
are the part that actually changes day-to-day behavior.

**D4 — `fail_under` threshold.** Magnus deliberately has none. Same reasoning applies
(a threshold invented before the first measurement either never fires or blocks unrelated
work). Now that a real baseline exists (64%), the owner *could* pin something just below
it. Recommend still not gating initially — but this is now an informed choice rather than
a blind one, so it is worth asking rather than assuming.

**D5 — scope of phase 2.** See section 6. Writing new tests to close the
`config.py`/`generate_config.py` gap is arguably a separate piece of work from "add
coverage measurement." Ask whether to do it in this same branch/PR or split it.

---

## 6. Phase 2: act on what the measurement finds (this is where the value is)

Magnus's coverage work was *not* just plumbing. Its CHANGELOG records that the first
coverage run exposed that **23 of 36 NSI/LIV wrappers and 25 exported Hamiltonian
builders were executed by nothing at all** — some appeared in a `parametrize` list but
only in tests that inspected a signature or source text without ever *calling* the
function, so a mistyped keyword would have shipped unnoticed. The fix was two
**self-discovering** structural tests (`test_every_bsm_wrapper_runs_and_is_unitary`,
`test_every_exported_hamiltonian_builder_is_hermitian`) that enumerate their own subjects
from the module and from `__all__`, so a name added later is swept without anyone
remembering to extend a list. **Both were verified to fail against a deliberately broken
library before being kept.** Replicate all three of those disciplines.

PyFC's equivalent findings, straight from the section 2 baseline:

1. **`pyfc/config.py` — 10%.** `parse_arguments()` and `generate_sample_config()` are the
   entire CLI/JSON config layer, and essentially nothing tests them. This is the single
   biggest gap. Note this layer was touched repeatedly in the last release
   (`n_restarts`/`neighbor_seeding`/`scipy_method` wiring) and each time correctness was
   established by *manually* running the CLI — exactly the kind of thing a test should be
   doing. A good self-discovering test here: assert that every key in `config.py`'s base
   config dict has a matching `--flag` in `parse_arguments()` and vice versa (this would
   have caught the `scipy_method`-missing-from-wizard bug found by hand last round).
2. **`pyfc/generate_config.py` — 8%.** The interactive wizard. Testable by feeding
   scripted stdin (the last session drove it with `yes "" | python -m pyfc.generate_config`)
   or by calling `get_input`'s helpers directly. A test that runs the wizard accepting all
   defaults and asserts the resulting JSON matches `compute_fc_intervals`'s actual
   defaults would be high-value — that exact mismatch was a real bug fixed last round.
3. **`pyfc/unbinned.py` — 34%.** Lower than expected given `test_pdf_components.py` and
   the new `test_unbinned_toys.py`. Worth reading the `--cov-report=term-missing` output
   to see which branches are cold.
4. **`pyfc/toys.py` — 62%** and **`pyfc/optimizers.py` — 59%** (the latter partly a D2
   artifact). Check the missing branches before assuming they need new tests.

Use `--cov-report=html` and read `htmlcov/` to see *which* branches are cold rather than
guessing from the summary percentages.

**Discipline to carry over from Magnus**: any new test written to close a gap must be
verified to actually fail against a deliberately broken version of the code before it is
kept. A test that raises coverage without being able to detect a defect is worse than no
test, because it makes the number look better while catching nothing.

---

## 7. Incidental finding, flag to the owner (out of scope, do not fix silently)

While building the package during the release, the wheel was observed to contain
`tests/` and `xbranch_compare/` as **top-level installed packages** — i.e.
`pip install PyFeldmanCousins` ships the test suite and the cross-branch regression
harness into the user's `site-packages`, where they can collide with any other project's
top-level `tests` package. Cause: `[tool.setuptools.packages.find] where = ["."]` with no
`include`/`exclude` filter. Fix would be `include = ["pyfc*"]`, but that changes what the
published distribution contains, so it is a deliberate packaging decision for the owner,
not a drive-by edit. It is adjacent to this task only because writing
`[tool.coverage.run] source = ["pyfc"]` raises the same "what actually *is* the package"
question. **Mention it; do not fix it as part of the coverage work.**

---

## 8. Verification checklist

- [ ] `pytest tests/ -v` still green under **normal** JIT (99 tests; 94 pass + 5 skip
      without `ultranest`, or 99 pass with it).
- [ ] The two-run coverage recipe completes and produces a combined report; total is in
      the same ballpark as the 64% baseline in section 2 (higher if D2 is adopted).
- [ ] `pyfc/binned.py` reports ~90%, **not** ~10% — this is the canary that the
      JIT-disable half of the recipe is actually working.
- [ ] A bare `pytest tests/ --cov` locally measures the same thing CI measures (i.e. all
      settings live in `pyproject.toml`, not in CI flags) — Magnus states this explicitly
      as a design goal and it is worth preserving.
- [ ] `git status --short` clean after a coverage run (`.gitignore` covers `.coverage`,
      `coverage.xml`, `htmlcov/`).
- [ ] Sphinx docs build clean: `cd docs && make html` — PyFC currently has **28
      pre-existing warnings**, all unrelated docstring-formatting issues in `pyfc/*.py`.
      That count should not increase. `rm -rf build` afterwards.
- [ ] `python scripts/check_changelog_sync.py` passes — PyFC keeps `CHANGELOG.md` and
      `docs/source/changelog.rst` in **hand-maintained structural sync**, enforced by CI
      (`.github/workflows/changelog-sync.yml`). Any CHANGELOG entry must be mirrored
      bullet-for-bullet into the `.rst`. Read that script's module docstring before
      editing either file.
- [ ] If README's CI section test counts change, recompute them from an actual
      `pytest tests/ --collect-only -q` rather than editing by hand — that section has
      gone stale twice before and was rewritten to category counts specifically to stop
      that happening.
- [ ] `bash xbranch_compare/run_comparison.sh` — only if any `pyfc/` source changed;
      pure config/CI/docs work does not need it.

## 9. Suggested execution order

1. Read the Magnus reference files (section 3) and PyFC's current state (section 4).
2. Independently re-verify the section 1 numba finding — do not take this brief's word
   for it. One quick run of `test_smoothing.py` with and without `NUMBA_DISABLE_JIT=1`
   is enough to confirm the 10%/90% split.
3. Put decisions D1-D5 to the owner before writing code.
4. Phase 1 (plumbing): `pyproject.toml`, `.gitignore`, `pytest.yml` (including the branch
   filter widening), README, `installation.rst`, File Trees, CHANGELOG + `changelog.rst`.
5. Get a clean CI run on `dev-coverage` and read the summary-page table.
6. Phase 2 (act on findings), scoped per D5 — with the Magnus discipline: self-discovering
   tests, each verified to fail against deliberately broken code.
7. Final verification pass (section 8), then PR into `main`.

**Repo conventions worth knowing** (established across the last several PyFC sessions):
commits are one-logical-change each with a body explaining the *why*, not just the what;
PRs are opened with `gh pr create` and merged only after CI is green; and this repo's
`gh` CLI hits a snap-confinement error reading `/etc/gitconfig` — prefix every `gh`
invocation with `GIT_CONFIG_NOSYSTEM=1` (plain `git` is unaffected). Handoff briefs like
this one get archived into `docs/dev/` and committed at the end of the session that
executes them.
