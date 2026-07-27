# Brief: Pre-push cleanup on `dev-no-templates` (findings #1-9 + CHANGELOG sync workflow)

**Purpose of this document**: handoff brief for a fresh chat session, to be pasted or
`@`-referenced at the start of that session. It was written by an assistant with full
context of a prior, large multi-session refactor on this branch, immediately followed by
a full-codebase review the repo owner (Mauricio Bustamante) requested before pushing to
GitHub. This document covers **only the follow-up cleanup from that review** — the main
refactor itself is **done, committed, and not part of this brief's scope** (summarized
below for context only).

Repository: `/home/mbustamante/Research/FeldmanCousins` (PyFC, GPLv3, sole author is the
repo owner). Branch: **`dev-no-templates`** (already exists, already has the refactor
commits below). Continue working directly on this branch. Do **not** touch `main` or
`dev`.

---

## 0. Context: what already happened on this branch (do NOT redo this)

`dev-no-templates` branched off `dev` at commit `17f3ffd` and already contains 12 commits
(in order): `7044362`, `d93dd81`, `52a90ca`, `1e5ef97`, `8f36a52`, `fb8ffde`, `2e0e05c`,
`d4e787a`, `ac37bf4`, `4b8f628`, `fe35e9b`, `13600ca`. Together they:

1. Removed `S_model`/`B_model` entirely from `compute_fc_intervals` and every
   optimizer/toy-generation function, replacing them with:
   - Binned: fixed templates referenced via closure instead of passed as arguments.
     `compute_rates_func(params, S_sumw2, B_sumw2) -> (mu, sigma2)`.
   - Unbinned: a general `pdf_components: list[callable]` (any length, not just 2),
     with `compute_rates_func(params, probs) -> (expected_total, p_events)`.
2. Added support for arbitrary N-dimensional binned `data`/`mu`/`sigma2` shapes (`calc_nll`
   flattens internally before summing).
3. Renamed `S_sigma2`/`B_sigma2` to `S_sumw2`/`B_sumw2` everywhere (these are per-bin sums
   of squared MC weights, `sum(w_i^2)` — the standard variance estimator for a weighted MC
   sample), since `S_`/`B_` no longer had `S_model`/`B_model` to justify the prefix and
   "sigma2" was an imprecise/overloaded name.
4. Added a cross-branch regression harness at `xbranch_compare/` (uses a temporary `git
   worktree` of `dev` to prove the refactor changed signatures only, never numerics;
   kept permanently as a reusable tool for future breaking changes).
5. Replaced the old monolithic `examples/pyfc_tutorial.ipynb` (whose heaviest cells took
   20+ minutes each) with four lightweight, narrated notebooks:
   `pyfc_quickstart_tutorial.ipynb`, `pyfc_high_dimensional_tutorial.ipynb`,
   `pyfc_algorithmic_features_tutorial.ipynb`, `pyfc_strategy_comparison_tutorial.ipynb`
   — plus the pre-existing `pyfc_joint_constraints_tutorial.ipynb`, updated in place.
6. Added `generate_corner_plot` to `pyfc/__init__.py`'s top-level exports (alongside the
   already-exported `compute_fc_intervals`), and switched every doc/notebook example from
   `from pyfc.orchestrator import compute_fc_intervals` / `from pyfc.plotting import
   generate_corner_plot` to `from pyfc import compute_fc_intervals, generate_corner_plot`
   — the internal-submodule-reaching style was a "local checkout" tell, inappropriate for
   examples meant to work against a `pip`-installed package.
7. Bumped `pyproject.toml` to `version = "0.10.0"` (unreleased) and wrote a "BREAKING
   CHANGES" CHANGELOG.md/docs/source/changelog.rst entry (a new heading convention for
   this project, already used once, not open for reconsideration in this brief).

**Current state**: full test suite green (`pytest tests/ -v`, 44 tests), cross-branch
harness passes with zero mismatches (`bash xbranch_compare/run_comparison.sh`), all 5
example notebooks execute cleanly with embedded outputs, `git worktree list` is clean.

After that refactor landed, the repo owner asked for **a full scan of the codebase for
errors, inconsistencies, and potential improvements** (report-only, nothing implemented).
Eleven findings came back; **the repo owner wants only #1-9 fixed now** (below). Findings
#10 (`pyproject.toml` claims `requires-python = ">=3.8"` but CI only tests 3.9-3.11) and
#11 (repo-wide `ruff format`/`ruff check` non-compliance, pre-existing, CI's lint job has
`continue-on-error: true` so it doesn't block) are **explicitly out of scope** — do not
fix them unless the repo owner asks separately.

---

## 1. Finding #1 (HIGH PRIORITY — repo owner is especially concerned about this one)

### The bug

`calc_nll` (binned.py) and the two toy generators that also flatten a rate array crash
under numba nopython mode when given a **non-contiguous** N-D array — e.g. a transposed
2D histogram. This was flagged as a theoretical risk during the original N-D-support work
("verify empirically; if `.reshape(-1)` raises on non-contiguous input, fall back to
`np.ascontiguousarray(x).reshape(-1)`") but was never actually triggered during that work,
so the guard was never added. It has now been **reproduced**:

```python
import numpy as np
from numba import njit
from pyfc.binned import calc_nll

@njit(fastmath=True, nogil=True)
def compute_rates(params, S_sumw2, B_sumw2):
    mu = params[0] * np.ones_like(S_sumw2) + params[1]
    return mu, S_sumw2

N_obs_2d = np.array([[5.0, 3.0], [8.0, 2.0]])
N_obs_T = N_obs_2d.T  # non-contiguous view -- e.g. from reshaping a histogram's axes
S_sumw2 = np.zeros_like(N_obs_T)
B_sumw2 = np.zeros_like(N_obs_T)
params = np.array([2.0, 1.0])

calc_nll(params, N_obs_T, S_sumw2, B_sumw2, False, compute_rates)
# -> NotImplementedError: incompatible shape for array
```

This is a **plausible real-world trigger**, not a contrived edge case: the entire point of
the N-D binned-data feature is letting users lay out bins however is natural for them
(e.g. "I have a `(cos_theta, E)` histogram but want `(E, cos_theta)`, so I do `data.T`").
Anyone who transposes, or otherwise non-trivially slices/strides, their histogram before
passing it to `compute_fc_intervals` will hit this crash with a confusing low-level numba
error instead of either working correctly or failing with a clear message.

### Exact locations (5 call sites, all in `pyfc/binned.py`, all need the same fix)

- `calc_nll` (currently ~line 118-120):
  ```python
  mu_flat = mu.reshape(-1)
  N_obs_flat = N_obs.reshape(-1)
  sigma2_flat = sigma2_arr.reshape(-1)
  ```
- `generate_and_fit_toys_grid_1d` (currently ~line 375):
  ```python
  mu_true_flat = mu_true.reshape(-1)
  ```
- `generate_and_fit_toys_grid_2d` (currently ~line 430):
  ```python
  mu_true_flat = mu_true.reshape(-1)
  ```

(Re-grep `\.reshape(-1)` in `pyfc/binned.py` before editing — line numbers will have
drifted slightly from other work, and this greps cleanly since `pyfc/unbinned.py` has
**zero** `.reshape` calls, confirmed, so the fix is binned-only.)

### The fix (already verified working under numba, both correctness and the
no-op-when-already-contiguous property)

Wrap each `.reshape(-1)` call with `np.ascontiguousarray(...)` first:

```python
mu_flat = np.ascontiguousarray(mu).reshape(-1)
N_obs_flat = np.ascontiguousarray(N_obs).reshape(-1)
sigma2_flat = np.ascontiguousarray(sigma2_arr).reshape(-1)
```

and analogously for the two `mu_true.reshape(-1)` sites. Verified directly under
`@njit(fastmath=True, nogil=True)`:

```python
@njit(fastmath=True, nogil=True)
def test_fix(arr):
    return np.ascontiguousarray(arr).reshape(-1)

test_fix(N_obs_T)  # -> array([5., 8., 3., 2.]) -- correct, matches N_obs_T.flatten()
test_fix(np.array([1.0, 2.0, 3.0]))  # -> still works for the already-contiguous case
```

`np.ascontiguousarray` is a no-op (returns the same array, no copy) when the input is
already C-contiguous, so this fix has **zero performance cost on the hot, common path**
(the vast majority of real usage, where users pass ordinary `np.array`/`np.zeros`/
`np.random.poisson(...)`-constructed arrays, which are always contiguous by default) —
only non-contiguous inputs pay for a copy, and those would otherwise have crashed anyway.

### What to do

1. Apply the fix at all 5 call sites in `pyfc/binned.py`.
2. Update `calc_nll`'s docstring note about numba/contiguity if one exists (there is a
   comment thread about this in the original N-D-support commit's message — check `git
   log -p` on `pyfc/binned.py` around commit `7044362` for the exact prior wording if you
   want to fold this into an updated docstring note, not required but nice-to-have).
3. **Add a regression test** to `tests/test_nd_binned.py` (already has flatten-invariance
   tests for contiguous 2D arrays) covering exactly this case: build a 2D array, take a
   transposed (or otherwise strided) non-contiguous view, confirm `calc_nll` no longer
   raises and produces the same NLL as the contiguous equivalent. Suggested test:
   ```python
   def test_calc_nll_handles_non_contiguous_input():
       """A transposed (non-contiguous) N-D array must not crash calc_nll under numba."""
       params = np.array([2.0, 1.0])
       N_obs_2d = np.array([[5.0, 3.0, 9.0], [8.0, 2.0, 4.0]])
       S_sumw2_2d = np.zeros((2, 3))
       B_sumw2_2d = np.zeros((2, 3))

       N_obs_T = np.ascontiguousarray(N_obs_2d).T  # build then transpose -> non-contiguous
       # Careful: constructing a genuinely non-contiguous array for the test needs a
       # transpose or a slice-with-step; verify with `.flags['C_CONTIGUOUS']` before
       # asserting the test actually exercises the non-contiguous path, since some numpy
       # operations may silently return a contiguous copy depending on version/shape.
       assert not N_obs_T.flags['C_CONTIGUOUS']

       nll_transposed = calc_nll(params, N_obs_T, S_sumw2_2d.T, B_sumw2_2d.T, False, _compute_rates_nd)
       nll_flat = calc_nll(params, N_obs_T.reshape(-1).copy(), ..., False, _compute_rates_nd)  # reference
       # (fill in a correct reference comparison; the point is: no crash, and the value
       # matches what you'd get from an equivalent, deliberately-contiguous computation)
   ```
   (This sketch has a placeholder — work out the cleanest, most direct reference
   comparison when you implement it; the existing tests in that file already establish
   the pattern of comparing a 2D result against its `.reshape(-1)` 1D-flattened
   equivalent, so mirror that but starting from a non-contiguous 2D array instead.)
4. Also add `pdf(data)`-style non-contiguous coverage is **not** needed for
   `pyfc/unbinned.py` (confirmed zero `.reshape` calls there — this bug is binned-only).
5. Run the full test suite (`pytest tests/ -v`) and the cross-branch harness (`bash
   xbranch_compare/run_comparison.sh`) after the fix — both must stay green. The harness
   doesn't currently exercise a non-contiguous array, so it passing is expected and not
   itself proof of the fix; rely on the new unit test for that.

---

## 2. Finding #2: `pyproject.toml`'s `docs` extra references a nonexistent package

`pyproject.toml` line 33:
```toml
docs = ["sphinx", "sphinx-theme-name", "sphinxcontrib-bibtex"]
```
`"sphinx-theme-name"` is a literal placeholder that was never filled in — it is not a real
PyPI package. `docs/source/conf.py` actually configures `html_theme = 'sphinx_rtd_theme'`
(confirmed), and `.github/workflows/pages.yml` (the workflow that actually builds/deploys
docs) installs `sphinx sphinx-rtd-theme sphinxcontrib-bibtex` directly, **bypassing** this
broken extras group entirely — which is exactly why CI never caught it. Anyone who runs
`pip install -e ".[docs]"` locally would get a dependency-resolution failure.

**Fix**: change line 33 to:
```toml
docs = ["sphinx", "sphinx-rtd-theme", "sphinxcontrib-bibtex"]
```
This is a one-line fix. No test suite implications (this extras group isn't imported by
any Python code, only used by `pip install`). After fixing, consider (optional,
nice-to-have) updating `.github/workflows/pages.yml` to install via `pip install -e
".[docs]"` instead of the current separate `pip install sphinx sphinx-rtd-theme
sphinxcontrib-bibtex` line, so there's only one place listing doc dependencies and they
can't drift apart again — but this is optional polish, not required by the finding itself.

---

## 3. Finding #3: README.md has a stale checkpoint path

README.md, in the "Outputs, Plots, and Checkpointing" section, "The Checkpoint Engine
(`warm_start`)" subsection (currently ~line 526):

> By default, `warm_start` is set to `True`. PyFC dynamically writes its state to
> `fc_output/checkpoint_fc.npz` after processing each 1D slice of the parameters of
> interest.

The actual checkpoint path is `os.path.join(save_directory, "checkpoint_fc.npz")`
(confirmed in `pyfc/orchestrator.py`), and the default `save_directory` is
`"output/example_fc_output"` (confirmed in `pyfc/config.py` and repeated correctly
elsewhere in this same README file, e.g. the Configuration Parameters table and the
"Stored Results & Custom Plotting" subsection just below it). So the correct default path
is `output/example_fc_output/checkpoint_fc.npz`, not `fc_output/checkpoint_fc.npz` —
`fc_output` looks like a leftover from an earlier default that was changed elsewhere in
the README but missed in this one sentence. This is a documentation-only fix (pre-existing
typo, unrelated to the S_model/B_model refactor).

**Fix**: change `fc_output/checkpoint_fc.npz` to `output/example_fc_output/checkpoint_fc.npz`
(or, better, phrase it generically as "`checkpoint_fc.npz` inside your `save_directory`
(default: `output/example_fc_output/`)" to avoid this drifting again if the default ever
changes). Also grep the rest of README.md and all `docs/source/*.rst` files for any other
stray `fc_output` mentions before considering this done — confirmed at review time that
`docs/source/outputs.rst` does **not** repeat this mistake, but re-verify since docs may
have changed since.

---

## 4. Finding #4: `REFACTOR_BRIEF_dev-no-templates.md` is untracked in the repo root

The original handoff brief that kicked off the whole S_model/B_model refactor
(`REFACTOR_BRIEF_dev-no-templates.md`) is sitting in the repo root, **untracked** (neither
committed nor gitignored) — confirmed via `git status --short`. This is presumably also
true of **this very file** (`FOLLOWUP_BRIEF_dev-no-templates.md`) once you're reading it
in the new session, since it was written the same way. Before pushing to GitHub, decide
and act on one of:

- **Commit both as project history** — e.g. move them into a `docs/dev/` or `planning/`
  subfolder (not the repo root) and commit, so future contributors can see the design
  rationale behind this refactor.
- **Delete them** once their content is fully absorbed elsewhere (CHANGELOG.md already
  captures the *what changed*; these briefs mostly capture *process*/*handoff context*
  that may not be worth preserving long-term).
- **Leave them out of git entirely** (add both filenames to `.gitignore`, or simply never
  `git add` them) if the repo owner considers them personal working notes, not project
  artifacts.

**This is the repo owner's call, not yours to decide unilaterally** — ask explicitly
before doing any of the above (don't just pick one and proceed) if it isn't already
obvious from how the new session's conversation started (e.g. if the owner pastes this
brief and immediately gives you a preference, that's your answer; otherwise ask).

---

## 5. Finding #5: `.gitignore` doesn't cover the default `output/` directory

`pyfc/config.py`'s hardcoded default `"save_directory": "output/example_fc_output"`
(used both by `parse_arguments()`'s base config and by `pyfc/orchestrator.py`'s `__main__`
demo block) means running `python -m pyfc.orchestrator` or the CLI demo from the repo root
creates an `output/` directory that is **not** covered by any existing `.gitignore` entry
— confirmed via `git check-ignore -v output/example_fc_config/foo.txt` returning exit code
1 (not ignored). This bit the previous session multiple times (had to manually `rm -rf
output` after every demo run). Left uncleaned, this would show up as untracked files ready
to be accidentally `git add -A`'d by a future contributor.

**Fix**: add to `.gitignore` (suggest right after the existing `examples/*_output/`
block, or wherever reads cleanest):
```
output/
fc_output/
```
(Include `fc_output/` too, defensively — it's the stale default mentioned in finding #3,
and even though `pyfc/config.py`'s actual default is `output/example_fc_output`, it's
possible some user-facing example or a user's own script still defaults to or references
`fc_output/`; ignoring both costs nothing.)

---

## 6. Finding #6: `.gitignore` has a redundant/stale egg-info entry

Current `.gitignore` (16 lines total) has both:
```
PyFC.egg-info/
...
PyFeldmanCousins.egg-info/
```
`PyFC.egg-info/` (line 5) appears to be a leftover from before the package was renamed to
`PyFeldmanCousins` in `pyproject.toml` (see the historical commit `f4c49c0 "Changed
project name to avoid PyPI conflict"` for when that rename happened) — the actual,
currently-relevant entry is `PyFeldmanCousins.egg-info/` (line 15), which correctly
matches `pyproject.toml`'s current `name = "PyFeldmanCousins"`. The `PyFC.egg-info/` line
is harmless (an editable install under the old name would never actually be created
again) but redundant/misleading.

**Fix**: remove the stale `PyFC.egg-info/` line, keep `PyFeldmanCousins.egg-info/`. Purely
cosmetic, zero functional risk either way — do this alongside finding #5's edit to the
same file in one pass.

---

## 7. Finding #7: `pyfc/__init__.py` has a stale authoring-artifact comment

Current full content of `pyfc/__init__.py`:
```python
"""
Feldman-Cousins Frequentist Analysis Framework
"""

# Assuming the file is named orchestrator.py based on the previous code base
# You can also expose the config generator if you want it easily accessible
from .generate_config import main as generate_config
from .orchestrator import compute_fc_intervals
from .plotting import generate_corner_plot

__all__ = [
    "compute_fc_intervals",
    "generate_config",
    "generate_corner_plot",
]
```
The comment `# Assuming the file is named orchestrator.py based on the previous code
base` reads like a stale note-to-self from whoever originally scaffolded this file
(possibly an earlier AI-assisted authoring pass) — `orchestrator.py` has been the
established, correct filename for a long time now (confirmed: it's referenced
consistently everywhere else in the codebase, docs, and this entire refactor), so the
"assuming"/"previous code base" framing no longer makes sense and adds no value to a
reader. The second comment line (`# You can also expose...`) is mildly useful context but
also reads more like a suggestion-to-self than documentation.

**Fix**: remove both comment lines (or replace with something meaningful if you think a
one-line note on why `generate_corner_plot`/`generate_config` are re-exported here is
worth keeping — e.g. "`Re-exported at the top level so `from pyfc import X` works without
knowing PyFC's internal module layout.`" — your call, but don't just leave the current
stale wording). This is also where finding #9's docstring-header convention could
optionally be added to this file for consistency with the other 8 (see below;
`__init__.py` currently has **no** `Date:`/`Author:` header at all, so this is additive,
not a rename).

---

## 8. Finding #8: `pyfc/plotting.py` has minor dead code

Two small, harmless but noteworthy dead-code remnants in `generate_corner_plot`
(pre-existing, not introduced by the refactor, `plotting.py` was never touched by it):

- Lines ~175-176: a commented-out duplicate of the live condition immediately above it:
  ```python
  if np.min(z_diff) <= (0.0 + tol) and np.max(z_diff) >= (0.0 - tol):
      ax.contour(X, Y, z_diff, levels=[0.0], colors=[colors[idx % len(colors)]], linewidths=2)
  # if np.min(z_diff) <= 0.0 <= np.max(z_diff):
  #     ax.contour(X, Y, z_diff, levels=[0.0], colors=[colors[idx % len(colors)]], linewidths=2)
  ```
- Line ~215: an unreachable `# plt.close()` immediately after `return fig` at the end of
  the function (the function returns before this line could ever execute, commented-out
  or not).

**Fix**: delete both dead-code remnants. Zero behavioral risk (they're either commented
out already or unreachable). Re-grep exact current line numbers before editing (this file
hasn't been touched since these numbers were recorded, so they should still be accurate,
but confirm).

---

## 9. Finding #9: module docstring dates, replaced with a Created/Last-modified + version scheme

### The ask (repo owner's exact instruction)

Every `pyfc/*.py` module docstring currently has a single line `Date: July 24, 2026`
(same literal date in all of them — clearly a fixed/templated stamp from when the project
was scaffolded, not meaningfully updated per file since). The repo owner wants this
replaced with two lines instead of one:
```
Created: v<YY> (July 24, 2026)
Last modified: v<XX>
```
where `<YY>` is the PyFC version the file was created in, and `<XX>` is the version it was
last meaningfully modified in (equal to `<YY>` if it hasn't been modified since creation).

### Pre-computed version timeline (do not redo this git archaeology — it's already done)

`pyproject.toml`'s `version` field has taken exactly 4 values across this repo's full
history (confirmed via `git log --all -p -- pyproject.toml`):

| Version | Introduced at (commit, date) |
|---|---|
| `0.1.0` | `1e58465`, 2026-07-24 ("Added metadata to use and install PyFC as a module") |
| `0.9.0` | `f4c49c0`, 2026-07-25 ("Changed project name to avoid PyPI conflict") |
| `0.9.2` | `04eb1d5`, 2026-07-25 ("Updated to v0.9.2") — this is `main`'s current tip |
| `0.10.0` | `fb8ffde`, 2026-07-27, **on `dev-no-templates`, unreleased** |

All 8 files with the `Date: July 24, 2026` docstring line were originally created (per
`git log --follow --diff-filter=A`) on 2026-07-23 or 2026-07-24 — i.e. before or exactly
at the point `pyproject.toml` first declared `version = "0.1.0"`. Since there is no
formal version before `0.1.0`, and the existing docstring date (July 24, 2026) already
matches when `0.1.0` was established, **use `v0.1.0` as the creation version for all 8
files** — this is a deliberate simplification, consistent with the pre-existing date
convention and with `pyproject.toml`'s actual recorded history; do not try to invent an
earlier pseudo-version.

For "last modified," each file's most recent modification commit (as of right now, on
`dev-no-templates`) was checked against `git merge-base --is-ancestor <commit> 04eb1d5`
to determine whether the change shipped at-or-before v0.9.2, or is part of the
still-unreleased v0.10.0 payload. Result (**exact values to use, already verified**):

| File | Created | Last modified |
|---|---|---|
| `pyfc/binned.py` | `v0.1.0 (July 24, 2026)` | `v0.10.0` |
| `pyfc/unbinned.py` | `v0.1.0 (July 24, 2026)` | `v0.10.0` |
| `pyfc/optimizers.py` | `v0.1.0 (July 24, 2026)` | `v0.10.0` |
| `pyfc/toys.py` | `v0.1.0 (July 24, 2026)` | `v0.10.0` |
| `pyfc/orchestrator.py` | `v0.1.0 (July 24, 2026)` | `v0.10.0` |
| `pyfc/config.py` | `v0.1.0 (July 24, 2026)` | `v0.10.0` |
| `pyfc/plotting.py` | `v0.1.0 (July 24, 2026)` | `v0.9.0` |
| `pyfc/generate_config.py` | `v0.1.0 (July 24, 2026)` | `v0.9.0` |

(`pyfc/plotting.py` and `pyfc/generate_config.py` were both last touched by commit
`9c1d2b4` "Minor modifications from ruff" — a commit made after `0.1.0` was set but
before `0.9.0` was set, so its changes shipped as part of the `0.9.0` release. Every
other file listed was touched by this session's refactor and is therefore last-modified
`v0.10.0`, still unreleased as of this writing.)

**If `pyproject.toml`'s version changes again before this branch is released** (e.g. the
repo owner decides on `1.0.0` instead of `0.10.0` — this was discussed as a possibility
during the original refactor and left as `0.10.0` per the repo owner's choice, but
re-confirm it hasn't changed since), update every `Last modified: v0.10.0` occurrence
above to match, since they all refer to "whatever version this currently-unreleased
branch ships as," not a fixed string.

### Exact line numbers to edit (re-grep `^Date: July 24, 2026` before editing in case
line numbers drifted from other work; these are accurate as of the review that produced
this brief)

| File | Line with `Date:` | Line with `Author:` |
|---|---|---|
| `pyfc/binned.py` | 10 | 11 |
| `pyfc/unbinned.py` | 14 | 15 |
| `pyfc/optimizers.py` | 47 | 48 |
| `pyfc/toys.py` | 16 | 17 |
| `pyfc/orchestrator.py` | 25 | 26 |
| `pyfc/config.py` | 13 | 14 |
| `pyfc/plotting.py` | 9 | 10 |
| `pyfc/generate_config.py` | 33 | 34 |

Example transformation (using `pyfc/binned.py` as the template — apply the analogous
substitution to all 8 files using each file's own Created/Last-modified values from the
table above):

Before:
```
Date: July 24, 2026
Author: Mauricio Bustamante (mbustamante@gmail.com)
```
After:
```
Created: v0.1.0 (July 24, 2026)
Last modified: v0.10.0
Author: Mauricio Bustamante (mbustamante@gmail.com)
```

### `pyfc/__init__.py`

This file currently has **no** `Date:`/`Author:` header at all (it's a 3-line docstring
with no metadata block). It was effectively created alongside `pyproject.toml`'s first
`0.1.0` commit (`1e58465`, 2026-07-24) and was last modified this session (`v0.10.0`, this
refactor's top-level-export work). Adding a matching header here is optional/additive
(not required by the finding, which was specifically about the 8 files that already have
a now-stale date) — your call whether to add one for consistency; if you do, use `Created:
v0.1.0 (July 24, 2026)` / `Last modified: v0.10.0` / `Author: Mauricio Bustamante
(mbustamante@gmail.com)` to match the others.

### Verification

This is a docstring-only, zero-behavioral-risk change. After editing, run `pytest tests/
-v` once (should be unaffected, but confirm) and do a final `grep -rn "^Date: July 24,
2026" pyfc/` to confirm zero remaining hits.

---

## 10. New request: auto-sync `CHANGELOG.md` → `docs/source/changelog.rst`

The repo owner asked: *"would it be possible to add a workflow that translates the
CHANGELOG.md into changelog.rst every time CHANGELOG.md is updated?"* This was raised
independently of the 9 findings above — treat it as a tenth, separate deliverable.

### Why this isn't a trivial mechanical translation

Throughout the refactor, `CHANGELOG.md` (Markdown) and `docs/source/changelog.rst` (RST)
were kept in parallel **by hand**, bullet-for-bullet, and they currently match exactly in
structure (verified: both have 13 bullet points under the `[0.10.0]`/`0.10.0` heading).
But `changelog.rst` also has **RST-only enhancements that don't exist in the Markdown
source and can't be mechanically derived from it**:
- A `.. warning::` admonition wrapping the "NLL values in the previously-unphysical
  region will differ from prior runs" callout.
- A `.. warning::` admonition wrapping the "Migration hazard" callout in the BREAKING
  CHANGES section.
- `:ref:`joint-simplex-constraints`` and `:doc:`quickstart`` Sphinx cross-references,
  which have no Markdown equivalent (README.md just uses a plain anchor link
  `#handling-jointsimplex-constrained-parameters` for the same concept).

A naive Markdown→RST converter (e.g. bare `pandoc -f markdown -t rst CHANGELOG.md`) would
reproduce the bullet text and headings fine, but would **flatten these admonitions back
into plain paragraphs and drop the Sphinx cross-references** — i.e. full, unattended
automation would silently regress the RST doc's quality every time it ran, unless you
build in a way to preserve or reintroduce those enhancements.

### Recommended approach: a CI *check*, not an auto-fix (present this choice explicitly
to the repo owner before implementing — don't just pick one silently)

**Option A (recommended, safer, less to build/maintain):** A new GitHub Actions workflow
(e.g. `.github/workflows/changelog-sync.yml`) that triggers on `push`/`pull_request` when
`CHANGELOG.md` changes, and runs a small conversion script that structurally compares
`CHANGELOG.md` against `docs/source/changelog.rst` (e.g.: same version headings present
in the same order, same number of top-level bullets per section, same bullet *text*
modulo Markdown-vs-RST inline-formatting differences like `` `code` `` vs `` ``code`` ``)
and **fails the build with a clear message** if they've drifted, prompting a human (or an
AI session) to manually reconcile — i.e. run the same conversion script locally to
regenerate the RST body, then hand-restore/re-check the admonitions and cross-references,
exactly as was done by hand throughout this refactor. This avoids `contents: write`
permissions, auto-commit-loop-avoidance complexity, and the risk of silently degrading the
RST doc's quality on every sync.

**Option B (more automation, more complexity/risk):** A workflow that actually
regenerates `docs/source/changelog.rst`'s content and commits it back automatically.
This needs: `contents: write` permission; a path-filtered trigger plus a
`git diff --quiet` no-op guard (or a `[skip ci]` commit-message marker) to avoid an
infinite trigger loop when the bot's own commit touches `CHANGELOG.md`-adjacent paths; and
critically, **some mechanism to preserve the RST-only admonitions/cross-references**
across regenerations — e.g. only auto-syncing content between two marker comments in
`changelog.rst` (`.. changelog-sync-start` / `.. changelog-sync-end`) and leaving
hand-added enhancements outside those markers permanently, which only works if
admonitions never need to live *inside* an auto-synced bullet (they currently do, e.g.
wrapping specific callouts within a bullet's body) — so this option likely still needs
occasional manual touch-up even with markers, just less often than full manual sync.

**Suggested starting point regardless of which option is chosen**: try `pandoc -f gfm -t
rst CHANGELOG.md` first (if `pandoc` is available in the environment) and diff its output
against the current `docs/source/changelog.rst` structure, to see how close a completely
generic converter gets before deciding whether to hand-roll a bespoke script. If pandoc's
output is structurally close (same headings, bullets, inline-code handling) modulo the
admonitions/cross-references, that's a strong signal Option A's "structural diff, fail on
drift" check can just shell out to pandoc internally rather than needing custom
Markdown-parsing code. If pandoc handles this repo's specific CHANGELOG conventions
poorly (e.g. mishandles the nested sub-bullets under "compute_rates_func signature
changes for both likelihood types," or the multi-paragraph bullets), a small
purpose-built script covering only this changelog's actual patterns (version headings,
subheadings, single-level bullets, inline single-backtick code, occasional nested
bullets) will likely be more reliable than fighting a generic tool's edge cases.

**Ask the repo owner to confirm Option A vs. B (recommend A) before implementing**, since
this is a real design tradeoff with no universally-correct answer, not something to
silently decide.

### Acceptance bar for whichever option is built

- The new workflow must not break existing CI (`pytest.yml`, `lint.yml`, `pages.yml`,
  `publish.yml` should all remain green).
- Test it end-to-end: make a trivial `CHANGELOG.md` edit (e.g. a new bullet under a new
  `## [Unreleased]` section, added and then removed again after testing, or use a
  throwaway branch) and confirm the new workflow behaves as designed (fails the check /
  or successfully auto-commits, depending on which option) before considering this done.

---

## 11. Suggested milestone breakdown

Mirroring this repo's established incremental-commit discipline (one reviewable commit
per logical change, not one giant commit):

1. Finding #1 (the `np.ascontiguousarray` fix + new regression test) — highest priority,
   do this first, in its own commit. Full test suite + cross-branch harness must stay
   green.
2. Findings #2, #3 (pyproject.toml `docs` extra typo; README stale path) — trivial,
   independent, can be one small commit together or two tiny ones.
3. Findings #5, #6 (`.gitignore` additions/cleanup) — one small commit.
4. Finding #4 (`REFACTOR_BRIEF_dev-no-templates.md` / this file's disposition) — **ask the
   repo owner first**, then act; commit whatever the decision was.
5. Finding #7 (`__init__.py` stale comment) — trivial, own commit or folded into #9's
   commit since both touch docstring-adjacent content in nearby files.
6. Finding #8 (`plotting.py` dead code) — trivial, own commit.
7. Finding #9 (docstring Created/Last-modified headers, all 8 files) — mechanical, use the
   pre-computed table above verbatim, one commit.
8. The CHANGELOG-sync workflow (§10) — **confirm Option A vs B with the repo owner
   first**, then implement and test end-to-end, own commit(s).
9. Final pass: `pytest tests/ -v` (44 tests must pass), `bash
   xbranch_compare/run_comparison.sh` (zero mismatches), `git worktree list` (must stay
   clean — remember to clean up if the changelog-sync workflow testing used a throwaway
   branch/worktree), final `git status --short` review before considering this done.

## 12. Final acceptance checklist

- [ ] Finding #1 fixed at all 5 call sites in `pyfc/binned.py`; new regression test added
      and passing; full test suite + cross-branch harness green.
- [ ] Finding #2: `pyproject.toml`'s `docs` extra fixed to `sphinx-rtd-theme`.
- [ ] Finding #3: README.md's stale `fc_output/checkpoint_fc.npz` path corrected.
- [ ] Finding #4: `REFACTOR_BRIEF_dev-no-templates.md` (and this file) disposition
      decided **with the repo owner's explicit input**, not unilaterally, and acted on.
- [ ] Finding #5: `.gitignore` covers `output/` (and defensively `fc_output/`).
- [ ] Finding #6: `.gitignore`'s stale `PyFC.egg-info/` line removed.
- [ ] Finding #7: `__init__.py`'s stale comment removed/replaced.
- [ ] Finding #8: `plotting.py`'s two dead-code remnants removed.
- [ ] Finding #9: all 8 files' docstrings updated to the Created/Last-modified format
      using the pre-computed version table; `grep -rn "^Date: July 24, 2026" pyfc/`
      returns zero hits.
- [ ] CHANGELOG-sync workflow: approach (Option A vs B) confirmed with the repo owner,
      implemented, tested end-to-end.
- [ ] `pytest tests/ -v` green (44 tests, or more if you added any).
- [ ] `bash xbranch_compare/run_comparison.sh` shows zero mismatches.
- [ ] `git worktree list` clean.
- [ ] `git status --short` reviewed, nothing unexpected left uncommitted or untracked.
