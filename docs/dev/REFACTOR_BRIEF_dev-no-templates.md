# Brief: Remove `S_model`/`B_model` templates + support N-dimensional data

**Purpose of this document**: this is a handoff brief for a fresh chat session. Paste
this whole file (or point Claude at its path) to start that session. It was written
by an assistant with full context of the current codebase after a design discussion
with the repo's author (Mauricio Bustamante). Everything in this brief reflects
decisions already made in that discussion, plus a small number of items explicitly
marked **OPEN DECISION** that still need the author's sign-off before or during
implementation.

Repository: `/home/mbustamante/Research/FeldmanCousins` (PyFC, GPLv3, sole author is
the repo owner). Base branch for this work: **`dev-no-templates`**, already created
off `dev` (no commits on it yet — it currently points at the same commit as `dev`,
which is `17f3ffd` at the time of writing). Do the work there. Do **not** touch
`main` or `dev` directly.

---

## 0. Why this refactor (context from the design discussion)

`compute_fc_intervals(data, S_model, B_model, grids, compute_rates_func=None, ...)`
currently takes `S_model`/`B_model` as required positional arguments, separate from
the user-supplied `compute_rates_func` (which maps `params -> expected rates`). An
audit (file:line citations below) established:

- **Binned mode**: `calc_nll` never reads `S_template`/`B_template` itself — it only
  forwards them as positional args into `compute_rates_func`, plus three internal
  spots use only `len(S_template)` for shape/sizing (not values). This means
  `S_model`/`B_model` are almost pure legacy pass-through for binned models: a user's
  `compute_rates_func` could just as well close over its own template arrays (module
  globals or closures — confirmed numba-`@njit`-compatible; this exact pattern is
  already used in `examples/pyfc_joint_constraints_tutorial.ipynb`'s `TEMPLATE_TAU`).
- **Unbinned mode**: `S_model`/`B_model` are *callables* (PDFs), each evaluated
  **exactly once** per fit and the resulting arrays reused across every subsequent
  NLL evaluation in that fit (measured: 68 calls to `calc_nll_unbinned` per single
  `unconditional_fit_scipy` call, all reusing one `S_model(data)` evaluation). This
  is a genuine, load-bearing performance optimization, not legacy cruft. It must be
  preserved — but the *hardcoded pair* (`S_model`, `B_model`) is itself an
  unnecessary rigidity: signal/background is just "2 components"; a general
  `pdf_components` **list** of any length is a strict generalization that keeps the
  caching property while dropping the forced 2-way split and the confusing fact that
  `S_model`/`B_model` mean *arrays* for binned but *callables* for unbinned (same
  parameter name, different semantics depending on `likelihood_type` — a design
  wart in itself).
- A single combined PDF (`pdf(data)` evaluated once, no split) does **not** work as
  a replacement: the relative signal/background weighting is itself a free fit
  parameter, so it cannot be pre-combined before the optimizer runs without either
  baking in a fixed mixing ratio (wrong) or re-evaluating on every iteration
  (defeats the whole point of the caching).

Conclusion agreed with the author: remove `S_model`/`B_model` for binned entirely;
replace them with a generalized `pdf_components` (list of callables) for unbinned.
This is a **breaking API change** (signature of `compute_fc_intervals` and of every
user-supplied `compute_rates_func` changes) — treat it as such: version bump,
CHANGELOG "BREAKING CHANGES" heading, migration notes, and a dedicated cross-branch
regression harness (Part C below) proving the *only* thing that changed is the
signature, not the numerics.

Separately, the author also wants `data` (and the arrays `compute_rates_func`
produces/consumes) to support **arbitrary N-dimensional shapes**, not just flat 1D
vectors (e.g. a genuine 2D `(E, cos_theta)` binned histogram, evaluated directly
without the user having to flatten it first). This is bundled into the same branch
since it touches the same functions. **Important scope boundary**: this is about
the dimensionality of *data/observations*, not the dimensionality of the *parameter
scan* — `grids` (one 1D array per fit parameter, `n_params` of them) is a completely
separate, already-N-parameter-general concept and must not be touched.

---

## 1. Final API design (decided)

### 1.1 `compute_fc_intervals` new signature

Current (orchestrator.py:240-250):
```python
def compute_fc_intervals(data, S_model, B_model, grids, compute_rates_func=None, generate_toy_func=None,
                         cl=None, n_toys=2000, strategy="scipy", num_cores=None, verbose=1,
                         adaptive_toys=True, toy_batch_size=200,
                         sparsify_grid=True, warm_start=True,
                         likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                         output_file=None, save_log=False, save_directory="fc_output",
                         use_finite_mc_correction_binned=True, S_sigma2=None, B_sigma2=None,
                         compute_1D_intervals=True, compute_2D_intervals=True, param_names=None,
                         smooth_1d=False, smooth_2d=False, bounds_func=None,
                         constraints=None, scipy_method=None,
                         n_restarts=1, neighbor_seeding=True):
```

New:
```python
def compute_fc_intervals(data, grids, compute_rates_func=None, generate_toy_func=None,
                         pdf_components=None,
                         cl=None, n_toys=2000, strategy="scipy", num_cores=None, verbose=1,
                         adaptive_toys=True, toy_batch_size=200,
                         sparsify_grid=True, warm_start=True,
                         likelihood_type="binned", S_mc_pool=None, B_mc_pool=None,
                         output_file=None, save_log=False, save_directory="fc_output",
                         use_finite_mc_correction_binned=True, S_sigma2=None, B_sigma2=None,
                         compute_1D_intervals=True, compute_2D_intervals=True, param_names=None,
                         smooth_1d=False, smooth_2d=False, bounds_func=None,
                         constraints=None, scipy_method=None,
                         n_restarts=1, neighbor_seeding=True):
```

Only `data` and `grids` remain positional-meaningful (2 required positional args,
down from 4). `compute_rates_func` stays keyword-with-`None`-default (matching the
existing pattern where it's validated inside the body, not via a required-positional
slot — see orchestrator.py:380-381). Add analogous validation for `pdf_components`:

```python
if compute_rates_func is None:
    raise ValueError("You must provide a valid `compute_rates_func` to define your physical model.")
if likelihood_type == "unbinned" and not pdf_components:
    raise ValueError("You must provide a non-empty `pdf_components` list (of callables) for likelihood_type='unbinned'.")
if likelihood_type == "binned" and pdf_components:
    warnings.warn("`pdf_components` was supplied but is unused for likelihood_type='binned'; ignoring it.")
```
(add `import warnings` to orchestrator.py if not already present — it is not,
currently; check before adding, since other files in this repo already do
`import warnings` e.g. optimizers.py).

**MIGRATION HAZARD — read this twice**: because `S_model`/`B_model` are being
*removed* (not just renamed) from position 2/3, and `grids` moves from position 4
to position 2, any caller still using positional args will silently pass the wrong
value into the wrong parameter (e.g. what used to be `grids` landing in
`compute_rates_func`'s slot) rather than raising an error. **Every single call site
in this repo — source, tests, docs, notebooks — must be converted to fully
keyword-form calls** (`compute_fc_intervals(data=..., grids=..., compute_rates_func=...,
...)`) as part of this refactor, both to fix the immediate breakage and so this
class of hazard can't recur on a future signature change.

### 1.2 Binned `compute_rates_func` new signature

Current convention (used throughout binned.py, optimizers.py, toys.py, and every
binned example):
```python
def compute_rates_func(params, S_template, B_template, S_sigma2, B_sigma2) -> (mu, sigma2)
```

New:
```python
def compute_rates_func(params, S_sigma2, B_sigma2) -> (mu, sigma2)
```
Users who need fixed template arrays reference them via closure/module-global
instead of receiving them as arguments, e.g.:
```python
TEMPLATE = np.array([...])   # module-level constant

@njit(fastmath=True, nogil=True)
def my_compute_rates(params, S_sigma2, B_sigma2):
    mu = params[0] * TEMPLATE + params[1]
    return mu, S_sigma2   # or however sigma2 is derived
```
`mu` (and `sigma2`) may now be **any shape** (see Part B) — it no longer needs to
match a specific `S_template` shape since there is no `S_template` argument; it
only needs to match `data`'s shape (see Part B's shape-consistency note).

**OPEN DECISION #1**: `S_sigma2`/`B_sigma2` are, structurally, in the exact same
"basically pass-through" position that `S_model`/`B_model` were in (calc_nll never
reads them directly either — see binned.py:111, only the *returned* `sigma2_arr` is
used). The author did not ask to remove/rename these in the design discussion — only
`S_model`/`B_model` were discussed. **Do not remove or rename `S_sigma2`/`B_sigma2`
in this PR.** Flag this symmetry explicitly to the author as a natural follow-up
(their `S_`/`B_` naming now dangles without an `S_model`/`B_model` pair to justify
it), but leave them as-is for this task — confirm with the author before touching
them if it comes up mid-implementation.

### 1.3 Unbinned `compute_rates_func` new signature

Current:
```python
def compute_rates_func(params, s_probs, b_probs) -> (expected_total, p_events)
```

New:
```python
def compute_rates_func(params, probs) -> (expected_total, p_events)
```
where `probs` is a `list[np.ndarray]` (or `tuple[np.ndarray, ...]` — pick one and be
consistent; **recommend `list`**, since `unbinned.py` is pure Python, not numba —
there is no numba-tuple-homogeneity constraint here to worry about), one entry per
`pdf_components[i]`, each pre-evaluated **once** per fit via `pdf(data)`. Example
(2-component signal+background, replicating today's behavior exactly):
```python
def my_compute_rates_unbinned(params, probs):
    s_probs, b_probs = probs[0], probs[1]
    p_events = params[0] * s_probs + params[1] * b_probs
    return params[0] + params[1], p_events
```
3+ components (new capability, impossible to express today):
```python
def my_compute_rates_unbinned(params, probs):
    total = 0.0
    p_events = np.zeros_like(probs[0]) if len(probs[0]) else np.array([])
    for i, p in enumerate(probs):
        total += params[i]
        p_events = p_events + params[i] * p if len(p_events) else params[i] * p
    return total, p_events
```

### 1.4 `pdf_components` contract

```python
pdf_components: list[callable]   # each: pdf(data) -> np.ndarray of shape matching len(data)/events
```
Passed to `compute_fc_intervals` (unbinned mode only, required, no default that
makes sense — must error if missing per §1.1). Internally, evaluated once per fit as
`probs = [pdf(data) for pdf in pdf_components]` (mirroring today's
`s_probs = S_model(data); b_probs = B_model(data)` pattern) at every one of the call
sites enumerated in §2.3 below.

---

## 2. File-by-file, function-by-function change catalog

All line numbers below are current `dev`-branch (commit `17f3ffd`) line numbers —
they will drift slightly as edits land; re-grep as you go (`grep -n "S_model\|B_model"
pyfc/*.py` is your friend throughout this PR).

### 2.1 `pyfc/binned.py`

- **`calc_nll`** (binned.py:41-150). Signature:
  `calc_nll(params, N_obs, S_template, B_template, S_sigma2, B_sigma2, use_finite_mc, compute_rates_func)`
  → `calc_nll(params, N_obs, S_sigma2, B_sigma2, use_finite_mc, compute_rates_func)`.
  Body change at binned.py:111: `mu, sigma2_arr = compute_rates_func(params, S_template, B_template, S_sigma2, B_sigma2)`
  → `mu, sigma2_arr = compute_rates_func(params, S_sigma2, B_sigma2)`.
  **Also** (Part B, N-D support): immediately after that line, flatten both arrays
  before the per-bin loop:
  ```python
  mu_flat = mu.reshape(-1)
  N_obs_flat = N_obs.reshape(-1)
  sigma2_flat = sigma2_arr.reshape(-1) if use_finite_mc else sigma2_arr  # avoid reshaping when unused; see note below
  for i in range(len(N_obs_flat)):
      mu_i = mu_flat[i]
      n_obs = float(N_obs_flat[i])
      ...  # use sigma2_flat[i] instead of sigma2_arr[i] at binned.py:131
  ```
  (Rename the loop's `mu[i]`/`N_obs[i]`/`sigma2_arr[i]` reads at binned.py:114-115,131
  to `mu_flat[i]`/`N_obs_flat[i]`/`sigma2_flat[i]`.) **Numba check needed**:
  `.reshape(-1)` on a non-C-contiguous array can raise in nopython mode; if that
  surfaces during testing, fall back to `np.ascontiguousarray(x).reshape(-1)` — note
  this as a thing to verify empirically, not something to pre-emptively guard
  without evidence it's needed (adds cost on the hot path otherwise).
  **Shape-consistency validation (recommended addition)**: right after computing
  `mu_flat`/`N_obs_flat`, consider `if mu_flat.shape[0] != N_obs_flat.shape[0]: raise
  ValueError(...)` for a clear error instead of an obscure indexing crash — verify
  numba nopython mode's exception-message support (static strings are always safe;
  f-strings/dynamic messages are supported in recent numba but confirm against the
  numba version pinned in `pyproject.toml`).
  Update the docstring (binned.py:66-91) to reflect the new signature, that
  `N_obs`/`mu` may be any shape (not just 1D), and drop the now-stale
  `S_template`/`B_template` parameter docs.

- **`unconditional_fit_grid`** (binned.py:154-201), **`conditional_fit_grid_1d`**
  (binned.py:204-258), **`conditional_fit_grid_2d`** (binned.py:261-313): each only
  forwards `S_template`/`B_template` opaquely into `calc_nll` (confirmed — no direct
  indexing). Just drop `S_template, B_template` from each signature and from the
  `calc_nll(...)` call inside (binned.py:196, 253, 308) plus docstrings.

- **`generate_and_fit_toys_grid_1d`** (binned.py:317-378) and
  **`generate_and_fit_toys_grid_2d`** (binned.py:380-432): drop `S_template,
  B_template` from signature; update the two internal `compute_rates_func(...)` /
  `calc_nll`-family calls accordingly (binned.py:367,374-375 and 421,428-429). **Also**
  (Part B): replace
  ```python
  n_bins = len(S_template)
  mu_true, _ = compute_rates_func(true_params, S_template, B_template, S_sigma2, B_sigma2)
  ...
  for t in prange(n_toys):
      toy_N = np.zeros(n_bins)
      for i in range(n_bins):
          toy_N[i] = np.random.poisson(mu_true[i])
  ```
  with
  ```python
  mu_true, _ = compute_rates_func(true_params, S_sigma2, B_sigma2)
  mu_true_flat = mu_true.reshape(-1)
  n_bins = mu_true_flat.shape[0]
  ...
  for t in prange(n_toys):
      toy_N = np.zeros(n_bins)
      for i in range(n_bins):
          toy_N[i] = np.random.poisson(mu_true_flat[i])
  ```
  (`toy_N` stays flat — no need to reshape it back to N-D, since `calc_nll` flattens
  its own `N_obs` input regardless of the shape it's handed; a flat array reshaped
  by `.reshape(-1)` is simply itself, a no-op. Confirmed sufficient: `calc_nll`'s own
  internal flattening of `mu`/`N_obs` is what makes shape irrelevant here, not
  anything the toy generator needs to do.)

### 2.2 `pyfc/unbinned.py`

- **`calc_nll_unbinned`** (unbinned.py:25-85). Signature:
  `calc_nll_unbinned(params, len_obs, s_probs, b_probs, compute_rates_func)` →
  `calc_nll_unbinned(params, len_obs, probs, compute_rates_func)`. Body change at
  line 72 (currently `compute_rates_func(params, s_probs, b_probs)`) →
  `compute_rates_func(params, probs)`. No N-D concerns here — already confirmed
  agnostic to event feature-dimensionality (see Part B.2). Update docstring
  (unbinned.py:36-51) accordingly.

- **`unconditional_fit_grid_unbinned`** (unbinned.py:89-130): signature
  `(obs_events, S_pdf, B_pdf, full_grid_points, compute_rates_func)` →
  `(obs_events, pdf_components, full_grid_points, compute_rates_func)`. Body
  (unbinned.py:117-118, currently `s_probs = S_pdf(obs_events)...; b_probs =
  B_pdf(obs_events)...`) → `probs = [pdf(obs_events) if len_obs > 0 else np.array([])
  for pdf in pdf_components]`. Update the `calc_nll_unbinned(...)` call at line 125.

- **`conditional_fit_grid_unbinned_1d`** (unbinned.py:132-187) and
  **`conditional_fit_grid_unbinned_2d`** (unbinned.py:189-242): identical treatment,
  same substitution pattern (lines 162-163/219-220 for the pre-evaluation, line
  182/237 for the `calc_nll_unbinned` call).

- **`generate_and_fit_toys_grid_unbinned_1d`** (unbinned.py:246-294) and
  **`generate_and_fit_toys_grid_unbinned_2d`** (unbinned.py:296-...): drop
  `S_pdf, B_pdf` from signature, add `pdf_components`; update the two internal
  `unconditional_fit_grid_unbinned`/`conditional_fit_grid_unbinned_*` calls
  (unbinned.py:289-290 and the 2d equivalent) to pass `pdf_components` through
  instead of `S_pdf, B_pdf`.

### 2.3 `pyfc/optimizers.py`

Six functions, **identical treatment pattern** in each: signature drops
`S_model, B_model`, gains `pdf_components=None`; the unbinned branch's
pre-evaluation (`s_probs = S_model(data)...; b_probs = B_model(data)...`) becomes
`probs = [pdf(data) if len_obs > 0 else np.array([]) for pdf in pdf_components]`;
the binned branch's `calc_nll(p, data, S_model, B_model, S_sigma2, B_sigma2,
use_finite_mc, compute_rates_func)` calls drop `S_model, B_model`; the unbinned
branch's `calc_nll_unbinned(p, len_obs, s_probs, b_probs, compute_rates_func)` calls
become `calc_nll_unbinned(p, len_obs, probs, compute_rates_func)`.

- `unconditional_fit_scipy` (optimizers.py:271-372): pre-eval at 344-345, binned
  `calc_nll` call at 351.
- `conditional_fit_1d_scipy` (optimizers.py:373-493): pre-eval at 451-452, calls at
  466.
- `conditional_fit_2d_scipy` (optimizers.py:494-608): pre-eval at 561-562, calls at
  578.
- `unconditional_fit_ultranest` (optimizers.py:610-691): pre-eval at 673-674, calls
  at 684.
- `conditional_fit_1d_ultranest` (optimizers.py:692-795): **two** pre-eval sites —
  the degenerate 0-free-param early-return branch (752-753, feeding into the call at
  756) and the main path (763-764, feeding the call at 782). Don't miss the
  degenerate branch.
- `conditional_fit_2d_ultranest` (optimizers.py:796-...): same two-site pattern
  (853-854/857 and 864-865/885).

Update every one of these six functions' docstrings (each currently documents
`S_model, B_model : callable or array_like` — replace with `pdf_components : list of
callable, optional`).

### 2.4 `pyfc/toys.py`

- **`_worker_unbinned_toy`** (toys.py:38-92): the packed-args tuple (toys.py:53,63)
  currently carries `S_model, B_model`; replace with `pdf_components`. Update the
  docstring's tuple description (toys.py:52-54) and the six downstream
  `unconditional_fit_scipy`/`conditional_fit_*_scipy`/`conditional_fit_*_ultranest`
  calls (toys.py:75,80,83,86,88,90) to pass `pdf_components=pdf_components` instead
  of `S_model, B_model` positionally.

- **`generate_and_fit_toys_python`** (toys.py:95-...): signature currently has
  `S_model, B_model, bounds_list` as positional params (toys.py:96) — replace with
  `bounds_list` alone, plus a new `pdf_components=None` keyword param. Docstring at
  toys.py:131. The args-tuple construction for the unbinned/multiprocessing branch
  (toys.py:174) needs `pdf_components` substituted in for `S_model, B_model`. The
  **binned** branch (toys.py:185-186):
  ```python
  mu_true, _ = compute_rates_func(true_params, S_model, B_model, S_sigma2, B_sigma2)
  toys_binned_data = np.random.poisson(mu_true, size=(n_toys, len(S_model)))
  ```
  → (drop S_model/B_model from the call; generalize the `size=` tuple for N-D per
  Part B):
  ```python
  mu_true, _ = compute_rates_func(true_params, S_sigma2, B_sigma2)
  toys_binned_data = np.random.poisson(mu_true, size=(n_toys,) + mu_true.shape)
  ```
  Note `toys_binned_data[t]` (used later, e.g. toys.py's `fit_single_toy`) now has
  shape `mu_true.shape` (whatever N-D shape the user's model produces) instead of
  always being a 1D row — this flows correctly into `unconditional_fit_scipy`/etc.
  as `toy_data` since those treat `data` opaquely aside from `len()`/PDF-evaluation
  (unbinned path only; binned path passes it straight to `calc_nll` which now
  flattens internally regardless of shape).
  Ten scipy/ultranest call sites inside `fit_single_toy` (toys.py:196,200,203,206,
  209,211) need the same `S_model, B_model` → `pdf_components=pdf_components`
  substitution as §2.3.

### 2.5 `pyfc/orchestrator.py`

- **`compute_fc_intervals`** signature and docstring: see §1.1 in full.

- **Sigma2 defaults** (orchestrator.py:411-413):
  ```python
  if likelihood_type == "binned":
      if S_sigma2 is None: S_sigma2 = np.zeros_like(S_model)
      if B_sigma2 is None: B_sigma2 = np.zeros_like(B_model)
  ```
  → replace `S_model`/`B_model` with `data` (Part B: `data` is always present,
  non-optional, and — per the shape-consistency contract in §1.2 — must already
  have the same shape `compute_rates_func`'s `mu` will produce):
  ```python
  if likelihood_type == "binned":
      if S_sigma2 is None: S_sigma2 = np.zeros_like(data)
      if B_sigma2 is None: B_sigma2 = np.zeros_like(B_model)  # <- fix both lines
  ```
  i.e. both become `np.zeros_like(data)`.

- **All ~19 call sites** forwarding `S_model, B_model` (orchestrator.py:503, 505,
  507, 509, 545, 547, 549, 562, 577, 579, 581, 644, 646, 648, 650, 658, 660, 662,
  664, 672, 674, 676 — every `unconditional_fit_*`/`conditional_fit_*`/
  `generate_and_fit_toys_*` call in Phase 0/1/2): drop `S_model, B_model` from each,
  add `pdf_components=pdf_components` to the scipy/ultranest/toys.py calls (not the
  grid-strategy calls, which don't need it since grid functions get `pdf_components`
  directly per §2.1/§2.2, not through this parameter name — **double check naming
  consistency**: the grid-path functions in binned.py/unbinned.py should also accept
  a parameter literally named `pdf_components` for the unbinned grid functions, to
  keep the vocabulary uniform across the whole codebase). Grep for
  `unconditional_fit_grid\|conditional_fit_grid\|unconditional_fit_scipy\|
  conditional_fit_.*_scipy\|unconditional_fit_ultranest\|conditional_fit_.*_ultranest\|
  generate_and_fit_toys` in orchestrator.py to enumerate precisely; there are more
  sites than in the audit above once you include the "rerun" branches in Phase 2's
  `eval_2d_point` (orchestrator.py:658-664) — re-grep, don't trust this list as
  exhaustive without re-verifying against the actual file at implementation time.

- **`__main__` demo block** (orchestrator.py:~798-859 for the binned example,
  further down for unbinned): both `example_compute_rates_binned` and
  `example_compute_rates_unbinned` need full signature rewrites (drop
  `S_template, B_template` args / merge `s_probs, b_probs` into `probs`), and both
  `compute_fc_intervals(...)` calls (orchestrator.py:~840-ish and the unbinned
  equivalent) need converting to the new keyword-only, `pdf_components`-based call
  shape. **Also convert the unbinned demo to demonstrate `pdf_components=[s_pdf,
  b_pdf]`** (a natural, low-risk place to show the new API in the flesh, since it's
  literally the file that defines it).

### 2.6 Explicitly unaffected (verify, don't blindly trust)

- `pyfc/plotting.py`, `pyfc/config.py`, `pyfc/generate_config.py`, `pyfc/__init__.py`:
  confirmed zero references to `S_model`/`B_model`/`S_template`/`B_template` in the
  current audit. Re-grep at the start of this work to confirm still true (in case
  anything changed since), but expect no changes needed here.
- The `results` dict / `.npz` checkpoint structure, `plotting.py`'s corner-plot
  logic, and the `sparsify_grid` 2D spline interpolation: all operate on
  **parameter-grid**-shaped arrays (indexed by `grids`/`n_params`), never on
  `data`'s own shape. Confirm this remains true but expect no changes.

---

## 2b. Naming: is `pdf_components` final?

**OPEN DECISION #2** (minor, but worth a beat): the author only said "something more
descriptive than `S_model`/`B_model`", floating `pdf` as an example. This brief
recommends `pdf_components` (plural, since it's a list — a singular `pdf` name reads
oddly for a 2+-element list). If the author prefers different wording (`pdfs`,
`density_functions`, `component_pdfs`), it's a pure rename with no semantic
consequences — settle it in the first exchange of the new session rather than
guessing further.

---

## Part B: N-dimensional data support — summary of what's needed

(Already interleaved into §2 above; this section is the "why" and the parts of the
codebase confirmed to need **no** change.)

- **Binned**: real change required. `calc_nll`'s per-bin loop assumed 1D
  (`len(N_obs)`, `mu[i]`, `N_obs[i]`); fixed via internal `.reshape(-1)` flattening
  (§2.1). The toy generators' `len(S_template)`/elementwise Poisson sampling assumed
  1D `mu_true`; fixed the same way. `orchestrator.py`'s sigma2-default construction
  switches from `np.zeros_like(S_model)` to `np.zeros_like(data)` (shape-agnostic
  either way, this substitution is required regardless by §2.5, and happens to also
  be N-D-safe for free). **Key invariant to preserve and test**: NLL is a sum over
  all bins regardless of how they're laid out spatially — a (10,20) 2D histogram and
  its (200,) flattened equivalent must give byte-identical NLL for the same
  `compute_rates_func`. Write a same-branch test asserting exactly this
  (`test_nd_binned.py`, see §4).
- **Unbinned**: already almost fully compatible. `calc_nll_unbinned` never touches
  `data`'s shape directly (only `probs`, which is always 1D — one density value per
  event — regardless of how many features describe each event). `len_obs = len(data)`
  already does the right thing for `data.shape == (n_events, n_features)` (numpy
  `len()` returns the first-axis size). **Convention to document, not enforce in
  code**: events are indexed along `data`'s first axis. **Known gotcha to document**
  (not a PyFC-internals bug, but will bite users): the tutorial's bootstrap pattern
  `np.random.choice(mc_pool, size=n, replace=True)` only works for a 1D pool;
  resampling N-D event feature vectors requires index-based resampling instead:
  `idx = np.random.choice(len(mc_pool), size=n, replace=True); toy_events =
  mc_pool[idx]`. Update the quickstart/tutorial `generate_toy_func` example to show
  this pattern (or at minimum add a callout note), since it's silently wrong
  otherwise for N-D pools.
- **Explicitly out of scope / unaffected**: `grids` (list of 1D per-parameter test-value
  arrays), `n_params`, the 1D/2D *parameter* scan machinery, `results` dict
  structure, `.npz`/`.json` output schema, `plotting.py`. These are a different,
  already-N-general axis (number of *fit parameters*) and must not be conflated with
  the data-dimensionality work here.

---

## Part C: Cross-branch comparison harness (behavioral regression gate)

**Goal**: prove, empirically, that `dev` and `dev-no-templates` produce
numerically-identical results for equivalent models — the refactor changed
*signatures only*, never *numerics*.

### C.1 Why a harness (not just unit tests) is needed

The two branches define an incompatible Python API under the same package name
`pyfc` — you cannot `import` both in one process. Use **git worktrees** (already a
proven pattern in this repo's history — see the "Full regression" work done for the
constraint-handling PR, which used exactly this approach and a JSON-diff comparator;
grep this repo's git log / prior session transcripts for `git worktree add` if you
want the historical precedent) to check out each branch into its own directory, run
an equivalent script in each, and diff the saved results externally.

### C.2 Harness design

1. **One script per branch style**, e.g. `scripts/xbranch_old_api.py` (targets
   `dev`'s API) and `scripts/xbranch_new_api.py` (targets `dev-no-templates`'s API).
   Keep them in a shared location (recommend a new top-level `xbranch_compare/`
   directory in the repo, not committed permanently unless the author wants it kept
   — ask; default to keeping it, since it's a valuable regression artifact for
   future signature changes too).
2. **Both scripts must define numerically-identical models.** To avoid transcription
   drift masquerading as a "real" discrepancy, define the shared numeric constants
   (template arrays, PDF shapes, RNG seeds, grids, `n_toys`) **once**, in a small
   `xbranch_compare/shared_constants.py` importable by both scripts (this file has no
   `pyfc` dependency, so it's identical/importable regardless of which worktree's
   `pyfc` is on `sys.path`).
3. **Defensive `sys.path` handling**: at the top of each script,
   `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` is NOT enough —
   you additionally need the *worktree root* (containing `pyfc/`) on `sys.path`. Do
   not rely on cwd-based implicit resolution (this bit a previous session — running
   a script from a subdirectory silently imported a *different*, stale, pip-installed
   `pyfc` instead of the local checkout). Be explicit:
   ```python
   import sys, os
   sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
   ```
   assuming the scripts live one level below the worktree root — adjust to wherever
   you actually place them, but **verify** with `import pyfc; print(pyfc.__file__)`
   at the top of a throwaway run before trusting any numeric output, exactly as was
   necessary in prior work in this repo (a stale PyPI-installed 0.9.2 `pyfc` in one
   conda env silently shadowed the local checkout once already — do not repeat that
   mistake un-noticed).
4. **Scenarios to cover** (write all of these; do not stop at the trivial case):
   - (a) Binned, 1D data, 3-parameter model — direct analog of the existing
     `orchestrator.py.__main__` demo, exercising the "old templates-as-args" vs
     "new templates-as-closure" equivalence.
   - (b) Binned, genuinely 2D histogram data (e.g. shape `(4, 3)`) — same physical
     model as (a) but reshaped, to additionally exercise Part B on the new branch
     (no old-branch equivalent possible, since `dev` doesn't support N-D data at
     all — for this scenario, the "comparison" is: new-branch 2D result must match
     new-branch 1D-flattened result exactly, a same-branch invariant, not a
     cross-branch one; still valuable and must be included).
   - (c) Unbinned, 2 pdf_components (signal+background) — direct analog of the
     existing unbinned demo, old `S_model,B_model` vs new `pdf_components=[s,b]`.
   - (d) Unbinned, 3 pdf_components — new-branch-only capability check (no `dev`
     equivalent), validated against an independently-hand-computed reference NLL for
     a couple of parameter points (not a full `compute_fc_intervals` run necessarily
     — a direct `calc_nll_unbinned`-level check is enough and much cheaper).
5. **Comparator**: reuse the recursive dict/array-diff function developed for the
   earlier "Full regression: orchestrator demo before/after" work in this repo
   (tolerant `np.allclose` with `atol=1e-6, rtol=1e-4`, NaN-pattern-aware, walks
   nested dicts). If that snippet isn't retrievable from history, it's simple enough
   to rewrite from scratch: recursively compare dict values; for array-like leaves,
   compare shape, NaN-mask, then `np.allclose` on the non-NaN entries; report every
   mismatch with a max-abs-diff, don't stop at the first one.
6. **Keep the outputs small/fast**: small grids (e.g. 8 points/param), small
   `n_toys` (e.g. 20-30), `num_cores=1`, `sparsify_grid=False` unless testing
   sparsification specifically — this harness needs to run in well under a minute
   per branch, not reproduce a full physics analysis.
7. **Run order**: `git worktree add <path> dev-no-templates` (a second worktree next
   to the current checkout; `dev` itself can just be the current worktree, no need
   for a third copy), run scenario scripts in each, diff, then **remove the worktree**
   when done (`git worktree remove <path>`) — don't leave stray worktrees lying
   around after the PR lands.

### C.3 Acceptance bar

Zero mismatches on scenarios (a) and (c) (true cross-branch equivalence). Scenario
(b) and (d) must each pass their own same-branch invariant/reference check. Any
mismatch in (a)/(c) is a signal the refactor introduced a numeric regression, not
just a signature change — do not merge until resolved.

---

## 3. Docs / README / CHANGELOG / notebooks — everywhere `S_model`/`B_model`-shaped
## examples currently live

Every one of these needs its `compute_rates_func` example(s) and every
`compute_fc_intervals(...)` call rewritten to the new signature (§1.1-1.3), **and**
converted to fully-keyword calls per the migration hazard note in §1.1. Counts from
a fresh grep against `dev` at time of writing (re-verify, these will shift as the
work on `dev` continues in parallel):

- `README.md`: 12 matches for `S_model|B_model|S_template|B_template|S_pdf|B_pdf` —
  covers the Quick Start Guide's binned+unbinned examples (§"Quick Start Guide") and
  the "Handling Joint/Simplex-Constrained Parameters" section's worked `bounds_func`/
  `constraints` code (which currently shows the old 5-arg/2-arg
  `compute_rates_func` signatures — these must be updated too, since they're
  reproduced verbatim in the notebook and in `methodology.rst`).
- `docs/source/quickstart.rst`: 10 matches — full binned+unbinned tutorial rewrite.
- `docs/source/methodology.rst`: 2 matches — the joint/simplex-constraints worked
  examples.
- `docs/source/configuration.rst`, `index.rst`, `installation.rst`, `outputs.rst`,
  `changelog.rst` (existing entries), `pyfc.rst`, `modules.rst`, `references.rst`:
  0 matches currently — no code-example rewrites needed there, but
  `installation.rst`'s "File Tree" section and `README.md`'s "File Tree" section
  will both need a fresh sync pass at the end (new files: the `xbranch_compare/`
  harness if kept, any new test files — see §4) using the same `git ls-files`-driven
  regeneration approach used previously in this repo's history for file-tree syncs.
- `examples/pyfc_tutorial.ipynb`: 6 of 10 code cells reference the old names — full
  notebook rebuild needed (see §3.1).
- `examples/pyfc_joint_constraints_tutorial.ipynb`: 4 of 9 code cells reference the
  old names — full notebook rebuild needed (see §3.1). Note this notebook's own
  templates (`TEMPLATE_E`, `TEMPLATE_MU`, `TEMPLATE_TAU`) are **already** closures
  referenced as globals inside `@njit` functions — it is the *existing proof* that
  the new binned pattern works; its `compute_rates_func`s just need the
  `S_template, B_template` **arguments** dropped from their signatures (the
  templates themselves, as globals, don't change) plus the `compute_fc_intervals`
  calls updated to drop `S_model=TEMPLATE_E, B_model=TEMPLATE_MU` and go fully
  keyword. Its `NonlinearConstraint` bonus section is unaffected (doesn't touch
  `S_model`/`B_model`).
- `CHANGELOG.md` and `docs/source/changelog.rst`: add a new entry. **Given the
  breaking nature of this change, recommend a "## BREAKING CHANGES" subheading**
  (this repo's CHANGELOG currently only uses "### Changed"/"### Added" — a
  precedent-setting heading choice, flag it to the author rather than silently
  deciding). Cover: (1) `S_model`/`B_model` removed from `compute_fc_intervals` and
  every optimizer/toy function; (2) new `pdf_components` list mechanism replacing
  them for unbinned; (3) `compute_rates_func` signature changes for both binned
  (`(params, S_sigma2, B_sigma2)`) and unbinned (`(params, probs)`); (4) `data`/`mu`
  now support arbitrary N-dimensional shapes for binned models (with the
  event-indexing convention documented for unbinned N-D events); (5) migration
  guidance pointing at the rewritten quickstart/README examples.

### 3.1 Notebook rebuild procedure (proven pattern from prior work in this repo)

1. Edit the notebook-builder-style source (or edit the `.ipynb` cells' `source`
   arrays directly via `nbformat`) to the new API.
2. **You will hit a kernel/environment mismatch** if you try to execute via
   `jupyter nbconvert --execute` using this machine's registered Jupyter kernels —
   they point at a separate conda env (`py310`) with an old PyPI-released `PyFC`
   installed, not this checkout. Fix: `pip install -e .` (editable, `--no-deps`) the
   `dev-no-templates` worktree into whichever Python environment will run the
   execution, or register a fresh throwaway kernel pointed at an interpreter that
   already resolves `pyfc` to the local checkout, execute, then remove the throwaway
   kernel afterward (`jupyter kernelspec uninstall <name> -f`) — don't leave build
   artifacts lying around in the user's Jupyter kernel list.
3. Execute with `jupyter nbconvert --to notebook --execute --inplace
   --ExecutePreprocessor.timeout=250 --ExecutePreprocessor.kernel_name=<kernel>
   <notebook>.ipynb`, then **read the actual output cells back** (`json.load` the
   `.ipynb`, print every `stream`/`error` output) to confirm zero errors and that the
   printed numeric narrative still makes sense (e.g. the plateau-trap demo in
   `pyfc_joint_constraints_tutorial.ipynb` must still show the profiled parameter
   stuck at the bounds midpoint in the "no constraint handling" cell) — do not trust
   "nbconvert exited 0" alone as proof of correctness.

---

## 4. Test suite updates needed

Files confirmed (grep) to reference the old names and need signature rewrites:
`tests/test_bounds_func.py`, `tests/test_constraints.py`, `tests/test_optimizers.py`,
`tests/test_statistics.py`, `tests/test_restarts.py`, `tests/test_smoothing.py` (54
total matches across these 6 files at time of writing). Confirmed **unaffected**:
`tests/test_core.py`, `tests/test_io.py`, `tests/test_multiprocessing.py`,
`tests/test_numba_compilation.py` — re-grep to confirm this hasn't changed, but no
edits expected there.

For each affected test file: every dummy `compute_rates_func`/`rate_func` defined
inline needs its signature updated (drop `S_template, B_template` for binned /
merge `s_probs, b_probs` into `probs` for unbinned); every direct call to
`calc_nll`, `calc_nll_unbinned`, or any of the six `optimizers.py` functions needs
its positional args updated to match the new signatures in §2.1-2.3.

**New test files to add** (in addition to fixing the existing 6):
- `tests/test_nd_binned.py`: the flatten-invariance check from Part B (2D-shaped
  `mu`/`N_obs` must give byte-identical `calc_nll` output vs the same data
  flattened to 1D), plus a smoke test that a genuine 2D histogram flows correctly
  through a full `compute_fc_intervals` call (small grid/n_toys, per C.2.6's
  performance guidance).
- `tests/test_pdf_components.py`: unbinned 2-component and 3+-component correctness
  (compare against a hand-computed reference NLL for 3+ components, since there's no
  old-API equivalent to diff against).
- Consider whether the cross-branch harness scripts (§C) belong under `tests/` proper
  (pytest-discoverable) or in a separate `xbranch_compare/` directory run manually/
  in CI as a distinct job — **OPEN DECISION #3**, recommend the latter (they need
  git worktrees and multi-environment setup that doesn't fit pytest's normal
  single-environment model cleanly), but confirm with the author.

---

## 5. Suggested milestone breakdown (don't do this as one giant commit)

Mirroring the incremental-commit discipline already used in this repo's git history
(each fix as its own reviewable commit with its own test pass):

1. `pyfc/binned.py` + `pyfc/unbinned.py`: core signature changes + N-D flattening,
   with the new `tests/test_nd_binned.py`/`tests/test_pdf_components.py` passing
   against these two files in isolation (some downstream breakage in
   optimizers.py/toys.py/orchestrator.py is expected/fine at this checkpoint since
   they haven't been updated yet — just don't run the full suite as a gate here).
2. `pyfc/optimizers.py`: all six functions + docstrings.
3. `pyfc/toys.py`: worker + `generate_and_fit_toys_python`.
4. `pyfc/orchestrator.py`: `compute_fc_intervals` signature, validation, all call
   sites, `__main__` demo. **Full test suite must pass at the end of this
   checkpoint** (after fixing the 6 affected test files in the same commit or the
   next one — whichever grouping reads more cleanly as a diff).
5. Cross-branch comparison harness (§C) — run it, confirm zero mismatches, **before**
   proceeding to docs (if it fails, something in steps 1-4 is wrong; fix before
   spending time on docs that would need re-touching anyway).
6. README.md + all rst docs + CHANGELOG.md/changelog.rst.
7. Both notebooks rebuilt/re-executed.
8. Final full-repo grep for `S_model\|B_model\|S_template\|B_template\|S_pdf\|B_pdf`
   outside of historical CHANGELOG entries/git log — should return **zero** hits in
   live code/docs by the end.

---

## 6. Explicit open decisions (confirm with the author early in the new session)

1. **§1.2**: leave `S_sigma2`/`B_sigma2` as explicit args (not folded into closures
   too) — recommended default, but the naming asymmetry (`S_`/`B_` prefixes with no
   more `S_model`/`B_model` to justify them) should be flagged, not silently ignored.
2. **§2b**: is `pdf_components` the final name, or does the author want something
   else (`pdfs`, `density_functions`, ...)?
3. **§4**: cross-branch harness location — `tests/xbranch_compare/` (pytest-adjacent)
   vs. a separate top-level `xbranch_compare/` directory outside the normal test
   run. Also: does the author want this harness **kept** in the repo long-term as a
   reusable regression tool for future signature changes, or was it meant as a
   one-off validation only (delete after the PR merges)?
4. **§3**: the "BREAKING CHANGES" CHANGELOG heading — precedent-setting wording
   choice for this project's changelog conventions; confirm before adding.
5. **Version bump**: `pyproject.toml` currently pins `version = "0.9.2"`. A breaking
   public-API change like this conventionally warrants at least a minor bump (if
   pre-1.0, breaking changes are commonly signaled via a minor version bump per
   SemVer's pre-1.0 carve-out) or a major bump to `1.0.0`. Not decided here — ask.

---

## 7. Final acceptance checklist

- [ ] `grep -rn "S_model\|B_model\|S_template\|B_template\|S_pdf\|B_pdf" pyfc/ tests/ README.md docs/source/*.rst examples/*.ipynb`
  returns zero hits (except inside CHANGELOG history entries describing the *old*
  behavior, which should stay worded in the past tense).
- [ ] All existing tests pass on `dev-no-templates` (`pytest tests/ -v`).
- [ ] New `test_nd_binned.py` and `test_pdf_components.py` pass.
- [ ] Cross-branch harness (§C) shows zero mismatches on scenarios (a)/(c); (b)/(d)
  same-branch invariants hold.
- [ ] Both notebooks re-executed with zero cell errors, outputs re-embedded.
- [ ] README.md + all affected rst docs updated; File Tree sections re-synced if new
  files were added.
- [ ] CHANGELOG.md + docs/source/changelog.rst updated with a clear breaking-change
  entry.
- [ ] All 5 open decisions in §6 resolved and reflected in the final diff.
- [ ] No stray git worktrees left behind (`git worktree list` clean).
