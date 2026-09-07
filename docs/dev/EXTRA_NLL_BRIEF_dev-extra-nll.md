# Brief: a hook for Gaussian nuisance constraints in the binned likelihood

Written 2026-09-07 from the `GeoneutrinoDecay` session, which is the requesting
consumer. Nothing here has been changed in this repository; the whole change is
yours to make. Naming follows the established convention — rename this file to
match whichever branch you work on (`EXTRA_NLL_BRIEF_<branch>.md`) and archive
it into `docs/dev/` at the end, as was done for `REFACTOR_BRIEF`,
`FOLLOWUP_BRIEF` and `FIXES_BRIEF`.

---

## 1. What is being asked

PyFC's binned likelihood cannot express a Gaussian constraint on a nuisance
parameter. Add a way to supply one.

Concretely: a caller wants to say *"`norm_reactor` is known to 4.6% from an
external measurement"*, and have that term enter the NLL that every conditional
and unconditional fit minimises, so that it also enters the toy fits and hence
the critical value of the test statistic.

This is not the same as `bounds_func` or `constraints`. Those express **hard**
geometric relations on the parameter box — a simplex `a + b <= 1`, or bounds on
a free parameter that depend on a fixed one. What is missing is a **soft**
term: a parameter with a preferred value and a width, contributing
`0.5 ((x - c)/s)^2` to the NLL, which is how every external measurement of a
systematic enters a physics likelihood.

The README's warning against "hand-rolled smooth penalty functions" is about
something different again — penalising *unphysical rate regions* by multiplying
the rate — and does not apply. This is a term added to the NLL, not a
distortion of the model.

## 2. Why it matters, with a number

The requesting analysis constrains neutrino decay using geoneutrinos. It has 13
parameters, of which **11 carry Gaussian constraints**:

| parameter | width | source of the constraint |
|---|---|---|
| `log10_tau1_over_m1`, `log10_tau2_over_m2` | free | the parameters of interest |
| `norm_crust_local_U238`, `_Th232` | 11% | published site-specific crustal model |
| `norm_crust_roc_U238`, `_Th232` | 12% | published crustal model |
| `norm_mantle_U238`, `_Th232` | 35% | bulk-silicate-Earth model spread |
| `norm_reactor` | 4.6% | Borexino's own 87 ± 4 TNU prediction |
| `norm_alpha_n`, `norm_accidental`, `norm_li9` | 16%, 0.5%, 28% | measured backgrounds |
| `delta_e_scale` | 1 sigma | detector energy scale |

plus a **one-sided** constraint on a *derived* quantity (the sum of the three
neutrino masses, bounded from above by cosmology), which is a function of a
parameter rather than a parameter itself.

Measured in that analysis, on the same dataset:

```
with the geochemical constraints    limit = 4.47e-09 s/eV
with them removed                   NO LIMIT IS SET AT ALL
```

Without the constraints the normalisations simply absorb the signal. So handing
this model to PyFC today would not fail — it would converge and return a
well-formed FC interval for a model that is unconstrained and sets no limit.
A confident wrong answer, which is the failure mode worth engineering against.

## 3. The proposed interface

A single optional callable, threaded from `compute_fc_intervals` down to
wherever the NLL is assembled:

```python
extra_nll : callable, optional
    ``extra_nll(params) -> float``, added to the NLL at every evaluation.
    Receives the FULL parameter vector (length ``len(grids)``), in the same
    order as ``grids``, with any scan-fixed values already substituted.
    Must return a finite non-negative float.  Called once per NLL evaluation,
    so it should be cheap.
```

Rationale for this shape rather than alternatives:

- **A callable, not an array of (centre, width) triples.** Triples cover the
  common case but not the one-sided cosmological bound, nor a constraint on a
  quantity *derived* from several parameters. A callable covers all three and
  costs nothing extra to implement. If you want the ergonomic case too, a small
  helper that *builds* such a callable from a list of `(index, centre, width)`
  is a natural addition, but the primitive should be the callable.
- **Full parameter vector, not the free subspace.** The free subspace changes
  between the unconditional fit, the 1D scan and the 2D scan; a caller writing
  against the full vector writes one function that works in all three. This
  matches the convention already chosen for `constraints`, which the docstring
  says are "expressed in the **full** parameter-vector space ... PyFC projects
  them down internally".
- **Added, not multiplied.** It is a log-likelihood term.

## 4. The hard part: `calc_nll` is JIT-compiled

`pyfc/binned.py:41` —

```python
@njit(fastmath=True, nogil=True)
def calc_nll(params, N_obs, S_sumw2, B_sumw2, use_finite_mc, compute_rates_func):
```

A `@njit` function cannot accept an arbitrary Python callable. **Do not put the
hook inside `calc_nll`.** Add it at the call sites, which fall into two groups
with different constraints:

**Group A — plain Python closures. These are easy.**

| file | line | enclosing |
|---|---|---|
| `pyfc/optimizers.py` | 354 | `def cost(params)` inside `unconditional_fit_scipy` |
| | 472 | `def cost(free_p)` inside `conditional_fit_1d_scipy` |
| | 596 | `def cost(free_p)` inside `conditional_fit_2d_scipy` |
| | 740 | inside `unconditional_fit_ultranest` (note the leading `-`) |
| | 831 | inside an ultranest path returning `(nll, p)` |
| | 856 | inside a conditional ultranest path (leading `-`) |

Each is an ordinary Python function handed to scipy or ultranest. Adding
`+ extra_nll(p_full)` there is straightforward. Watch the three sites that
return a **negated** NLL for ultranest — the sign of the added term must follow.
Watch also that the conditional sites receive `free_p`, the free subspace, and
must reassemble the full vector before calling the hook; the code that
substitutes the fixed value(s) is already right there.

**Group B — inside `@njit` functions. These need a decision.**

| file | line | enclosing (all `@njit(fastmath=True, nogil=True)`) |
|---|---|---|
| `pyfc/binned.py` | 317 | `unconditional_fit_grid` |
| | 374 | `conditional_fit_grid_1d` |
| | 429 | `conditional_fit_grid_2d` |

This is the `strategy="grid"` path. Three options, in my order of preference:

1. **Require a numba-jitted callable when `strategy="grid"`.** Numba can call a
   `CPUDispatcher` passed as an argument, which is already how
   `compute_rates_func` reaches these same functions — so the machinery exists
   and the pattern is established in this file. The cost is that a grid-strategy
   user must write `@njit` penalties. For the requesting analysis they are
   quadratics and would jit without difficulty.
2. **Raise a clear `NotImplementedError`** when `extra_nll` is given with
   `strategy="grid"`, naming the two strategies that do support it. Cheap,
   honest, and leaves the door open. Unacceptable only if grid is the strategy
   people actually use for constrained fits.
3. **Evaluate the penalty outside the jitted loop.** Only correct if the grid
   fitters return the argmin rather than only the minimum, and only if the
   penalty cannot change *which* grid point wins — which it can. Mentioned only
   to be dismissed.

Whichever is chosen, say so in the `extra_nll` docstring, because a user whose
penalty is silently ignored on one strategy and applied on another will get two
different answers and no warning.

## 5. Threading

`extra_nll` has to reach the optimizers from `compute_fc_intervals`. It is
passed alongside `compute_rates_func` everywhere the latter already goes, so
the mechanical change is to follow that parameter. There are also the toy-fitting
paths in `toys.py` — **the penalty must apply to toy fits too**, or the critical
value of the test statistic is computed under a different likelihood than the
data statistic, and the coverage the whole method exists to guarantee is lost.
That is the single most important correctness point in this brief.

## 6. Tests

Per the established practice in this repo, **mutate the code and confirm each
new test fails** before keeping it; automate it as a script that applies the
mutation, runs the one test, asserts non-zero exit, and restores the file
byte-for-byte. Three tests that looked correct and detected nothing are on
record here.

Specific mutations worth checking, each of which a good test suite should catch:

- Drop `extra_nll` from the **toy** path but keep it in the data path. The
  interval will still be produced and will look plausible; only a coverage test
  or a direct comparison of thresholds will notice. This is the defect most
  likely to survive.
- Drop it from the conditional fit but keep it in the unconditional one, and
  vice versa. Each biases the test statistic in a different direction.
- Get the sign wrong on the ultranest sites.
- Reassemble the full vector in the wrong order in a conditional fit — this
  will pass a one-parameter test and fail only with two or more scanned
  parameters and an asymmetric penalty.

A useful analytic oracle: with **one** parameter, no data (or a single bin with
`n_obs` equal to `mu`), and a penalty `0.5((x-c)/s)^2`, the profile likelihood
interval must reproduce `c +- s * z` exactly, and the FC interval must agree
with it in the asymptotic limit where Wilks holds. That gives a closed-form
target rather than a regression value.

Coverage: use `bash scripts/run_coverage.sh`, not a bare `pytest --cov`. The
canary is `binned.py` reporting ~90% rather than ~10%; ~10% means the
JIT-disabled half of the split run did not execute. Also worth running the
suite under `~/anaconda3/envs/py39repro/bin/python`, the oldest interpreter in
the CI matrix, since local dev is 3.12 and that gap is where version-specific
failures have hidden before.

## 7. Documentation and release

- The README has a section "Handling joint/simplex-constrained parameters"
  explaining `bounds_func` and `constraints`. `extra_nll` belongs beside it,
  with the distinction stated explicitly: those are hard constraints on the
  parameter space, this is a soft term in the likelihood. A reader who does not
  see the three contrasted will pick the wrong one.
- CHANGELOG entry. Note the existing hazard: headings there have been deleted
  by anchoring a new entry on the previous version heading — verify all headings
  after editing.
- Version: bump the patch digit only unless you decide otherwise. Note that
  0.20.0 is release-ready but deliberately unpublished, so this can land without
  forcing a release.

## 8. Acceptance, from the requesting side

Once this exists, the geoneutrino analysis will call
`compute_fc_intervals(..., extra_nll=...)` with a callable that adds, over 11
parameters, `0.5((x_i - 1)/s_i)^2`, plus a one-sided term on a derived mass sum
that is flat below its bulk and quadratic above. The check that it worked, on
our side, is that the FC interval reduces to the profile-likelihood interval we
already have (4.47e-09 s/eV for one Earth model) in the regime where Wilks is
valid, and departs from it near the physical boundary, which is the whole reason
for wanting FC. If the FC interval instead comes out *unbounded*, the penalty is
not reaching the fits.

## 9. What this is for

Geoneutrinos are electron antineutrinos from uranium and thorium decay inside
the Earth. Their baselines reach an Earth diameter at a few MeV, giving the
largest L/E of any terrestrial antineutrino source, which makes them a probe of
neutrino decay. The limit is on a lifetime bounded below by zero and sits right
at the reach of the baseline — precisely where Wilks' theorem is least
trustworthy and where FC earns its keep. Hence this request.
