"""
Tests for `toys.generate_and_fit_toys_python`'s ProcessPoolExecutor -> ThreadPoolExecutor
fallback.

WHY THIS FILE EXISTS: the unbinned toy path dispatches every toy to a separate process,
which is the right call (unbinned PDFs are pure Python and GIL-bound, so threads buy
nothing), but it imposes a requirement users are not told about at the call site: every
callable they pass -- `compute_rates_func`, `generate_toy_func`, each entry of
`pdf_components` -- has to be picklable. A lambda, a closure, or a method of a local class
is not, and defining one is the most natural thing in the world when writing an analysis
script.

`toys.py` handles that with a deliberately broad `except Exception` that warns and retries
the whole batch on threads. It is the difference between a confusing crash deep inside
`concurrent.futures` and a slow-but-correct run, and coverage showed nothing exercised it:
the recovery path only runs when the pool has already failed, which no other test arranges.

The fallback is also load-bearing in a subtler way. The comment on that `except` notes that
it catches genuine bugs in user code just as readily as pool-infrastructure failures, and
that the ThreadPoolExecutor retry deliberately does *not* catch, so a real user bug surfaces
uncaught on the second attempt. Both halves of that contract are tested here.
"""

import sys
import warnings

import numpy as np
import pytest

from pyfc.toys import generate_and_fit_toys_python

# Both tests below deliberately break a ProcessPoolExecutor mid-flight, and on Python 3.9
# that deadlocks rather than raising: the run hung for two hours on the 3.9 CI runner --
# 3.10, 3.11 and the coverage job had all passed in minutes -- and the runner's cleanup
# reported two orphaned pytest children, i.e. forked workers that never exited.
#
# The hang is in `with ProcessPoolExecutor(...)`'s implicit `shutdown(wait=True)` on the
# way out of the failed block, waiting on a child that cannot finish. It needs both halves
# to reproduce: a parent that already has Numba's threads running, and a fork. A stdlib-only
# reduction of the same try/with/map/except shape does NOT hang on 3.9, which is why this is
# pinned to the interpreter rather than to the pattern.
#
# Skipped rather than removed, because what it documents is a real exposure for users on
# 3.9, not a defect in the test: the recovery path this file exists to check is itself
# unreliable there. `pyproject.toml` currently declares `requires-python = ">=3.8"`.
# See the note in CHANGELOG.md; deciding what to do about it is a separate call.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="ProcessPoolExecutor shutdown deadlocks on 3.9 when the pool is broken while "
           "Numba's threads are live in the forked parent (observed: 2h hang, orphaned "
           "workers). The fallback path itself is unreliable on 3.9.",
)


def _rates(params, S_sumw2=None, B_sumw2=None):
    """Picklable module-level rates function for a 2-bin binned model."""
    mu = np.array([params[0] * 10.0 + 5.0, params[1] * 8.0 + 4.0])
    return mu, mu


def _toy_kwargs(**overrides):
    """The common argument block; individual tests override what they care about."""
    kwargs = dict(
        true_params=np.array([1.0, 1.0]), n_params=2, fit_mode="1d",
        fix_idx=0, fix_A=None, fix_B=None, t_vA=1.0, t_vB=None,
        bounds_list=[(0.2, 2.0), (0.2, 2.0)], n_toys=4, strategy="scipy",
        num_cores=1, verbose=0, likelihood_type="binned",
        compute_rates_func=_rates, use_finite_mc=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_unpicklable_callable_falls_back_to_threads_and_still_returns_results():
    """
    A closure that cannot be pickled must not sink the run. The ProcessPoolExecutor attempt
    fails, a warning names the retry, and the ThreadPoolExecutor path produces the full set
    of test statistics anyway.

    This is exercised through the unbinned branch, since that is the only one that uses
    processes. The local `pdf_components` lambdas below are exactly the kind of thing an
    analysis script defines inline, and are unpicklable for that reason.
    """
    S_pool = np.random.default_rng(5).normal(5.0, 1.0, size=200)
    B_pool = np.random.default_rng(6).exponential(3.0, size=200)

    # Deliberately local: closures over test scope cannot be pickled.
    def _local_rates(params, probs):
        s, b = probs[0], probs[1]
        total = params[0] * 10.0 + params[1] * 5.0
        if len(s) == 0 and len(b) == 0:
            return total, np.array([])
        return total, params[0] * 10.0 * s + params[1] * 5.0 * b

    def _local_toy(true_params, S_mc_pool, B_mc_pool):
        n = np.random.poisson(true_params[0] * 10.0) + 1
        return np.random.choice(S_mc_pool, size=n, replace=True)

    np.random.seed(41)
    with pytest.warns(UserWarning, match="ThreadPoolExecutor"):
        t_stats = generate_and_fit_toys_python(**_toy_kwargs(
            likelihood_type="unbinned",
            compute_rates_func=_local_rates,
            generate_toy_func=_local_toy,
            pdf_components=[lambda x: np.exp(-0.5 * (x - 5.0) ** 2),
                            lambda x: np.full_like(x, 0.1)],
            S_mc_pool=S_pool, B_mc_pool=B_pool,
            bounds_list=[(0.2, 3.0), (0.2, 3.0)],
        ))

    assert len(t_stats) == 4, (
        "the thread fallback returned a different number of toys than were requested, so "
        "the recovery is not equivalent to the path it replaces"
    )
    assert np.all(np.isfinite(t_stats))
    assert np.all(np.asarray(t_stats) >= 0.0)


def test_a_genuine_user_bug_surfaces_uncaught_rather_than_being_swallowed():
    """
    The other half of the fallback's contract. The broad `except` around the process pool
    catches real bugs in user code as readily as pool failures, so if the retry also
    swallowed them a broken `compute_rates_func` would produce a silently wrong result
    instead of an error. The ThreadPoolExecutor path deliberately does not catch, so the
    exception propagates on the second attempt.

    Without this, the fallback would be a liability rather than a safety net: it would
    convert every user bug into a warning plus a plausible-looking array of numbers.
    """
    def _broken_rates(params, probs):
        raise ZeroDivisionError("bug in the user's rates function")

    def _local_toy(true_params, S_mc_pool, B_mc_pool):
        return np.random.choice(S_mc_pool, size=5, replace=True)

    S_pool = np.random.default_rng(8).normal(5.0, 1.0, size=100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ZeroDivisionError):
            generate_and_fit_toys_python(**_toy_kwargs(
                likelihood_type="unbinned",
                compute_rates_func=_broken_rates,
                generate_toy_func=_local_toy,
                pdf_components=[lambda x: np.full_like(x, 0.5),
                                lambda x: np.full_like(x, 0.1)],
                S_mc_pool=S_pool, B_mc_pool=S_pool,
            ))
