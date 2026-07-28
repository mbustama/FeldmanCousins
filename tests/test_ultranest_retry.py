"""
Tests for _run_ultranest_with_retry, the workaround for a still-open UltraNest
bug (confirmed present through at least v4.5.0 and the current GitHub master)
where ReactiveNestedSampler._find_strategy's internal random subsample can
occasionally produce a degenerate 1D `logweights` array and crash with
`IndexError: too many indices for array: array is 1-dimensional, but 2 were
indexed`. These tests use a fake sampler class (not real UltraNest sampling)
to deterministically exercise the retry logic itself.
"""
import numpy as np
import pytest

from pyfc.optimizers import ULTRANEST_AVAILABLE, _run_ultranest_with_retry

pytestmark = pytest.mark.skipif(not ULTRANEST_AVAILABLE, reason="UltraNest is required for these tests")


class _FailNTimesSampler:
    """Fake sampler raising the known bug's IndexError on its first `n_failures` runs."""

    call_count = 0

    def __init__(self, param_names, log_likelihood, prior_transform, log_dir=None):
        pass

    def run(self, **kwargs):
        _FailNTimesSampler.call_count += 1
        if _FailNTimesSampler.call_count <= _FailNTimesSampler.n_failures:
            raise IndexError("too many indices for array: array is 1-dimensional, but 2 were indexed")
        return {"maximum_likelihood": {"logl": -1.0, "point": np.array([0.5])}}


class _AlwaysUnrelatedErrorSampler:
    def __init__(self, param_names, log_likelihood, prior_transform, log_dir=None):
        pass

    def run(self, **kwargs):
        raise IndexError("some unrelated indexing bug, not the logweights one")


def test_retry_recovers_after_one_failure(monkeypatch):
    """A single known-bug failure must be retried with a fresh sampler and succeed."""
    import pyfc.optimizers as optimizers_module

    _FailNTimesSampler.call_count = 0
    _FailNTimesSampler.n_failures = 1
    monkeypatch.setattr(optimizers_module.ultranest, "ReactiveNestedSampler", _FailNTimesSampler)

    with pytest.warns(UserWarning, match="known internal bug"):
        result = _run_ultranest_with_retry(["p1"], lambda p: -1.0, lambda c: c, {})

    assert _FailNTimesSampler.call_count == 2
    assert result["maximum_likelihood"]["logl"] == -1.0


def test_retry_gives_up_after_max_attempts(monkeypatch):
    """If every attempt hits the known bug, the last exception must propagate."""
    import pyfc.optimizers as optimizers_module

    _FailNTimesSampler.call_count = 0
    _FailNTimesSampler.n_failures = 10  # always fails, more than max_attempts
    monkeypatch.setattr(optimizers_module.ultranest, "ReactiveNestedSampler", _FailNTimesSampler)

    with pytest.warns(UserWarning, match="known internal bug"):
        with pytest.raises(IndexError, match="too many indices for array"):
            _run_ultranest_with_retry(["p1"], lambda p: -1.0, lambda c: c, {}, max_attempts=3)

    assert _FailNTimesSampler.call_count == 3


def test_unrelated_index_error_is_not_retried(monkeypatch):
    """An IndexError with a different message must propagate immediately, not be swallowed as a retry."""
    import pyfc.optimizers as optimizers_module

    monkeypatch.setattr(optimizers_module.ultranest, "ReactiveNestedSampler", _AlwaysUnrelatedErrorSampler)

    with pytest.raises(IndexError, match="unrelated indexing bug"):
        _run_ultranest_with_retry(["p1"], lambda p: -1.0, lambda c: c, {})
