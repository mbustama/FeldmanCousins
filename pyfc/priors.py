"""
Prior Construction Module

Builders for the `extra_nll` callable that `compute_fc_intervals` accepts -- soft
constraints on nuisance parameters, as opposed to the hard geometry expressed by
`bounds_func` and `constraints`.

Nothing here adds capability. `extra_nll(params) -> float` receives the FULL
parameter vector and returns a scalar, so it has always been able to express a
correlated Gaussian, a one-sided bound, or any nonlinear shape you can write down.
What this module adds is that the three ways of getting such a prior wrong are
hard to reach from here, because each of them is silent:

  1. UNITS.  PyFC's NLL is `-2 ln L`, so a Gaussian block contributes
     `d^T Sigma^-1 d` with NO factor of 0.5.  The `-ln L` convention makes every
     constraint weaker by sqrt(2) with a converged fit and a plausible interval.

  2. STALENESS.  Numba freezes a global or closed-over array BY VALUE when it
     compiles.  Rebinding the name is ignored, and so is mutating the array in
     place -- measured, all three of `M[0,0] = x`, `M *= x` and `M[:] = new`.
     Build the covariance, hand it to a builder here, and never touch it again:
     each call returns a fresh dispatcher carrying its own frozen copy.

  3. FORMULATION.  `d @ inv @ d` costs ~154 ns per call almost regardless of size
     -- that is BLAS dispatch overhead, not arithmetic -- while an explicit
     symmetric expression over a 2x2 block costs ~1.2 ns net.  For a PAIRWISE
     prior that is a factor of ~68, paid at every NLL evaluation of every fit of
     every toy.  The builders here pick the formulation by size; the crossover
     was measured at n ~= 35 and is exposed as `MATMUL_THRESHOLD`.

A note on marginal vs. conditional widths:
------------------------------------------
Profiling a Gaussian returns the MARGINAL, not the conditional.  With covariance
`Sigma`, the profile-likelihood test statistic for one parameter of a block is
`t(x_i) = (x_i - c_i)^2 / Sigma_ii`, so the 1-sigma point is at `c_i + sqrt(Sigma_ii)`
-- the square root of a COVARIANCE element, never `1/sqrt((Sigma^-1)_ii)`.  Those
two differ by a factor of 2.3 at a correlation of 0.9.  If your source document
quotes per-parameter uncertainties and a correlation matrix, they are marginal
sigmas and `gaussian_block_from_correlation` is the builder that wants them.

Created: v0.20.0
Author: Mauricio Bustamante (mbustamante@gmail.com)

This file was released as part of the PyFC code, stored at
https://github.com/mbustama/FeldmanCousins, which exists under a GNU GPL v3 License.
"""

import numpy as np
from numba import njit


# --- 0. Formulation threshold ---

# Below this many constrained parameters the builders emit an explicit symmetric
# double loop over the frozen inverse; at or above it they emit `d @ inv @ d`.
#
# Measured on this machine (ns per call, numba 0.60, fastmath, driver baseline
# subtracted), symmetric loop vs. matmul:
#
#       n=2     0.5 vs 146      n=20    113 vs 199
#       n=4     2.7 vs 153      n=30    234 vs 254
#       n=8     9.5 vs 144      n=40    366 vs 279   <- matmul wins
#       n=13     59 vs 181      n=60    720 vs 430
#
# The loop is O(n^2) in real arithmetic while matmul is O(n^2) in BLAS with a
# large constant, so the crossover is where that constant is finally amortised.
# It is a tunable rather than a law: pass `threshold=` to a builder to override,
# which is also how the test suite checks the two branches agree.
MATMUL_THRESHOLD = 35


# --- 1. Validation ---

def _validate_block(indices, centres, cov):
    """
    Checks a Gaussian block up front, in Python, where the message can be clear.

    Every one of these failures is otherwise silent or near-silent.  A covariance
    that is not positive definite is the worst of them: `d^T Sigma^-1 d` is then
    unbounded below along some direction, so the "prior" actively pushes the fit
    away instead of constraining it, and the run completes with a plausible-looking
    interval built on a likelihood that has no minimum.
    """
    idx = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).ravel())
    cen = np.ascontiguousarray(np.asarray(centres, dtype=np.float64).ravel())
    cv = np.ascontiguousarray(np.asarray(cov, dtype=np.float64))

    n = idx.size
    if n == 0:
        raise ValueError("indices is empty: a prior over no parameters is not a prior.")
    if np.any(idx < 0):
        raise ValueError(
            f"indices must be non-negative positions in the full parameter vector; got {idx.tolist()}. "
            "Negative (Python-style) indexing is rejected because it silently means something "
            "different once the number of parameters changes."
        )
    if np.unique(idx).size != n:
        raise ValueError(
            f"indices contains duplicates: {idx.tolist()}. A repeated parameter would be "
            "constrained twice by one block, which is not what a covariance matrix means."
        )
    if cen.size != n:
        raise ValueError(f"centres has length {cen.size} but indices has length {n}; they must match.")
    if cv.ndim != 2 or cv.shape[0] != cv.shape[1]:
        raise ValueError(f"cov must be a square 2-D matrix; got shape {cv.shape}.")
    if cv.shape[0] != n:
        raise ValueError(f"cov is {cv.shape[0]}x{cv.shape[0]} but {n} parameters were given.")
    if not np.allclose(cv, cv.T, rtol=1e-10, atol=1e-12):
        worst = np.max(np.abs(cv - cv.T))
        raise ValueError(
            f"cov is not symmetric (max |cov - cov.T| = {worst:.3e}). A covariance matrix is "
            "symmetric by construction, so an asymmetric one means the matrix was built wrong "
            "-- most often a correlation matrix filled in on one triangle only."
        )
    try:
        np.linalg.cholesky(cv)
    except np.linalg.LinAlgError:
        eigs = np.linalg.eigvalsh(cv)
        raise ValueError(
            f"cov is not positive definite (smallest eigenvalue {eigs.min():.3e}). "
            "The resulting penalty would be unbounded below along at least one direction, so "
            "the fit would be pushed away from the constraint rather than towards it, and "
            "nothing downstream would report a problem. Check for a correlation of exactly "
            "+-1, a duplicated row, or more parameters than the measurement actually constrains."
        ) from None

    inv = np.linalg.inv(cv)
    # Symmetrise: inv() of a symmetric matrix is symmetric only up to rounding, and the
    # explicit-loop branch reads the upper triangle only. Without this the two branches
    # would disagree in the last bits for no reason anyone could find later.
    inv = np.ascontiguousarray(0.5 * (inv + inv.T))
    return idx, cen, inv


# --- 2. Gaussian blocks ---

def gaussian_block(indices, centres, cov, threshold=None):
    """
    Builds a correlated multivariate Gaussian prior over a subset of parameters.

    This is the general case: `indices` may name one parameter, two (the pairwise
    case), or all of them, and `cov` is the full covariance over that subset, so
    off-diagonal terms are ordinary rather than special.

    Parameters
    ----------
    indices : sequence of int
        Positions in the FULL parameter vector, in `grids` order, that this block
        constrains. Must be unique and non-negative.
    centres : sequence of float
        The measured central value of each, in the same order as `indices`.
    cov : (n, n) array_like
        Covariance -- NOT its inverse, and not a correlation matrix. Must be
        symmetric and positive definite. Use `gaussian_block_from_correlation` if
        your source quotes sigmas and correlations separately.
    threshold : int, optional
        Override `MATMUL_THRESHOLD` for this block. Only affects speed, never the
        value returned; the test suite uses it to check the two branches agree.

    Returns
    -------
    callable
        A numba-jitted `prior(params) -> float` giving `d^T cov^-1 d` in `-2 ln L`
        units, suitable for `compute_fc_intervals(..., extra_nll=prior)` under any
        strategy including `"grid"`.

    Notes
    -----
    The covariance is inverted and frozen HERE. Mutating `cov` afterwards has no
    effect on the returned prior -- by design, since numba would ignore the change
    anyway and doing it at build time makes that visible rather than surprising.
    Call the builder again to change the constraint.
    """
    idx, cen, inv = _validate_block(indices, centres, cov)
    n = int(idx.size)
    limit = MATMUL_THRESHOLD if threshold is None else int(threshold)

    if n < limit:
        # Explicit symmetric form: diagonal terms once, off-diagonal terms twice.
        # Reads the upper triangle only, which is why `inv` is symmetrised above.
        @njit(fastmath=True, nogil=True)
        def prior(params):
            total = 0.0
            for i in range(n):
                d_i = params[idx[i]] - cen[i]
                total += d_i * d_i * inv[i, i]
                for j in range(i + 1, n):
                    total += 2.0 * d_i * (params[idx[j]] - cen[j]) * inv[i, j]
            return total
    else:
        @njit(fastmath=True, nogil=True)
        def prior(params):
            d = params[idx] - cen
            return d @ inv @ d

    return prior


def gaussian_block_from_correlation(indices, centres, sigmas, correlation, threshold=None):
    """
    Same as `gaussian_block`, but from the form external results are usually quoted in.

    Papers and calibration notes almost always give a central value, a per-parameter
    uncertainty, and a correlation matrix -- not a covariance. Converting by hand is
    an easy place to drop the outer product, which produces a matrix that is still
    symmetric and still positive definite, so nothing complains.

    `sigmas` are MARGINAL uncertainties: sigma_i is what you would quote for
    parameter i having profiled the others out, and it is what comes back out of a
    PyFC scan. See the module docstring on marginal vs. conditional.
    """
    sig = np.ascontiguousarray(np.asarray(sigmas, dtype=np.float64).ravel())
    corr = np.ascontiguousarray(np.asarray(correlation, dtype=np.float64))

    if np.any(sig <= 0.0):
        raise ValueError(f"sigmas must all be positive; got {sig.tolist()}.")
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] != sig.size:
        raise ValueError(
            f"correlation must be {sig.size}x{sig.size} to match {sig.size} sigmas; got shape {corr.shape}."
        )
    if not np.allclose(np.diag(corr), 1.0, rtol=0, atol=1e-10):
        raise ValueError(
            f"correlation must have unit diagonal; got {np.diag(corr).tolist()}. A diagonal of "
            "sigma^2 means this is already a covariance matrix -- pass it to gaussian_block instead."
        )
    if np.any(np.abs(corr) > 1.0 + 1e-12):
        raise ValueError("correlation has an entry outside [-1, 1]; it is not a correlation matrix.")

    return gaussian_block(indices, centres, np.outer(sig, sig) * corr, threshold=threshold)


# --- 3. Composition ---

def _add(first, second):
    """Sums two priors, staying jitted when both inputs are."""
    from numba.core.registry import CPUDispatcher
    import numba

    # Under NUMBA_DISABLE_JIT=1 the @njit decorator returns a plain function, so an
    # isinstance check would say "not jitted" for everything and silently drop the
    # whole chain to the Python branch. Ask the config instead, exactly as the
    # strategy="grid" guard in orchestrator.py does.
    if numba.config.DISABLE_JIT:
        both_jitted = True
    else:
        both_jitted = isinstance(first, CPUDispatcher) and isinstance(second, CPUDispatcher)

    if both_jitted:
        @njit(fastmath=True, nogil=True)
        def combined(params):
            return first(params) + second(params)
    else:
        def combined(params):
            return first(params) + second(params)

    return combined


def combine_priors(*priors):
    """
    Sums any number of priors into one `extra_nll`.

    Independent constraints add in log-likelihood space, so this is the operation
    that lets a correlated block, a second correlated block, and a hand-written
    one-sided bound coexist without anyone writing a function that sums them by
    hand and gets an index wrong.

    Composition is by nesting, and it is close to free: measured 3.03 ns for a
    single 2x2 block and 4.85 ns for three of them nested, because numba inlines
    through the chain.

    The result stays jitted -- and therefore usable with `strategy="grid"` -- only
    if every input is. Mixing one plain Python callable in returns a plain Python
    callable, which every other strategy accepts.
    """
    if not priors:
        raise ValueError("combine_priors() needs at least one prior.")

    total = priors[0]
    for nxt in priors[1:]:
        total = _add(total, nxt)
    return total
