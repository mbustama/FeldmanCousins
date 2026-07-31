#!/usr/bin/env bash
#
# Measure test coverage of `pyfc`, working around numba.
#
# WHY THIS SCRIPT EXISTS, rather than a plain `pytest tests/ --cov`:
#
# PyFC's mathematical core (`pyfc/binned.py`) is six `@njit(fastmath=True, nogil=True)`
# functions, `calc_nll` among them.  coverage.py traces Python bytecode, and numba does not
# execute any: it compiles those functions to machine code the first time they are called.
# So every line inside them is reported as missed no matter how hard the suite hits them.
# Measured on one machine, over `test_smoothing.py`, `test_finite_mc_likelihood.py` and
# `test_nd_binned.py`:
#
#     default (JIT on)        binned.py  10%      27.2 s
#     NUMBA_DISABLE_JIT=1     binned.py  90%       6.3 s
#
# 10% is not a real number -- `calc_nll` is among the most heavily tested functions in the
# package -- and publishing it would be actively misleading about the part of PyFC that
# most needs to be trustworthy.  `NUMBA_DISABLE_JIT=1` makes numba run the decorated
# functions as ordinary Python, which coverage can trace.
#
# It is not uniformly free, though.  Most files are *faster* without JIT, since they stop
# paying compilation overhead, but `strategy="grid"`'s toy generators are
# `@njit(parallel=True)` vectorized loops, and those degrade to plain Python loops.  A
# whole-suite run under `NUMBA_DISABLE_JIT=1` was abandoned after 20+ minutes (the same
# suite takes ~75 s normally).  Hence the split below: the pathological files run with JIT
# on and give up njit traceability, everything else runs with JIT off, and coverage
# combines the two via `--cov-append`.
#
# Usage:
#     scripts/run_coverage.sh              # terminal report with missing lines
#     scripts/run_coverage.sh --html       # ... and a browsable htmlcov/ tree
#     scripts/run_coverage.sh --xml        # ... and coverage.xml (Cobertura, for CI)
#
# What to measure -- which source, branch coverage, what is excluded -- is NOT set here.
# It lives in `[tool.coverage.run]` in pyproject.toml, so the bare `--cov` below and a
# developer's own `pytest --cov` measure exactly the same thing.  The only thing this
# script adds is the JIT split, which is an environment question, not a coverage setting.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Test files that must run with JIT *enabled*, because disabling it makes them
# pathologically slow rather than merely slower.  Measured under NUMBA_DISABLE_JIT=1:
#
#     test_sparsify_grid.py     > 90 s (timed out; ~14 s with JIT on)
#     test_adaptive_toys.py       72.5 s
#     everything else             < 4 s each
#
# Only `test_sparsify_grid.py` is listed.  `test_adaptive_toys.py` is deliberately left in
# the JIT-disabled run despite dominating its wall time: it is the only real exercise of
# `toys.py`'s adaptive-stopping logic, and moving it here would trade a genuine coverage
# signal for ~70 s.  It is the next candidate if run A's time ever becomes a problem.
#
# Note the failure mode of this list going stale is benign in one direction and not the
# other.  A NEW test file is picked up by run A automatically (run A is "everything except
# these"), so it is measured correctly and at worst runs slowly -- it is never silently
# skipped.  A file that BELONGS here but is missing shows up as a slow CI job, which is
# visible.  Nothing here can quietly produce a wrong number.
JIT_REQUIRED=(
    tests/test_sparsify_grid.py
)

want_html=0
want_xml=0
for arg in "$@"; do
    case "$arg" in
        --html) want_html=1 ;;
        --xml)  want_xml=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

ignores=()
for f in "${JIT_REQUIRED[@]}"; do
    ignores+=("--ignore=$f")
done

python -m coverage erase

# Run A -- JIT disabled, so binned.py's @njit internals become traceable Python.
# Everything except the files above.
echo "==> run A: NUMBA_DISABLE_JIT=1, all tests except ${JIT_REQUIRED[*]}"
NUMBA_DISABLE_JIT=1 python -m pytest tests/ "${ignores[@]}" -q \
    --cov --cov-append --cov-report=

# Run B -- JIT enabled, so the files above run at normal speed.  Coverage of anything
# these reach inside an @njit function is lost, which is the price of the split.
echo "==> run B: JIT enabled, ${JIT_REQUIRED[*]}"
python -m pytest "${JIT_REQUIRED[@]}" -q \
    --cov --cov-append --cov-report=

echo "==> combined report"
python -m coverage report

if [[ $want_html -eq 1 ]]; then
    python -m coverage html
    echo "==> wrote htmlcov/index.html"
fi

if [[ $want_xml -eq 1 ]]; then
    python -m coverage xml
    echo "==> wrote coverage.xml"
fi
