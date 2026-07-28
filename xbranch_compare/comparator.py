"""
Generic comparator for the dev vs dev-no-templates cross-branch regression
harness.

Works directly on the flat-keyed .npz archives that compute_fc_intervals's
`output_file` mechanism already produces (see orchestrator.py's
`_save_fc_archive`) -- no custom serialization needed, and no `pyfc`
dependency here either.
"""
import numpy as np


def compare_npz(path_a, path_b, label_a="A", label_b="B", atol=1e-6, rtol=1e-4):
    """
    Returns a list of human-readable mismatch strings (empty list == match).
    Walks every key present in either archive; for array-like leaves,
    compares shape, NaN-mask, then np.allclose on the non-NaN entries.
    Does not stop at the first mismatch -- collects all of them.
    """
    a = np.load(path_a, allow_pickle=True)
    b = np.load(path_b, allow_pickle=True)

    keys_a, keys_b = set(a.files), set(b.files)
    mismatches = []

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    if only_a:
        mismatches.append(f"Keys only in {label_a} ({path_a}): {sorted(only_a)}")
    if only_b:
        mismatches.append(f"Keys only in {label_b} ({path_b}): {sorted(only_b)}")

    for key in sorted(keys_a & keys_b):
        msg = _compare_arrays(a[key], b[key], key, atol=atol, rtol=rtol)
        if msg:
            mismatches.append(msg)

    return mismatches


def _compare_arrays(va, vb, key, atol, rtol):
    va = np.asarray(va)
    vb = np.asarray(vb)

    if va.shape != vb.shape:
        return f"[{key}] shape mismatch: {va.shape} vs {vb.shape}"

    if va.dtype.kind in "fc" or vb.dtype.kind in "fc":
        va = va.astype(float, copy=False)
        vb = vb.astype(float, copy=False)
        nan_a = np.isnan(va)
        nan_b = np.isnan(vb)
        if not np.array_equal(nan_a, nan_b):
            return f"[{key}] NaN pattern mismatch"

        finite_a = va[~nan_a]
        finite_b = vb[~nan_b]
        if finite_a.size == 0:
            return None
        if not np.allclose(finite_a, finite_b, atol=atol, rtol=rtol):
            max_diff = float(np.max(np.abs(finite_a - finite_b)))
            return f"[{key}] value mismatch, max abs diff = {max_diff:.3e}"
        return None

    if not np.array_equal(va, vb):
        return f"[{key}] value mismatch (non-float dtype)"
    return None


def report(mismatches, scenario_label):
    if not mismatches:
        print(f"[xbranch] {scenario_label}: MATCH (0 mismatches)")
        return True
    print(f"[xbranch] {scenario_label}: {len(mismatches)} MISMATCH(ES)")
    for m in mismatches:
        print(f"    - {m}")
    return False
