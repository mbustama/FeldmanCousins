"""
Diffs the .npz archives produced by xbranch_old_api.py and
xbranch_new_api.py for the cross-branch scenarios (a) and (c). Exits
non-zero if either scenario shows any mismatch (see comparator.py).

Scenario (b) (2D-histogram same-branch invariant) and (d) (3-component
hand-computed reference) are self-checked via assertions inside
xbranch_new_api.py itself -- this script only handles the genuinely
cross-branch comparisons.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparator import compare_npz, report

parser = argparse.ArgumentParser()
parser.add_argument("--old-dir", required=True)
parser.add_argument("--new-dir", required=True)
args = parser.parse_args()

all_ok = True
for scenario in ("scenario_a", "scenario_c"):
    old_path = os.path.join(args.old_dir, f"{scenario}.npz")
    new_path = os.path.join(args.new_dir, f"{scenario}.npz")
    mismatches = compare_npz(old_path, new_path, label_a="dev (old)", label_b="dev-no-templates (new)")
    ok = report(mismatches, scenario)
    all_ok = all_ok and ok

if not all_ok:
    print("[xbranch] FAILED: at least one cross-branch scenario mismatched.")
    sys.exit(1)

print("[xbranch] PASSED: all cross-branch scenarios match.")
