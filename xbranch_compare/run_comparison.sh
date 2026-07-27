#!/usr/bin/env bash
# Cross-branch regression harness driver: proves dev vs dev-no-templates
# produce numerically-identical results for equivalent models (the
# S_model/B_model removal refactor changed signatures only, never numerics).
#
# Usage: run from anywhere; paths below are resolved relative to this
# script's location, not the caller's cwd.
#
# What it does:
#   1. Adds a temporary `git worktree` for the `dev` branch (this repo's
#      current checkout is assumed to already be on dev-no-templates or a
#      descendant of it -- the harness runs the NEW api directly against
#      the current checkout, and only needs a worktree for the OLD api).
#   2. Runs xbranch_old_api.py against that worktree's pyfc.
#   3. Runs xbranch_new_api.py against the current checkout's pyfc.
#   4. Diffs scenario_a.npz and scenario_c.npz (cross-branch, must match
#      exactly) via comparator.py. Scenario (b)/(d) invariants are
#      self-checked (assertions) inside xbranch_new_api.py.
#   5. Removes the temporary worktree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKTREE_PATH="$(mktemp -d /tmp/pyfc_dev_worktree.XXXXXX)"
OUTPUT_OLD="${SCRIPT_DIR}/output/old"
OUTPUT_NEW="${SCRIPT_DIR}/output/new"

cleanup() {
    echo "[xbranch] Removing temporary worktree at ${WORKTREE_PATH}"
    git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE_PATH}" 2>/dev/null || rm -rf "${WORKTREE_PATH}"
}
trap cleanup EXIT

echo "[xbranch] Adding worktree for 'dev' at ${WORKTREE_PATH}"
git -C "${REPO_ROOT}" worktree add "${WORKTREE_PATH}" dev

echo "[xbranch] Running OLD api scenarios (dev)..."
python "${SCRIPT_DIR}/xbranch_old_api.py" --pyfc-root "${WORKTREE_PATH}" --output-dir "${OUTPUT_OLD}"

echo "[xbranch] Running NEW api scenarios (dev-no-templates)..."
python "${SCRIPT_DIR}/xbranch_new_api.py" --pyfc-root "${REPO_ROOT}" --output-dir "${OUTPUT_NEW}"

echo "[xbranch] Diffing cross-branch scenarios (a) and (c)..."
python "${SCRIPT_DIR}/compare_results.py" --old-dir "${OUTPUT_OLD}" --new-dir "${OUTPUT_NEW}"
