#!/usr/bin/env bash
# Launch an OpenEvolve run, git-committing the source first so the manifest's
# commit hash always points at the exact code that ran, and sending all output
# into a per-run folder.
#
# One-time setup (run once in this folder):
#   git init && git add -A && git commit -m "initial harness"
#
# Usage:
#   ./run.sh <run_id> <iterations> [notes...]
# Example:
#   SDWPF_N_ORIGINS=195 ./run.sh r001_islands3 60 "first full 195-origin run"

set -euo pipefail

RUN_ID="${1:?need a run_id, e.g. r001_islands3}"
ITERS="${2:?need iteration count, e.g. 60}"
shift 2 || true
NOTES="${*:-}"

OE_LAUNCHER="/Users/dorsam/Documents/SRIP 2026/openevolve-main/openevolve-run.py"
RUN_DIR="runs/${RUN_ID}"

if [ -e "$RUN_DIR" ]; then
  echo "ERROR: $RUN_DIR already exists. Pick a new run_id (don't overwrite history)."
  exit 1
fi
mkdir -p "$RUN_DIR"

# 1. Commit the current source BEFORE running, so HEAD = the code about to run.
#    --allow-empty means a run with no code changes still gets its own commit
#    (a clean per-run timeline). runs/ and data are excluded via .gitignore.
git add -A
git commit --allow-empty -q -m "run ${RUN_ID}: ${NOTES}"
COMMIT="$(git rev-parse HEAD)"

# 2. Manifest: the settings that define this run, including the ones that live
#    only on the command line (SDWPF_* env vars) and the exact commit.
{
  echo "run_id: ${RUN_ID}"
  echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "iterations: ${ITERS}"
  echo "notes: ${NOTES}"
  echo "git_commit: ${COMMIT}"
  echo "--- SDWPF_* environment ---"
  env | grep -E '^SDWPF_' | sort || echo "(none set -> evaluator defaults used)"
  echo "--- command ---"
  echo "python \"$OE_LAUNCHER\" initial_program.py evaluator.py --config config.yaml --iterations ${ITERS}"
} > "$RUN_DIR/run_manifest.txt"

echo "=== snapshot written to $RUN_DIR (commit ${COMMIT:0:8}) ==="
cat "$RUN_DIR/run_manifest.txt"
echo "=== launching ==="

# 3. Run, sending OpenEvolve's output into this run's folder.
python "$OE_LAUNCHER" initial_program.py evaluator.py \
  --config config.yaml --iterations "${ITERS}" \
  --output "$RUN_DIR/openevolve_output"

echo "=== done. everything for this run is in $RUN_DIR ==="
echo "To reconstruct the exact code that ran: git checkout ${COMMIT:0:8}"
