#!/usr/bin/env bash
# Island ablation: run each variant N_REPEATS times, all settings fixed except
# num_islands / migration. Uses run.sh so every run is snapshotted + committed.
#
# Usage:  ./ablation.sh <iterations> <n_repeats>
# Example: SDWPF_N_ORIGINS=195 ./ablation.sh 100 3
set -euo pipefail

ITERS="${1:?iterations, e.g. 100}"
REPS="${2:?repeats per variant, e.g. 3}"
OE="/Users/dorsam/Documents/SRIP 2026/openevolve-main/openevolve-run.py"

run_one () {   # $1=variant_label  $2=config_file  $3=rep_number
  local rid="${1}_r${3}"
  local rundir="runs/${rid}"
  [ -e "$rundir" ] && { echo "SKIP $rid (exists)"; return 0; }
  mkdir -p "$rundir"
  git add -A; git commit --allow-empty -q -m "ablation ${rid}"
  {
    echo "run_id: ${rid}"; echo "variant: ${1}"; echo "config: ${2}"
    echo "iterations: ${ITERS}"; echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit: $(git rev-parse HEAD)"
    env | grep -E '^SDWPF_' | sort || true
  } > "$rundir/run_manifest.txt"
  echo "=== $rid ==="
  python "$OE" initial_program.py evaluator.py --config "$2" \
    --iterations "$ITERS" --output "$rundir/openevolve_output"
}

for r in $(seq 1 "$REPS"); do run_one islands3 config_islands3.yaml "$r"; done
for r in $(seq 1 "$REPS"); do run_one islands1 config_islands1.yaml "$r"; done
echo "=== ablation complete: runs/islands{1,3}_r{1..$REPS} ==="
