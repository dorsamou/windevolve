#!/usr/bin/env bash
# path: windevolve_sdwpf/ablation/run_ablation.sh
#
# Description:
#   Runs the HPO vs model-discovery search-space ablation. Both arms use the same
#   evaluator, the same config body, the same seed behaviour and the same seed
#   list -- the ONLY difference is which region of the seed file is marked
#   evolvable. Runs are sequential within an arm and logged per run, so a crash
#   costs one run rather than the night.
#
# Example usage:
#   cd windevolve_sdwpf
#   export OPENAI_API_KEY="sk-ant-..."
#   tmux new -s abl
#   bash ablation/run_ablation.sh 2>&1 | tee ablation/runs/driver.log
#
#   # to run the two arms in parallel instead (halves wall clock):
#   tmux new -s abl_hpo   ; ARMS="hpo"   bash ablation/run_ablation.sh
#   tmux new -s abl_model ; ARMS="model" bash ablation/run_ablation.sh

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"

ARMS="${ARMS:-hpo model}"                 # which arms to run
SEEDS="${SEEDS:-1 2 3}"                   # repeats; 3 is your standing minimum
RUNS_DIR="${RUNS_DIR:-$HERE/runs}"        # where checkpoints and logs land
MAX_WINDOWS="${SDWPF_MAX_WINDOWS:-0}"     # 0 = all validation windows

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY is not exported. Set it to your Anthropic key." >&2
  exit 1
fi

mkdir -p "$RUNS_DIR"
export SDWPF_MAX_WINDOWS="$MAX_WINDOWS"
export SDWPF_EVAL_BUDGET="${SDWPF_EVAL_BUDGET:-600}"   # in-evaluator wall-clock guard

echo "=========================================================="
echo " search-space ablation"
echo " arms         : $ARMS"
echo " seeds        : $SEEDS"
echo " max windows  : ${MAX_WINDOWS} (0 = all)"
echo " eval budget  : ${SDWPF_EVAL_BUDGET}s per candidate"
echo " runs dir     : $RUNS_DIR"
echo " started      : $(date -Is)"
echo "=========================================================="

for arm in $ARMS; do
  for seed in $SEEDS; do
    tag="${arm}_seed${seed}"
    out="$RUNS_DIR/$tag"
    log="$RUNS_DIR/${tag}.log"

    if [[ -f "$out/COMPLETED" ]]; then
      echo "[skip] $tag already completed"
      continue
    fi

    echo "[run ] $tag  ->  $log  ($(date -Is))"
    mkdir -p "$out"

    # random_seed is the only per-run difference; everything else is shared.
    cfg="$out/config.yaml"
    sed "s/^random_seed:.*/random_seed: ${seed}/" "$HERE/config_${arm}.yaml" > "$cfg"

    start=$(date +%s)
    python -m openevolve.cli \
        "$HERE/seed_${arm}.py" \
        "$REPO/evaluator.py" \
        --config "$cfg" \
        --output "$out" \
        > "$log" 2>&1
    rc=$?
    dur=$(( $(date +%s) - start ))

    if [[ $rc -eq 0 ]]; then
      touch "$out/COMPLETED"
      echo "[done] $tag  rc=$rc  ${dur}s"
    else
      echo "[FAIL] $tag  rc=$rc  ${dur}s  -- see $log" >&2
    fi
  done
done

echo "=========================================================="
echo " finished: $(date -Is)"
echo " collect:  python ablation/collect_results.py"
echo "=========================================================="
