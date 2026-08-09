#!/usr/bin/env bash
# path: windevolve_sdwpf/leaderboard/FDSTT/run_all.sh
#
# Runs the complete FDSTT pipeline end to end and prints both scores.
#
#   ./leaderboard/FDSTT/run_all.sh            search fit + val score
#   ./leaderboard/FDSTT/run_all.sh --full     also refit on 245 days + test score
#
# Run from the repo root. Roughly 2-6 hours depending on GPU contention;
# the DMST training in step 4 dominates.

set -euo pipefail

HERE="leaderboard/FDSTT"
DEV="${FDSTT_DEVICE:-cuda:0}"
STRIDE="${STRIDE:-6}"
EPOCHS="${EPOCHS:-40}"
FULL=0
[ "${1:-}" = "--full" ] && FULL=1

echo "=============================================="
echo " FDSTT  (team HIK, rank 1, Baidu KDD Cup 2022)"
echo " device $DEV | stride $STRIDE | max epochs $EPOCHS"
echo "=============================================="

echo; echo "[1/6] graphs"
python "$HERE/graphs.py" --inspect

echo; echo "[2/6] fitted artifacts, days 1-197"
python "$HERE/data.py" fit --fit-dir "$HERE/fit_val"

echo; echo "[3/6] train/inference parity"
python "$HERE/data.py" check --fit-dir "$HERE/fit_val"

echo; echo "[4/6] ST-Tree (module 3)"
python "$HERE/sttree.py" fit --fit-dir "$HERE/fit_val"

echo; echo "[5/6] DMST (module 2) -- the long one"
python "$HERE/train.py" --fit-dir "$HERE/fit_val" --out "$HERE/ckpt_val" \
    --epochs "$EPOCHS" --stride "$STRIDE" --device "$DEV"

BEST=$(python -c "import json,sys;h=json.load(open('$HERE/ckpt_val/history.json'));\
c=[e for e in h if 'inner_score' in e];\
print(min(c,key=lambda e:e['inner_score'])['epoch'] if c else len(h))")
echo "best epoch: $BEST"

echo; echo "[6/6] VAL SCORE (24 windows, days 198-245)"
FDSTT_CKPT="$HERE/ckpt_val" FDSTT_FIT="$HERE/fit_val" FDSTT_DEVICE="$DEV" \
    python evaluation/evaluator.py "$HERE/program.py"

echo; echo "--- DMST alone, for HIK's ablation comparison ---"
FDSTT_CKPT="$HERE/ckpt_val" FDSTT_FIT="$HERE/fit_val" FDSTT_DEVICE="$DEV" \
FDSTT_DMST_ONLY=1 python evaluation/evaluator.py "$HERE/program.py"

if [ "$FULL" -eq 1 ]; then
    TRAIN_CSV="${SDWPF_TRAIN_FILE:-$HOME/srip/wind_forecasting_data/sdwpf_kddcup/sdwpf_245days_v1.csv}"
    echo; echo "=== FINAL REFIT on all 245 days, $BEST epochs ==="
    python "$HERE/data.py"   fit --fit-file "$TRAIN_CSV" --fit-dir "$HERE/fit_full"
    python "$HERE/sttree.py" fit --fit-file "$TRAIN_CSV" --fit-dir "$HERE/fit_full"
    python "$HERE/train.py" --fit-dir "$HERE/fit_full" --out "$HERE/ckpt_full" \
        --fit-file "$TRAIN_CSV" --epochs-fixed "$BEST" \
        --stride "$STRIDE" --device "$DEV"

    echo; echo "TEST SCORE (final_phase_test)"
    FDSTT_CKPT="$HERE/ckpt_full" FDSTT_FIT="$HERE/fit_full" FDSTT_DEVICE="$DEV" \
        python evaluation/competition_score.py "$HERE/program.py"
fi

echo; echo "done."
