#!/bin/bash
# Noise floor for the GRU baseline: retrain with N seeds, score each, report the spread.
# Run from the package root on the server, inside tmux (each seed is ~80 min).
set -e
cd /home/dorsa/srip/windevolve_sdwpf
export SDWPF_TRAIN_FILE="/home/dorsa/srip/wind_forecasting_data/sdwpf_kddcup/sdwpf_245days_v1.csv"
export SDWPF_TEST_DIR="/home/dorsa/srip/wind_forecasting_data/sdwpf_kddcup/final_phase_test"

SEEDS="1 2 3 4 5"            # add more if compute allows (5-10 recommended)
RESULTS_FILE="baseline_models/gru/noise_floor_scores.txt"
echo "seed,score" > "$RESULTS_FILE"

for s in $SEEDS; do
    CKPT="baseline_models/gru/checkpoints_seed${s}"
    echo "=== training GRU with seed $s -> $CKPT ==="
    SDWPF_GRU_CHECKPOINTS="$CKPT" python baseline_models/gru/gru_baseline.py \
        --seed "$s" --device cuda --data "$SDWPF_TRAIN_FILE"

    echo "=== scoring seed $s ==="
    SCORE=$(SDWPF_GRU_CHECKPOINTS="$CKPT" python evaluator.py baseline_models/gru/gru_baseline.py \
        | grep "competition_score" | awk '{print $2}')
    echo "${s},${SCORE}" >> "$RESULTS_FILE"
    echo "seed $s -> score $SCORE"
done

echo ""
echo "=== NOISE FLOOR SUMMARY ==="
python - <<'PY'
import numpy as np
rows = [l.strip().split(",") for l in open("baseline_models/gru/noise_floor_scores.txt").readlines()[1:]]
scores = np.array([float(r[1]) for r in rows])
print("seeds     :", len(scores))
print("scores    :", ", ".join("{:.3f}".format(x) for x in scores))
print("mean      : {:.3f}".format(scores.mean()))
print("std dev   : {:.3f}".format(scores.std(ddof=1)))
print("min-max   : {:.3f} - {:.3f}  (spread {:.3f})".format(scores.min(), scores.max(), scores.ptp()))
print()
print("Paper GRU : 47.850   |   your single run: 46.69")
gap = abs(47.850 - scores.mean())
print("gap from paper mean: {:.3f}  ({:.1f} std devs)".format(gap, gap / scores.std(ddof=1) if scores.std(ddof=1) > 0 else 0))
PY
