#!/usr/bin/env bash
# path: runs/search_policy_cv__aug14/retrain_full.sh
#
# WHAT THIS DOES
#   Retrains the four GRU checkpoint sets that the two champion programs load,
#   using the FULL 245-day training file instead of the per-fold files.
#
# WHY
#   The champions were searched under 5-fold CV, so their weights are per-fold:
#   F1's saw days 1-95, F5's saw days 1-215. The published baseline table
#   (climatology 50.487 / ARMA 49.605 / XGBoost 46.804 / GRU 46.809) was scored
#   with weights trained on all 245 days. To put a champion in that table it has
#   to be trained the same way.
#
#   This is not leakage: the competition test set is days 246+. The per-fold
#   weights exist for cross-validation only, where fold k must not see its own
#   validation block.
#
# HOW
#   Each set is trained by archive/<id>.py -- the champion program AS IT WAS at
#   the evaluation that produced those weights, not the final forecaster.py.
#   Training mode (pooled vs per-turbine), schedule, batch size and learning rate
#   live in module constants rather than flags, so a later version of the file
#   would silently train something different. Flags passed here are exactly the
#   ones the agents' own retrain scripts passed, and nothing more.
#
#   Output goes to <dir>/FULL/. The champions substitute "{fold}" into their
#   checkpoint paths, so scoring with pin="FULL" resolves every hardcoded
#   template in both programs with no edits to the programs themselves.
#
#   Per-turbine sets are sharded 8 ways across the GPU. Per-turbine seeding
#   (3407 + turbine id, or 1234 + id for ckpt_085) makes shards independent, so
#   a sharded run is identical to a serial one. The pooled set is one job.
#
# RESUMABLE
#   Any set whose output directory is already populated is skipped, so the
#   script can be re-run after an interruption without redoing finished work.
#
# COST
#   Roughly 4-6 hours on an idle RTX 5090. Run it in tmux.

set -euo pipefail

ROOT=$HOME/srip/windevolve_sdwpf
RUN=$ROOT/runs/search_policy_cv__aug14
G=$RUN/greedy-cv
A=$RUN/archive-cv
SHARDS="0-16 17-33 34-50 51-67 68-84 85-101 102-117 118-133"
export OMP_NUM_THREADS=4

FULL_CSV=$(python -c "import sys; sys.path.insert(0,'$ROOT'); from paths import TRAIN_FILE; print(TRAIN_FILE)")
echo "full training file : $FULL_CSV"
echo "started            : $(date)"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo

# sharded <trainer.py> <out_dir> <tag> [flags...]   -- 134 per-turbine models
sharded () {
  local trainer="$1" out="$2" tag="$3"; shift 3
  if [ -d "$out" ] && [ "$(ls "$out" 2>/dev/null | wc -l)" -ge 134 ]; then
    echo "SKIP $tag  ($(ls "$out" | wc -l) files already present)"; return
  fi
  echo "=== $tag  ->  $out"
  echo "    trainer $trainer   start $(date +%T)"
  mkdir -p "$out"
  for R in $SHARDS; do
    python "$trainer" --data "$FULL_CSV" --checkpoints "$out" \
      --turbines "$R" --device cuda "$@" >> "/tmp/${tag}_full.log" 2>&1 &
  done
  wait
  echo "    done $(date +%T)   $(ls "$out" | wc -l) files"
}

# single <trainer.py> <out_dir> <tag> [flags...]    -- one pooled model
single () {
  local trainer="$1" out="$2" tag="$3"; shift 3
  if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
    echo "SKIP $tag  ($(ls "$out" | wc -l) files already present)"; return
  fi
  echo "=== $tag  ->  $out   (pooled: ONE model for all 134 turbines)"
  echo "    trainer $trainer   start $(date +%T)"
  mkdir -p "$out"
  python "$trainer" --data "$FULL_CSV" --checkpoints "$out" \
    --device cuda "$@" > "/tmp/${tag}_full.log" 2>&1
  tail -4 "/tmp/${tag}_full.log"
  echo "    done $(date +%T)   $(ls "$out" | wc -l) files"
}

# ---- greedy champion 119 ----------------------------------------------------
# Two per-turbine sets at hidden 128, same config, different base seed. The pair
# is the agent's controlled test of within-class ensembling: earlier two-GRU
# averages lost, but always paired ckpt_005 with a weaker model, confounding
# "diversity does not pay" with "the partner was worse".
sharded "$G/archive/005.py" "$G/.cache/ckpt_005/FULL" ckpt_005 --hidden 128
sharded "$G/archive/085.py" "$G/.cache/ckpt_085/FULL" ckpt_085 --hidden 128 --seed 1234

# ---- archive champion 120 ---------------------------------------------------
# Member 1: the seed GRU at hidden 48. Full-data weights already exist as the
# baseline table's ckpt_full, so this is a copy rather than a retrain.
mkdir -p "$A/checkpoints"
if [ ! -d "$A/checkpoints/FULL" ]; then
  cp -r "$ROOT/baselines/gru/ckpt_full" "$A/checkpoints/FULL"
fi
echo "=== archive seed member: $(ls "$A/checkpoints/FULL" | wc -l) models (copied from ckpt_full)"

# Member 2: per-turbine refit at hidden 96.
sharded "$A/archive/015.py" "$A/.cache/ckpt_015/FULL" ckpt_015 --hidden 96

# Member 3: pooled farm model -- one network over all 134 turbines, farm-wide
# scaler, interleaved 4-in-32-day holdout, hidden 96, batch 512, lr 3e-4. All of
# that is set inside archive/009.py, so no flags are passed and no sharding is
# used (its original log reads "turbines: 0..133 (n=134)").
single "$A/archive/009.py" "$A/.cache/ckpt_009/FULL" ckpt_009

# ---- verify -----------------------------------------------------------------
echo
echo "finished $(date)"
echo "verify -- the four per-turbine sets need 134 files each;"
echo "          ckpt_009 is pooled and will hold fewer:"
for d in "$G"/.cache/ckpt_005/FULL "$G"/.cache/ckpt_085/FULL \
         "$A"/checkpoints/FULL "$A"/.cache/ckpt_015/FULL "$A"/.cache/ckpt_009/FULL; do
  printf "  %-74s %s\n" "${d#$RUN/}" "$(ls "$d" 2>/dev/null | wc -l)"
done
