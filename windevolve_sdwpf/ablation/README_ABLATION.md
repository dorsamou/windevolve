# Search-space ablation: HPO vs model discovery

Drop this folder in at `windevolve_sdwpf/ablation/`.

## What the experiment is

One variable: **which region of the seed program OpenEvolve is allowed to
rewrite.**

| Arm | Mutable region | Frozen |
|---|---|---|
| `hpo` | the `PARAMS` dict — 10 numeric knobs | all forecasting logic |
| `model` | the modelling functions | `PARAMS`, and the `forecast()` I/O contract |

Everything else is held constant by construction:

- `seed_hpo.py` and `seed_model.py` contain the **same logic and the same
  PARAMS**. `seed_model.py` was generated from `seed_hpo.py` by moving the
  markers, not by retyping. Verified: byte-identical predictions, max abs diff
  `0.0`, and both accepted by the real grader with the same score.
- The two YAML bodies are **md5-identical** below the comment header.
- Same evaluator, same validation windows, same 3 seeds, same iteration budget.

So a difference between arms is attributable to the search space. That is the
whole design.

**Three repeats is a floor.** Toledo et al. (2025) App. I resample the same
experiment down to 3, 5, 10 and 15 seeds and show algorithm rankings flipping
between draws; they recommend 10 minimum, 20 preferred. Your own islands
ablation already came out null at 3. If the per-arm ranges overlap here — and
they probably will — the correct statement is "no separation detected at n=3",
not "HPO wins".

**This is a val-val result.** `final_phase_test` stays locked.

## Setup

### 0. Build the validation split (this did not exist yet)

`val_windows.py` goes in `windevolve_sdwpf/`, not in this folder. Nothing runs
without it — the evaluator raises immediately if `val_split/` is absent.

```bash
cd windevolve_sdwpf

# look at the plan and the disk estimate first; writes nothing
python val_windows.py --train-file ~/srip/wind_forecasting_data/sdwpf_kddcup/wtbdata_245days.csv --dry-run

# build it
python val_windows.py --train-file <245-day csv> --n-windows 48

# check it
python val_windows.py --verify --verify-all
```

Defaults: targets in the last 30 days, 48 windows evenly spaced (deterministic,
no RNG), 14-day input / 48-hour output per window, matching the competition
geometry. `val_train.csv` is truncated at the first target step, so no validation
target ever appears in the training file — `--verify` checks that explicitly and
fails loudly if it doesn't hold.

Two things worth knowing. The split is written in the official on-disk format
with **NaNs preserved**: `official_metrics.turbine_scores` masks a timestep if
any column of that row is NaN, so filling missing values would silently change
which timesteps get scored. And 48 windows of 134 turbines is on the order of
**1–2 GB** — check the dry-run estimate against free space 
before committing.

Verified end to end here on a scaled-down synthetic dataset: split builds from a
deliberately shuffled source CSV, `--verify` passes, both seeds score through the
real `evaluator.py` to identical values, a manifest tampered to overlap the
targets is caught as `LEAKAGE`, and an all-zero forecaster is rejected by the
official degeneracy guard.

### 1. API key

Create a key in the Claude Console, then:

```bash
export OPENAI_API_KEY="sk-ant-..."      # yes, OPENAI_ — OpenEvolve reads that name
```

OpenEvolve speaks the OpenAI chat-completions protocol and Anthropic exposes an
OpenAI-compatible endpoint, so `api_base: https://api.anthropic.com/v1/` with a
Claude model ID works unmodified. Anthropic documents this as a compatibility
layer intended for evaluation rather than production, and notes some features
(prompt caching among them) aren't available through it — none of which this run
needs. Docs: <https://platform.claude.com/docs/en/api/openai-sdk>

Smoke-test the key before committing:

```bash
curl https://api.anthropic.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"claude-sonnet-5","max_tokens":16,
       "messages":[{"role":"user","content":"reply with OK"}]}'
```


### 2. Confirm the arms start level

```bash
python evaluator.py ablation/seed_hpo.py
python evaluator.py ablation/seed_model.py
```

The two `competition_score` values must match exactly. If they don't, stop.

## Run

```bash
cd windevolve_sdwpf
chmod +x ablation/run_ablation.sh
tmux new -s abl
export OPENAI_API_KEY="sk-ant-..."
bash ablation/run_ablation.sh 2>&1 | tee ablation/runs/driver.log
```

To halve wall-clock, run the arms in two tmux sessions:

```bash
tmux new -s abl_hpo   ; ARMS="hpo"   bash ablation/run_ablation.sh
tmux new -s abl_model ; ARMS="model" bash ablation/run_ablation.sh
```

Completed runs write a `COMPLETED` marker and are skipped on re-invocation, so
you can safely rerun after a crash.



```yaml
primary_model: "claude-haiku-4-5-20251001"
secondary_model: "claude-haiku-4-5-20251001"
```

## After run 

```bash
python ablation/collect_results.py --csv ablation/summary.csv
```

Prints per-run scores, per-arm mean/stdev/range, and a best-of-k curve within
each arm. It deliberately does not declare a winner.

## Files

| File | Purpose |
|---|---|
| `seed_hpo.py` | seed, EVOLVE-BLOCK around `PARAMS` only |
| `seed_model.py` | seed, EVOLVE-BLOCK around the modelling functions |
| `config_hpo.yaml` / `config_model.yaml` | identical bodies, one per arm |
| `run_ablation.sh` | 2 arms x 3 seeds, resumable, per-run logs |
| `collect_results.py` | summary table + best-of-k |
| `EVALUATOR_PATCH.md` | the two required evaluator fixes |
