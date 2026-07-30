# Autoresearch: SDWPF wind power forecasting

You are running an autonomous research loop. Your objective is to improve
48-hour-ahead wind power forecasting on the SDWPF / Baidu KDD Cup 2022 benchmark
by editing a single file and measuring the result.

**Keep going until told to stop.** Do not pause to ask whether you should
continue, and do not stop after a successful experiment. Run experiments back to
back indefinitely.

## Objective

Minimise `val_score`, the official competition metric:

    per turbine: MAE and RMSE in MW over the surviving (unmasked) steps
    per window:  summed across all 134 turbines
    overall:     averaged across windows, then score = 0.5 * (RMSE + MAE)

**Lower is better.** Because errors are *summed* over turbines, a few bad
turbines dominate the total — a method must work on all of them, not on average.

## Baselines to beat

<!-- TODO(dors): fill in from your baseline runs, all scored on the SAME
     validation split, before the first agent session. -->

| Model | val_score | Notes |
|---|---|---|
| Persistence | _TBD_ | the current seed |
| Climatology | _TBD_ | zero-skill floor |
| ARIMA | _TBD_ | |
| GBDT / XGBoost | _TBD_ | |
| GRU | _TBD_ | official KDD Cup baseline |
| DLinear | _TBD_ | |

Seed-to-seed standard deviation of the GRU baseline: **_TBD_ MW**.

**Any improvement smaller than that standard deviation is noise, not progress.**
For scale, the entire top ten of the original competition spanned about 0.4 MW.
Do not build a chain of changes on top of a difference that small.

## Files

| File | Rule |
|---|---|
| `forecaster.py` | **The one file you edit.** |
| `run_experiment.py` | Runs and scores. Do not modify. |
| `score_final.py` | Locked-test measurement. **Never run this.** |
| `../windevolve_sdwpf/metrics.py` | Official scorer. **Never modify.** |
| `../windevolve_sdwpf/test_data.py` | Official loader. **Never modify.** |
| `../windevolve_sdwpf/evaluator.py` | Shared scoring path. **Never modify.** |
| `../windevolve_sdwpf/val_windows.py` | Builds the split. **Never modify or run.** |
| `results.tsv` | Append one row per experiment. Never rewrite history. |
| `FAILURE_MODES.md` | Append an entry when something breaks or surprises you. |

## The contract

`forecaster.py` must expose:

```python
forecast(envs) -> np.ndarray        # shape (134, 288, 1), active power in kW
```

- `envs["path_to_test_x"]` — path to one window's infile. Your function reads it.
- `envs["path_to_train"]` — training CSV, already truncated to exclude the
  validation period. Safe to fit on.
- `envs["path_to_locations"]` — turbine x/y coordinates, for spatial methods.

**Critical performance constraint:** `forecast()` is called **once per window**.
Any expensive work — reading the training CSV, fitting a model, loading
checkpoints — must be done **once and cached at module level**, not on every
call. Getting this wrong makes an experiment tens of times slower for no gain,
and will usually blow the wall-clock budget.

Predictions must be finite, non-negative, and must vary across the horizon and
across turbines. The scorer rejects all-zero and (near-)constant output at the
farm level.

## Hard constraints

Violating any of these invalidates the experiment.

1. **Never modify anything in `../windevolve_sdwpf/`.** If you believe the
   scorer has a bug, stop and report it to the human rather than working around
   it.
2. **Never read, load, glob, or reference `final_phase_test`.** For the purposes
   of this loop it does not exist. It is scored once at the end by a script you
   do not run.
3. **Never change the horizon (288), the masking rules, the scoring function, or
   the validation split.** These are the benchmark, not the search space.
4. **Respect the wall-clock budget** in `run_experiment.py`. Do not raise it and
   do not move work outside the timed region.
5. **Keep the `RESULT ...` line format intact.**
6. **One hypothesis per experiment.** Change four things at once and a moved
   score teaches you nothing.

## The loop

1. **Set up** (first iteration only). Branch:
   `git checkout -b autoresearch/<tag>`. Confirm `results.tsv` has its header
   row. Run `python run_experiment.py` unmodified to record the seed baseline.
2. **Read** `program.md`, the current `forecaster.py`, and the last ~20 rows of
   `results.tsv`. Do not retry something already tried and failed.
3. **Hypothesise.** One sentence: what you are changing and what you expect to
   happen. Write it down *before* running, not after seeing the result.
4. **Edit** `forecaster.py`. One coherent change.
5. **Run.** `python run_experiment.py 2>&1 | tee logs/exp_<n>.log`
6. **Record.** Append a row to `results.tsv`.
7. **Keep or revert.**
   - Improved: `git add -A && git commit -m "exp <n>: <hypothesis> (<score>)"`.
     This is the new baseline.
   - Not improved, or failed: `git checkout -- forecaster.py`.
     Record it either way. A negative result is a result; do not hide it.
8. **Go to step 2.**

### results.tsv columns

```
exp	timestamp	hypothesis	val_score	rmse	mae	n_windows	elapsed_seconds	kept	notes
```

Tab-separated. `kept` is 1 or 0. `val_score` is `NaN` for a failed run.

## Handling failures

- **`valid=0`** — read the `reason` field. Record with `val_score=NaN`, revert,
  and try something different. Two consecutive failures on the same idea means
  abandon that idea.
- **Degeneracy rejection** (zero or near-constant predictions) — your output does
  not vary enough across the farm. This is a real constraint of the task, not a
  bug in the scorer.
- **`budget_status=OVER_BUDGET`** — the experiment is invalid. The usual cause is
  work inside `forecast()` that should have been cached at module level.
- **Suspiciously large improvement** — better than the best baseline by more than
  a couple of MW means look for a bug before believing it. Check the output
  shape, check you have not accidentally read ground truth from an outfile, and
  check for a degenerate prediction that slipped past the guards.

## Failure-mode log

This loop is itself the object of study. Append an entry to `FAILURE_MODES.md`
whenever something notable happens: an edit that did not apply cleanly, context
you lost track of, an idea you repeated without noticing, a score that moved for
a reason you could not explain, or a stretch where you were only producing
near-identical variations of the same idea.

Be specific and honest. These entries are a research output, not an admission of
failure — documenting how this loop behaves is half the point of running it.

## Research directions

<!-- TODO(dors): your judgement goes here. Prune and extend this list as you
     learn what the loop actually does with it. -->

Roughly in order of expected value:

1. **Better use of history.** The seed repeats the last 288 steps verbatim.
   Persistence decays badly over 48 h. Try a windowed mean of recent steps, a
   fit on recent trend, or blending toward the turbine's climatological profile
   as the horizon grows.
2. **Handling invalid and missing records** in the input history. The seed
   replaces NaN with zero, which drags forecasts toward zero. Interpolate, or
   carry the last valid reading forward, or drop and reindex.
3. **Time-of-day structure.** Wind has a diurnal cycle. A per-turbine
   time-of-day profile fitted once on `path_to_train` is cheap and likely helps.
4. **Fitting an actual model** on `path_to_train` — linear on lag features,
   gradient boosting, a small neural net. Fit once, cache at module level.
5. **Per-turbine vs. shared parameters.** The 134 turbines differ. Shared
   parameters generalise, per-turbine ones fit better. Which wins here is an
   empirical question worth an experiment.
6. **Horizon-aware blending.** The best predictor at step 1 is not the best at
   step 288. Consider varying the method or the blend weight across the horizon.
7. **Spatial structure.** `path_to_locations` gives turbine coordinates.
   Neighbouring turbines see correlated wind, and upstream turbines lead
   downstream ones.

## Notes

- **Single seed per experiment, deliberately.** Do not add seed averaging unless
  the human explicitly asks. The behaviour of this loop under a noisy fitness
  signal is part of what is being measured.
- Under a fixed budget, ten cheap tested ideas beat one expensive untested one.
- Prefer changes you can explain. A score improvement you cannot account for is
  a lead to investigate, not a result to bank.
