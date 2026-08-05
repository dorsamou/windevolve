# autoresearch — SDWPF wind power forecasting

This is an experiment to have an LLM agent do its own research. You are the
researcher. You will repeatedly modify a forecasting program, score it against
a fixed held-out validation split, and keep the changes that help.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `aug5`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current
   master. You are inside `arms/karpathy/workspace/`, which is its own
   standalone git repository. All git commands you run apply only here.
3. **Read the in-scope files**:
   - `forecaster.py` — the file you modify. The forecasting logic.
   - `../../../evaluation/evaluator.py` — how you are scored. **Read it.**
     Do not modify it.
4. **Verify the split exists**: run `python ../../../paths.py` and confirm
   `VAL_DIR` and `VAL_TRAIN_FILE` both report `[ok]`. If either is MISSING,
   stop and tell the user to generate the split first — you cannot score
   anything without it.
5. **Initialize results.tsv**: create `results.tsv` with just the header row.
   The baseline is recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The task

Predict active power (`Patv`, in kW) for 134 wind turbines, 48 hours ahead at
10-minute resolution — 288 steps per turbine per window.

`forecaster.py` exposes one function:

```python
def forecast(envs):
    """
    envs: dict with "path_to_test_x" (path to an infile CSV containing 14 days
          of history for all 134 turbines, stacked turbine-major), and
          optionally "capacity" (134) and "output_len" (288).
    returns: float array of shape (capacity, output_len, 1), predicted Patv in kW.
    """
```

The evaluator calls this once per validation window, then scores the
predictions using the same metric implementation the competition used, applied
to the held-out validation windows.

**What you CAN do:**
- Modify `forecaster.py`. Everything inside it is fair game: the forecasting
  method, features used, any model you fit, hyperparameters, preprocessing.

**What you CANNOT do:**
- Modify anything outside this workspace directory. `evaluation/` is read-only
  and contains the scorer. Treat it as physics.
- Create additional Python files. Everything lives in `forecaster.py`. If you
  want to fit a model, fit it lazily on the first `forecast()` call and cache
  it in a module-level dict.
- Read, open, or reference `final_phase_test/` in any way. That is the
  competition's held-out test set and it is locked. Using it invalidates the
  entire experiment.
- Install new packages, or add dependencies. Use only what is already present
  in the environment. The exact package list is in `../environment.txt`; check
  there before assuming something is available.

**Training data**: if you fit a model, train it on `VAL_TRAIN_FILE` (the
truncated CSV, days 1–197). Never on the full 245-day file — that contains the
validation period you are scored on, and using it is leakage that makes your
result meaningless. `paths.py` gives you the correct path.

**The goal is simple: get the lowest `competition_score`.** Lower is better. It
is in MW, summed across all 134 turbines, and is ½(RMSE + MAE) as the
competition defines it.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. Conversely, removing
something and getting equal or better results is a great outcome — that's a
simplification win. A 0.05 MW improvement that adds 40 lines of hacky code?
Probably not worth it. A 0.05 MW improvement from deleting code? Definitely
keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: your very first run should always be to establish the
baseline, so run the scorer on `forecaster.py` unmodified.

## Running an experiment

```
python ../../../evaluation/evaluator.py forecaster.py > run.log 2>&1
```

Redirect everything. Do NOT use `tee` or let output flood your context.

## Output format

On success the scorer prints:

```
program : forecaster.py
metric  : OFFICIAL competition grader (kddcup22-sdwpf-evaluation, verbatim)
val_dir : .../evaluation/splits/val_windows
  combined_score: -46.732145
  competition_score: 46.732145
  rmse: 52.104
  mae: 41.360
  latest_rmse: 48.221
  n_windows: 24
  eval_seconds: 31.4
  valid: True
```

Extract what you need with:

```
grep "competition_score:\|valid:\|eval_seconds:\|reason:" run.log
```

**Important**: the evaluator never raises. A broken or rejected program returns
`valid: False` with a `reason`, and a `combined_score` of about -1000000.
So an empty grep means the process itself died (syntax error, killed, OOM);
`valid: False` means your program ran but the evaluator rejected its output.
Read the `reason` — it names the specific check that failed.

Those checks exist because the metric can be gamed: a program producing
degenerate output can score well without forecasting anything. Failing a check
is not a bug in the scorer, it is the scorer working. Do not attempt to work
around one. Shaping output to satisfy a check rather than to forecast
accurately would make the entire experiment worthless.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	competition_score	eval_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. `competition_score` in MW (e.g. 46.732145) — use 999.999999 for crashes and
   for `valid: False`
3. `eval_seconds`, rounded to .1f — use 0.0 for crashes
4. status: `keep`, `discard`, `invalid`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	competition_score	eval_seconds	status	description
a1b2c3d	46.732145	31.4	keep	baseline persistence
b2c3d4e	45.118203	33.1	keep	blend persistence with diurnal climatology
c3d4e5f	47.902118	30.8	discard	linear trend extrapolation on last 6h
d4e5f6g	999.999999	12.2	invalid	per-turbine constant, rejected by output check
e5f6g7h	999.999999	0.0	crash	nan in output, ValueError in reshape
```

Also append to `FAILURE_MODES.md` whenever something fails in a way that
teaches you something — a check you did not anticipate, a constraint of the
task you had not noticed, an approach that cannot work for a structural reason.
One short paragraph per entry, newest at the bottom. This file is your memory
across the whole run; read it when you are stuck.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/aug5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `forecaster.py` with an experimental idea by directly hacking the code
3. `git commit`
4. Run the experiment:
   `python ../../../evaluation/evaluator.py forecaster.py > run.log 2>&1`
5. Read out the results:
   `grep "competition_score:\|valid:\|eval_seconds:\|reason:" run.log`
6. If the grep output is empty, the process died. Run `tail -n 50 run.log` to
   read the stack trace and attempt a fix. If you can't get it working after a
   few attempts, give up on that idea and move on.
7. Record the result in `results.tsv` (do NOT commit `results.tsv` — leave it
   untracked, so a `git reset` does not erase your log)
8. If `competition_score` improved (lower), you "advance" the branch, keeping
   the git commit
9. If it is equal or worse, `git reset` back to where you started

You are a completely autonomous researcher trying things out. If they work,
keep. If they don't, discard. You advance the branch so that you can iterate.
If you feel like you're getting stuck, you can rewind, but do this very
sparingly (if ever).

**Timeout**: a scoring run over 24 windows should take a few minutes. If a run
exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: if a run crashes, use your judgment. Something dumb and easy to fix
(a typo, a missing import) — fix it and re-run. If the idea itself is
fundamentally broken, skip it, log `crash`, and move on.

**NEVER STOP**: once the experiment loop has begun (after the initial setup), do
NOT pause to ask the human whether you should continue. Do NOT ask "should I
keep going?" or "is this a good stopping point?". The human may be asleep and
expects you to continue working indefinitely until manually stopped. You are
autonomous. If you run out of ideas, think harder — re-read the in-scope files
for new angles, re-read `FAILURE_MODES.md`, try combining previous near-misses,
try more radical changes to the forecasting approach. The loop runs until the
human interrupts you, period.