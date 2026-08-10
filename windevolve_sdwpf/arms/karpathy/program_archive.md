# autoresearch — SDWPF wind power forecasting

This is an experiment to have an LLM agent do its own research. You are the
researcher. You will repeatedly modify a forecasting program, score it against
a fixed held-out validation split, and keep the changes that help.

This brief is shared by two concurrent runs that start from the same model and
differ only in how the loop selects what to work on next. Do not look for the
other run.

## Setup

Your worktree has already been created and seeded by the human. You are not
starting from scratch and you should not create a branch. To set up:

1. **Confirm your run parameters**: the human's opening message gives you a run
   tag and an evaluation budget N. If either is missing, ask before proceeding.
2. **Confirm the git state**: `git status` and `git log --oneline -1`. You
   should be on a branch named `autoresearch/<tag>` with one seed commit and a
   clean tree. If the tree is dirty or the branch name does not match your tag,
   stop and tell the human.
3. **Read the in-scope files**:
   - `forecaster.py` — the file you modify. The forecasting logic. It has been
     seeded with a working baseline model; read it in full before changing it.
   - `../../../../evaluation/evaluator.py` — how you are scored. Read it. Do
     not modify it, and do not follow its imports into the other scorer files.
4. **Verify the split exists**: run `python ../../../../paths.py` and confirm
   `VAL_DIR` and `VAL_TRAIN_FILE` both report `[ok]`. If either is MISSING,
   stop and tell the human to generate the split first — you cannot score
   anything without it.
5. **Confirm `results.tsv` exists** with just its header row. It has been
   created for you. Do not commit it.
6. **Confirm and go**: report the seed model, the tag, and the budget.

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
  You may replace the seeded model entirely if you have reason to.

**What you CANNOT do:**
- Modify anything outside your own worktree directory. `evaluation/` is
  read-only and contains the scorer. Treat it as physics. `baselines/` is
  read-only and holds shared artifacts other runs depend on — never write
  there.
- Run git anywhere except your own worktree. Your worktree is the only
  repository you touch. Never `cd` to a parent directory and run git there,
  never stage or commit outside your worktree, and never `git add` a path
  beginning with `../`. If a command needs a directory change, use a subshell
  that returns, and never combine `cd` with a git write.
- Inspect other branches, other worktrees, or any other run's results. Several
  runs are executing concurrently from different starting models, and they must
  stay independent. Do not run `git log` or `git diff` against a branch other
  than your own.
- Create additional Python files. Everything lives in `forecaster.py`. If you
  want to fit a model, fit it lazily on the first `forecast()` call and cache
  it in a module-level dict, or retrain a checkpoint (see below).
- Build your own validation harness, proxy scorer, or internal test split, or
  evaluate candidate configurations against anything other than the official
  scorer. See "No surrogate scoring" below — this one is easy to violate by
  accident, so read it.
- Read, open, or reference `final_phase_test/` in any way. That is the
  competition's held-out test set and it is locked. Using it invalidates the
  entire experiment.
- Install new packages, or add dependencies. Use only what is already present
  in the environment. The exact package list is in `../../environment.txt`;
  check there before assuming something is available.
- Read, open, or reference any file under `val_windows/outfile/`. Those are the
  ground-truth targets for the validation windows. Your program receives an
  infile path and must forecast from history alone. Reading the corresponding
  outfile — or deriving its path from the infile path — is not forecasting, and
  invalidates the entire run. The evaluator reads the outfiles; you never do.
- Read `competition_score.py`. It scores against the locked test set and
  contains nothing relevant to your task.

**Training data**: if you fit a model, train it on `VAL_TRAIN_FILE` (the
truncated CSV, days 1–197). Never on the full 245-day file — that contains the
validation period you are scored on, and using it is leakage that makes your
result meaningless. `paths.py` gives you the correct path.

**Checkpoints**: your seed loads 134 per-turbine GRU checkpoints from
`./checkpoints/`. That directory is the seed's and is IMMUTABLE — never write
to it, never delete from it, never train into it.

You may retrain. Retraining all 134 turbines takes about an hour, so treat it
as a deliberate, expensive move rather than a routine step. Two hard rules:

1. **Retrain into a versioned directory.** Use `.cache/ckpt_<id>/`, where
   `<id>` is the id you will log for that experiment. Then set the checkpoint
   path INSIDE `forecaster.py` to that literal directory, so the code and the
   weights it depends on travel together. A program whose code says
   `.cache/ckpt_014/` must keep behaving identically if it is re-run later.
   Never point a program at a directory that a subsequent retrain will
   overwrite.
2. **Retraining is capped at 15 times.** Count your `retrain` rows in
   `results.tsv` before starting another. This is a wall-clock limit, not a
   quality judgment.

Scoring a program takes about 8 seconds against a 600-second in-evaluator
budget, so there is a great deal of room for inference-time computation before
timeouts matter. Training is where the cost is.

## No surrogate scoring

`python ../../../../evaluation/evaluator.py` is your ONLY source of feedback
about how good a forecaster is. Every comparison you make between two candidate
forecasters must go through it, and must be logged as a row in `results.tsv`.

Concretely, you may NOT:

- carve your own train/validation split out of `VAL_TRAIN_FILE` and score
  candidates against it
- write a scoring function — RMSE, MAE, the competition metric, or any proxy
  for forecast quality — and use it to compare candidates
- sweep a hyperparameter by evaluating many settings internally and keeping the
  best
- hold out any period, turbine subset, or set of windows for your own scoring
- do any of the above inline inside `forecaster.py`, in a scratch script, in a
  Python one-liner, or in your head from printed numbers

You MAY inspect the training data freely: plot it, compute summary statistics,
look at distributions, autocorrelations, per-turbine levels, missingness. The
line is between *understanding the data* and *scoring a forecaster*. Computing
the mean Patv of turbine 7 is understanding. Computing what a candidate would
have predicted for a held-out period and how far off it was is scoring, and it
is not allowed.

This is a constraint of the experiment, not a claim that internal validation is
bad practice. This run measures how far the loop gets on a fixed number of
evaluations, so an evaluation has to mean the same thing in every run. A
self-built harness lets you test twenty ideas and log one, which makes the
count meaningless. Fitting model parameters on the training data is fine and
expected — that is training, not scoring. Choosing between two candidate
forecasters is scoring, and it costs one row.

If you catch yourself having done this, do not hide it: note it in
`FAILURE_MODES.md` and tell the human when you report at the end.

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
baseline, so run the scorer on `forecaster.py` unmodified. This is evaluation
1 of your budget.

## Running an experiment

```
python ../../../../evaluation/evaluator.py forecaster.py > run.log 2>&1
cp run.log logs/<id>.log        # same <id> you will log in results.tsv
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

The TSV has a header row and 6 columns:

```
id	parent_id	competition_score	eval_seconds	status	description
```

1. `id` — zero-padded 3-digit sequence number, starting at 001 for the seed
   baseline. This is also the name of the program's archive file.
2. `parent_id` — the id of the program this one was edited from (`000` for the
   seed)
3. `competition_score` in MW (e.g. 46.732145) — use 999.999999 for crashes and
   for `valid: False`
4. `eval_seconds`, rounded to .1f — use 0.0 for crashes
5. status: see the loop section below for your arm's status values
6. short text description of what this experiment tried

Example:

```
id	parent_id	competition_score	eval_seconds	status	description
001	000	44.537366	8.3	scored	seed baseline unmodified
002	001	44.118203	9.1	scored	clip negative predictions to zero
003	002	45.902118	8.8	scored	linear trend extrapolation on last 6h
004	002	999.999999	2.2	invalid	per-turbine constant, rejected by output check
005	002	999.999999	0.0	crash	nan in output, ValueError in reshape
006	002	999.999999	0.0	retrain	refit with hidden=96 into .cache/ckpt_006/
```

Also append to `FAILURE_MODES.md` whenever something fails in a way that
teaches you something — a check you did not anticipate, a constraint of the
task you had not noticed, an approach that cannot work for a structural reason.
One short paragraph per entry, newest at the bottom. This file is your memory
across the whole run; read it when you are stuck.

## The experiment loop

This arm does NOT keep a single lineage and does NOT revert. Every program you
score is kept in the archive, whatever it scored, and each iteration starts
from a program the sampler chooses for you — which may be worse than the best
one found so far. That is deliberate. Do not "fix" it by going back to your
best program on your own initiative.

LOOP until the budget is exhausted:

1. Choose the starting point by running the sampler. It overwrites
   `forecaster.py` with the program it selected and prints its id:
   `python ../../sample_parent.py`
   Use the program it gives you. Do not overrule it, do not re-run it hoping
   for a different draw, and do not paste in a different program instead.
2. Read `forecaster.py`. It may not be the program you worked on last
   iteration. Orient yourself before editing.
3. Tune `forecaster.py` with an experimental idea by directly hacking the code
4. `git commit`
5. Run the experiment:
   `python ../../../../evaluation/evaluator.py forecaster.py > run.log 2>&1`
6. Read out the results:
   `grep "competition_score:\|valid:\|eval_seconds:\|reason:" run.log`
7. If the grep output is empty, the process died. Run `tail -n 50 run.log` to
   read the stack trace and attempt a fix. If you can't get it working after a
   few attempts, give up on that idea and move on.
8. Copy the scored program to `archive/<id>.py`, and record the result in
   `results.tsv` with `parent_id` set to the id the sampler printed (do NOT
   commit `results.tsv` — leave it untracked).
9. Go back to step 1. There is no keep/discard decision and no `git reset`.

Status values for this arm: `scored`, `invalid`, `crash`, `retrain`. There is
no `keep` or `discard` — a worse score is still a kept archive entry.

You are a completely autonomous researcher. Because poor programs stay in the
archive and can come back as starting points, an idea that scores badly is not
wasted: it may be the right base for a later idea. Use `results.tsv` and
`FAILURE_MODES.md` to keep track of what each lineage has already tried.

**Timeout**: a scoring run over 24 windows takes about 8 seconds for the seed.
If a run exceeds 20 minutes, kill it and treat it as a failure.

**Crashes**: if a run crashes, use your judgment. Something dumb and easy to fix
(a typo, a missing import) — fix it and re-run. If the idea itself is
fundamentally broken, skip it, log `crash`, and move on.

**Archiving**: after EVERY scored experiment, copy the exact program you scored
into the archive so it is recoverable:

```
cp forecaster.py archive/<id>.py
```

Do this even for `invalid` and `crash` rows. The archive is the record of what
was tried; a program that is not archived did not happen.

## The budget

Your budget is N evaluations, given to you in the opening message.

**An evaluation is one data row in `results.tsv`.** Every row counts toward N,
whatever its status: `scored`, `invalid`, `crash`, and `retrain` all consume
one unit. A retraining run counts even though it produces no score.
Reading files, thinking, editing code, and fixing a syntax error before you
score do not count — only actually running the scorer, or retraining.

Check your remaining budget with `wc -l results.tsv` (subtract 1 for the
header). Spend it deliberately: a crash costs the same as a good idea. Because
the scorer is your only feedback (see "No surrogate scoring"), every comparison
between two candidates costs one row — so prefer a small number of substantial,
well-reasoned changes over sweeping one knob across many values.

**Do NOT pause to ask the human whether you should continue.** Do NOT ask
"should I keep going?" or "is this a good stopping point?". The human may be
asleep and expects you to work until the budget is spent. You are autonomous.
If you run out of ideas, think harder — re-read the in-scope files for new
angles, re-read `FAILURE_MODES.md`, try combining previous near-misses, try
more radical changes to the forecasting approach.

**When `results.tsv` reaches N data rows, STOP.** Do not start another
experiment. Report:

- your final (best) `competition_score` and the commit that produced it
- the seed model's score from evaluation 1
- how the budget was spent: counts of `keep` / `discard` / `invalid` /
  `crash` / `retrain`
- a short summary of what worked and what did not

Then wait for the human.
