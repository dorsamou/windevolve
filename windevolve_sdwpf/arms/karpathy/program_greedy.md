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
4. **Verify the fold layout exists**: confirm `$SDWPF_FOLDS_ROOT` is set and
   contains `F1/` through `F5/`, each with a `fold_train.csv` and a `windows/`
   directory. If it is missing or incomplete, stop and tell the human to
   generate it first — you cannot score anything without it:

   ```
   echo $SDWPF_FOLDS_ROOT
   ls $SDWPF_FOLDS_ROOT
   ```
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
    envs: dict with
      "path_to_test_x"  path to an infile CSV: 14 days of history for all 134
                        turbines, stacked turbine-major
      "path_to_train"   the training CSV for THIS FOLD. If you fit a model, fit
                        it on this and nothing else. It changes between folds.
      "fold"            the fold name, "F1".."F5". Use it to key any cached
                        artefact you keep between calls.
      optionally "capacity" (134) and "output_len" (288).
    returns: float array of shape (capacity, output_len, 1), predicted Patv in kW.
    """
```

The evaluator calls this once per validation window, then scores the
predictions using the same metric implementation the competition used.

**Scoring is five-fold cross-validation.** Five folds, each with its own
training period and its own 15 validation windows — 75 windows in total. Fold k
trains only on days before its validation block, so nothing you fit can have
seen the windows it is scored on:

```
F1   train days 1-95    validate 96-125
F2   train days 1-125   validate 126-155
F3   train days 1-155   validate 156-185
F4   train days 1-185   validate 186-215
F5   train days 1-215   validate 216-245
```

Your `competition_score` is the mean across the five folds. The evaluator also
reports `fold_scores` and `fold_std` — both matter, see "Reading the fold
breakdown" below.

A single contiguous validation period was tried first and abandoned. It was a
calmer, lower-wind stretch than the data as a whole, so programs tuned against
it learned preferences that did not transfer. Five folds spanning different
periods is the fix.

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
- Read, open, or reference any file under any fold's `windows/outfile/`. Those are the
  ground-truth targets for the validation windows. Your program receives an
  infile path and must forecast from history alone. Reading the corresponding
  outfile — or deriving its path from the infile path — is not forecasting, and
  invalidates the entire run. The evaluator reads the outfiles; you never do.
- Read `competition_score.py`. It scores against the locked test set and
  contains nothing relevant to your task.

**Training data**: if you fit a model, train it on `envs["path_to_train"]` —
that fold's own training CSV — and nothing else. Never on the full 245-day
file, and never on a path you hardcode: both would include validation periods
you are scored on, and that is leakage that makes the result meaningless.

**Cached artefacts must be keyed by fold.** If you fit a model lazily and cache
it, the cache key must include `envs["fold"]`. All five folds are scored in ONE
process, so a cache that ignores the fold will serve fold 1's weights for every
later fold — the numbers still print, and they are silently wrong:

```python
_CACHE = {}

def _get_model(envs):
    key = envs["fold"]                    # NOT a bare module-level global
    if key not in _CACHE:
        _CACHE[key] = _fit(envs["path_to_train"])
    return _CACHE[key]
```

This has already produced one invalid result in this project. Treat it as a
rule, not a suggestion.

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

- carve your own train/validation split out of any `fold_train.csv` and score
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
python ../../../../evaluation/evaluator.py forecaster.py \
    --folds-root $SDWPF_FOLDS_ROOT > run.log 2>&1
cp run.log logs/<id>.log        # same <id> you will log in results.tsv
```

`--folds-root` is what makes this five-fold. Without it you score a single old
split and the number is not comparable to anything else in `results.tsv`.

Redirect everything. Do NOT use `tee` or let output flood your context.

## Output format

On success the scorer prints:

```
program : forecaster.py
metric  : OFFICIAL competition grader (kddcup22-sdwpf-evaluation, verbatim)
folds   : .../evaluation/splits_rolling_5x30
  combined_score: -46.732145
  competition_score: 46.732145
  pct_capacity: 23.250
  pct_capacity_std: 2.410
  fold_std: 4.845
  fold_worst: 51.203
  n_folds: 5
  fold_scores:
    F1  49.881
    F2  50.114
    F3  42.377
    F4  51.203
    F5  40.086
  rmse: 52.104
  mae: 41.360
  latest_rmse: 48.221
  n_windows: 75
  eval_seconds: 96.4
  valid: True
```

Extract what you need with:

```
grep "competition_score:\|fold_std:\|valid:\|eval_seconds:\|reason:" run.log
sed -n '/fold_scores:/,/^  [a-z]/p' run.log     # the per-fold breakdown
```

### Reading the fold breakdown

`competition_score` is the mean over the five folds and is what you optimise.
`fold_scores` and `fold_std` tell you something the mean cannot.

Folds differ in intrinsic difficulty, and every forecaster finds the same folds
hard — F5 is the easiest for all five baselines, F3 the second easiest. So a
large `fold_std` is normal and is NOT evidence your program is unstable. On the
baselines it runs 4.4 to 11.2 MW.

What matters is how a change moves the folds RELATIVE to its parent:

- lower on all five folds → a real improvement, keep it
- lower on one or two and higher on the rest → probably noise, and the mean can
  still come out lower. Be sceptical. Say so in the description.
- much better on one fold only → usually means the change suits one period's
  weather rather than forecasting better. That is the exact failure this
  five-fold split exists to catch.

A mean improvement smaller than about 1 MW is inside the noise of this split.
Do not treat a 0.2 MW gain as established just because the number went down.

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

The TSV has a header row and 8 columns:

```
id	parent_id	competition_score	fold_std	fold_scores	eval_seconds	status	description
```

1. `id` — zero-padded 3-digit sequence number, starting at 001 for the seed
   baseline. This is also the name of the program's archive file.
2. `parent_id` — the id of the program this one was edited from (`000` for the
   seed)
3. `competition_score` in MW, the MEAN over the five folds (e.g. 46.732145) —
   use 999.999999 for crashes and for `valid: False`
4. `fold_std` in MW, to 6dp — use 999.999999 for crashes and invalids
5. `fold_scores` — the five per-fold scores, comma-separated, in F1..F5 order,
   each to 6dp. Use `-` for crashes and invalids. Never use tabs here.
6. `eval_seconds`, rounded to .1f — use 0.0 for crashes
7. status: see the loop section below for your arm's status values
8. short text description of what this experiment tried

Column 5 is the important addition. It is the only record of WHERE a program
won or lost, and it cannot be recovered later — the mean alone cannot tell you
whether a change helped everywhere or got lucky on one period.

Example:

```
id	parent_id	competition_score	fold_std	fold_scores	eval_seconds	status	description
001	000	44.537366	4.845102	49.881,50.114,42.377,51.203,40.086	96.4	scored	seed baseline unmodified
002	001	44.118203	4.771044	49.402,49.688,41.905,50.744,39.612	98.1	scored	clip negatives to zero; lower on all five folds
003	002	45.902118	6.113827	48.998,53.771,43.010,55.882,40.107	97.8	scored	trend extrapolation; better on F1 only, worse elsewhere
004	002	999.999999	999.999999	-	2.2	invalid	per-turbine constant, rejected by output check
005	002	999.999999	999.999999	-	0.0	crash	nan in output, ValueError in reshape
006	002	999.999999	999.999999	-	0.0	retrain	refit with hidden=96 into .cache/ckpt_006/
```

Row 003 is the pattern to watch for: the mean got worse, but even if it had not,
winning on one fold while losing on three is a period-specific artefact rather
than a forecasting gain. Note that in the description when you see it.

Also append to `FAILURE_MODES.md` whenever something fails in a way that
teaches you something — a check you did not anticipate, a constraint of the
task you had not noticed, an approach that cannot work for a structural reason.
One short paragraph per entry, newest at the bottom. This file is your memory
across the whole run; read it when you are stuck.

## The experiment loop

The experiment runs on your existing branch. You keep a single lineage: the
branch advances when an idea helps and rewinds when it does not.

LOOP until the budget is exhausted:

1. Look at the git state: the current branch/commit we're on
2. Tune `forecaster.py` with an experimental idea by directly hacking the code
3. `git commit`
4. Run the experiment:
   `python ../../../../evaluation/evaluator.py forecaster.py --folds-root $SDWPF_FOLDS_ROOT > run.log 2>&1`
5. Read out the results:
   `grep "competition_score:\|fold_std:\|valid:\|eval_seconds:\|reason:" run.log`
6. If the grep output is empty, the process died. Run `tail -n 50 run.log` to
   read the stack trace and attempt a fix. If you can't get it working after a
   few attempts, give up on that idea and move on.
7. Copy the scored program to `archive/<id>.py`, and record the result in
   `results.tsv` (do NOT commit `results.tsv` — leave it untracked, so a
   `git reset` does not erase your log). `parent_id` is the id of the program
   you edited from, which is always the current best.
8. If `competition_score` improved (lower), you "advance" the branch, keeping
   the git commit. Log status `keep`.
9. If it is equal or worse, `git reset` back to where you started. Log status
   `discard`.

Status values for this arm: `keep`, `discard`, `invalid`, `crash`, `retrain`.

You are a completely autonomous researcher trying things out. If they work,
keep. If they don't, discard. You advance the branch so that you can iterate.
If you feel like you're getting stuck, you can rewind, but do this very
sparingly (if ever).

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
whatever its status: `keep`, `discard`, `invalid`, `crash`, and `retrain` all
consume one unit. A retraining run counts even though it produces no score.
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