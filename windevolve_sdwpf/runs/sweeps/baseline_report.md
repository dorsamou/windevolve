# Baseline suite: analysis, mechanics, and parameter sweeps

**SDWPF / Baidu KDD Cup 2022 — `windevolve_sdwpf`**
Status: complete. All results are from the 197/48 validation split, 24 windows,
scored with the official grader. Regenerate the tables with
`python analysis/plots.py sweeps summary regimes`; §9 lists the open items.

---

## 1. What this document is for

Before the evolutionary search runs, the project needs three things it does not
yet have:

1. **A defensible baseline set.** Not the best forecasters available — the
   *reference points* against which a discovered program is judged. If the set
   has a hole, the search will appear to discover whatever fills it.
2. **A single number per baseline, on the same ruler the search will use.** The
   seed program handed to OpenEvolve and to the agentic loop needs an initial
   score that is directly comparable to every candidate score those loops
   produce. Otherwise "evolution beat the baseline" is not a measurable claim.
3. **Knowledge of how sensitive each baseline is to its own parameters.** A
   search that improves on an *untuned* baseline has demonstrated tuning, not
   discovery. The sweeps below establish how much of the available headroom is
   ordinary hyperparameter movement.

---

## 2. Is this the right set of baselines?

### 2.1 Coverage

Six entries, arranged on two axes. The first is the usual one, model class:

| Tier | Model | Class | Fitted? |
|---|---|---|---|
| 0 | Seasonal naive | copy a block of history forward | no |
| 0 | Climatology | average by time of day | no |
| 1 | ARMA(1,0,1) | linear, per turbine | yes, 134 fits |
| 1 | ARIMA(1,1,1) | linear with differencing | yes, 134 fits |
| 2 | XGBoost | nonlinear trees, one global model | yes, 1 fit |
| 3 | GRU | recurrent neural net, per turbine | yes, 134 fits |

That axis is complete for a baseline tier. It spans no-training through
deep-learning, and it includes both extremes of the sharing question the project
is actually about — XGBoost is a *single* model shared by all 134 turbines, the
GRU is *134 independent* models. Those two are the structural bracket around
"does an evolved single function generalise across turbines," which makes them
the two most important entries in the table, not the most accurate ones.

The second axis is the one that matters more here, and it is the one the score
table does not show: **what information does the model actually read?**

| Model | Own past power | Own wind speed | Time of day | Other channels | Other turbines | Future weather |
|---|---|---|---|---|---|---|
| Seasonal naive | yes | – | implicit (lag 144) | – | – | – |
| Climatology | yes | – | yes | – | – | – |
| ARMA / ARIMA | yes | – | – | – | – | – |
| XGBoost | yes | yes | yes | – | – | – |
| GRU | yes | yes | – | all 10 modelled | – | – |

Two columns are empty for every baseline, and they should be named in the
write-up rather than left to be noticed:

- **Cross-turbine information is deliberately absent.** The official GRU
  baseline is purely temporal even though the turbine location file ships with
  the dataset, and competition entries that built spatial machinery gained on
  the order of 6 %. Leaving it out of the baseline tier is the right call; the
  leaderboard models will cover it.
- **Future weather is absent because it does not exist in the dataset.** This
  is not an omission, it is the problem. It is also the explanation for the
  single most important observation in the whole table (§2.2).

### 2.2 The observation that justifies the set

Measured on the 197/48 validation split (§5 has the full sweeps):

| Model | Reference | Best tuned |
|---|---|---|
| ARMA(1,0,1) | 44.976 | **44.324** |
| GRU | 45.136 | **44.537** |
| XGBoost (global) | 46.139 | **45.445** |
| Climatology (diurnal) | 47.768 | **47.047** |
| Seasonal naive (lag 432) | 59.424 | 59.424 |
| ARIMA(1,1,1) | 56.631 | — (a point on the ARMA `d` axis) |

Read down that column and the answer to "is this a good spread" writes itself.
The gap from seasonal naive to climatology is 12 MW. The gap from climatology to
the best model is **2.7 MW**. And the top three — a univariate linear model, a
recurrent network, and a gradient-boosted tree — span **1.1 MW**, with ARMA and
the GRU 0.21 apart.

That flat region at ~44.5 MW is the **no-future-weather wall**. Every model class
converges to it, which is strong evidence that the binding constraint is
*information* rather than capacity: without knowing whether the wind will blow in
30 hours, a 48-hour forecast cannot do much better than a conditional average,
and every sufficiently flexible method finds the same conditional average.

Two results in §5 corroborate this from the other direction, and they are what
turn the observation into an argument:

- **Adding wind speed to XGBoost changes nothing** (`power` → `base`, +0.31 MW,
  inside the noise floor). Present wind speed is nearly uninformative about power
  two days out.
- **Adding nine more channels makes it 2 MW worse** (`base` → `turbine`).
  Direction, temperatures, pitch angles and reactive power at the origin do not
  help a distant forecast; they give an over-capacity model more to overfit.

**Answering the two questions directly.** How good can just averaging be? Within
2.7 MW of everything else, at zero training cost. How good can a tree model be?
*Worse* than a univariate linear model, despite reading nine extra channels. Both
answers argue for keeping the set as it is — and both are more informative than
they would have been had the models separated.

### 2.3 Two changes

**(a) Demote ARIMA(1,1,1) from a baseline to a sweep point.** ARMA(1,0,1) and
ARIMA(1,1,1) are the same model at two settings of `d`; they occupy two rows of
a six-row table for one family. An ADF test on SDWPF power rejects a unit root
decisively (p ≈ 1e-26), so `d=1` over-differences and should lose — and it does,
by 5.7 MW. That is a *result about the data*, and it reads far better as the
`d` axis of the ARMA sweep than as a redundant baseline. `analysis/sweep.py`
puts it there.

**(b) Add a shrinkage control — as an axis, not a model.** The official metric
rewards uniform down-scaling substantially: scaling persistence by 0.25 moves it
from 86.49 to 62.39 MW, because shrinking an uninformative forecast toward the
conditional mean is the correct response under MAE and RMSE. An evolutionary
search will find this within a handful of generations. If no baseline has been
shrinkage-optimised, that discovery will be scored against un-shrunk baselines
and credited as model discovery when it is one line of calibration.

So every grid in `analysis/sweep.py` carries a `scale` axis. It costs nothing —
no refitting, one extra multiply — and it produces the control the write-up
needs: *"the evolved champion beats the shrinkage-optimised baseline by X,"*
which is a claim that survives review.

### 2.4 Baseline or competitor? Both, in two columns

There is a real tension in "these are baselines, we don't need more complex
models" and "I want the most optimal parameter setup". They pull opposite ways,
and the resolution is not to pick one.

A baseline is a *reference point*: its value comes from being the thing everyone
recognises, so it must stay recognisable. A tuned model is a *competitor*: its
value comes from being hard to beat. The report needs both, because they answer
different questions:

| | Question it answers |
|---|---|
| Reference configuration | does evolution beat the *published, recognisable* method? |
| Best tuned configuration | does evolution beat a *competent practitioner* with the same data? |

The second is the harder and more interesting bar, and it is the one a reviewer
will ask about. So every table carries both columns and neither is dropped.

**One caution, and it is the reason to keep the columns separate.** Hand-tuning
a baseline is doing model discovery by hand — which is precisely what the search
is supposed to do. A day spent engineering XGBoost features against 60 search
iterations is not a budget-matched comparison, and the project's own budget
accounting is *number of evaluations*. So: the tuned column is reported with its
evaluation count beside it, and any claim of the form "the search beat XGBoost"
names which column it beat.

**Verdict:** the set is sufficient. No new model classes are needed before the
leaderboard models arrive. The additions above are a control and a
reclassification, not new baselines.

---

## 3. Which data the baselines are scored on

### 3.1 The decision

**All baseline scoring and all sweeping happens on the 197/48 train/validation
split. `final_phase_test` is not touched.**

(Terminology: this is a *train/validation* split, not "val/val" — the 197-day
file is the training data, the 48-day tail is the held-out validation period.)

Three reasons, in order of force:

1. **The seed program needs a score on the loop's own ruler.** OpenEvolve and
   the agentic loop both score candidates through `evaluation/evaluator.py`
   against the validation windows. A baseline number measured on
   `final_phase_test` is measured with a different ruler on different data, and
   comparing the two answers no question. The initial program's score and every
   candidate's score have to come from the same call.

2. **A sweep is model selection, and model selection on the test set consumes
   it.** This is the decisive argument. A sweep is 19–32 evaluations per model,
   with the best one kept. Run against `final_phase_test`, that is 19–32 rounds
   of feedback from windows that are supposed to have supplied zero. Competition
   teams got one submission. After a sweep on those windows, the resulting
   number is no longer comparable to the leaderboard, and the project's central
   comparability claim is gone — for a set of numbers that were never going to
   be the headline result.

3. **It is already the codified protocol.** `README.md` states that
   `final_phase_test` is read only by `competition_score.py`, once, at the end,
   on a selected champion. `evaluator.py` has no path to it. Scoring baselines
   there would require deliberately breaking the harness's own design.

### 3.2 Two things this obliges the write-up to say

- **Swept-best scores are selected on validation and are optimistically
  biased.** So the baseline table carries two columns: the **reference** score
  (the default configuration, unselected, the honest baseline number) and the
  **swept-best** score (selected on validation). The search arm is selected on
  validation too, so comparing swept-best against champion is the fair
  like-for-like; comparing champion against the *reference* column would flatter
  the search.

- **`final_phase_test` gets exactly one visit.** At the end of the project,
  score the baselines and every arm's champion in one pass. The
  validation-to-test movement is the generalisation gap, and it is a reportable
  result rather than something to hide. Some movement is expected by design: a
  champion is refit on the full 245 days at final scoring but on the truncated
  file during search.

### 3.3 The split itself

197 training days / 48 validation days, 24 windows. 48 days is 6 912 steps,
which is exactly 24 × 288, so at `--n-windows 24` the window starts are spaced
6 624/23 = 288.0 apart and the windows **tile** the validation period: every
ground-truth timestep is scored exactly once, none twice, none skipped. 49 days
would have left a ragged half-window. The window starts are evenly spaced with
no RNG, so the split is bit-reproducible from the arguments alone.

These are already the defaults in `evaluation/val_windows.py`.

---

## 4. The baselines, side by side

**Each baseline already has a `.md` beside its `.py`, and those remain the
reference for how it works.** This section does not repeat them. It gives the
one thing a per-model document cannot: all five on the same axes, so they can be
compared rather than read one at a time.

| | `baselines/naive/` | `baselines/classical/` (climatology) | `baselines/classical/` (arima) | `baselines/tree/` | `baselines/gru/` |
|---|---|---|---|---|---|
| **Fitted?** | no | no | yes, 134 models | yes, 1 model | yes, 134 models |
| **Trains on** | — | — | `val_train.csv`, `Patv` only | `val_train.csv`, all channels | `val_train.csv`, 10 channels |
| **Reads at predict** | window `Patv` | window `Patv` | last 432 steps of `Patv` | window history, all channels | last 144 steps × 10 channels |
| **Sees wind speed?** | no | no | no | yes | yes |
| **Sees other turbines?** | no | no | no | only in the `farm`/`neighbour` tiers | no |
| **Shared across turbines?** | n/a | n/a | no | **yes — one model** | no — 134 models |
| **Deterministic?** | yes | yes | yes | yes, given the seed | **no — dropout is live at inference** |
| **Cost of one config** | seconds | seconds | ~5–10 min | ~3–5 min | ~1 GPU-hour |
| **Full documentation** | `baseline_naive.md` | `baseline_climatology.md` | `baseline_arima.md` | `baseline_gbdt.md` | `gru_baseline.md` |

Every one satisfies the same contract, which is what makes them comparable:

```python
forecast(envs) -> np.ndarray, shape (134, 288, 1), active power in kW
```

`envs["path_to_test_x"]` is one window's infile — **270 144 rows**: 134 turbines
stacked turbine-major, 2 016 rows each (14 days at 10-minute resolution), 13
columns. The program reads it itself. `envs["path_to_train"]` is `val_train.csv`,
the 197-day truncated file, and is the **only** file a fitted baseline may fit
on. `forecast()` is called once per window (24 times), so anything expensive is
cached at module level on the first call.

### 4.1 The three that changed for this report

Parameters were added to three baselines so they could be swept. **In all three
cases the defaults reproduce the previous behaviour exactly, verified by
comparison rather than by inspection**, so no already-reported number moves.
Each file's `.md` has a "Parameters added for the sweep" section describing them.

| File | What was added | Default | Verified |
|---|---|---|---|
| `baseline_climatology.py` | `N_DAYS`, `AGG`, `HALFLIFE_DAYS`, `SMOOTH_STEPS` | all off / original | forecasts bit-identical to the original |
| `baseline_naive.py` | none — but the reference config moved from `LAG=144` to `LAG=432`, because `LAG=144` is rejected by the official zero-prediction guard (§5.3) | `LAG=432` | unchanged code |
| `baseline_gbdt.py` | six nested feature tiers, `k`, `objective` | `base` tier | `base` feature matrix bit-identical, same column order |
| `gru_baseline.py` | four deviation switches, `hidden`, `lr_decay`, `--fixed-loop` | all official | defaults untouched; the switches are opt-in |

### 4.2 The two GRUs

This one needs a decision recorded, because the file now supports two different
models.

The port reproduces the official code's behaviour **including four outright
defects**, each verified against Paddle 3.3.1 rather than assumed
(`gru_baseline.md` §6):

| # | What the official does | What a correct loop does | Measured effect |
|---|---|---|---|
| 3 | **Never zeroes gradients** — no `clear_grad()` anywhere, so the buffer accumulates across every batch *and* every epoch | `optimizer.zero_grad()` each step | gradient norms grew 1.00×, 2.00×, 3.00×, 4.00× over four batches |
| 4 | **Applies the optimizer twice per batch** — Paddle's `minimize()` updates, then the loop calls `step()` too | one `step()` | exactly 2.00× the weight movement |
| 1–2 | Dropout live at inference *and* during validation | `.eval()` for both | noisy predictions; noisy early-stopping signal |
| 8 | Val loader shuffled with `drop_last` — a different random subset discarded each epoch | no shuffling when only measuring | early stopping fires on noise |

By batch *n*, deviation 3 means the applied gradient is the sum of all *n*
preceding ones. That is not the optimizer anyone intends to run, and it is the
most likely reason an earlier, ordinary PyTorch implementation scored better
than this one.

**Decision: the corrected loop is the project's GRU baseline.** The purpose of
these baselines is to be a comparator for the search and a plausible seed
program, not to reproduce a published number — so a knowingly-broken training
loop is the wrong default. The official loop stays in the sweep as one point,
because it costs nothing (the existing checkpoints are reused) and it documents
the provenance: it is the row that connects this project to the published
47.850.

Two cautions, so the earlier-versus-current comparison is not overstated. More
than the loop changed between those measurements: the **scorer** moved from the
development `metrics.py` to the real submission grader, and those read about
4.5% apart on identical inputs — larger than the whole spread of the baseline
table. The **data** may have moved too, and a validation-tail number is not
comparable with a `final_phase_test` number (`gru_baseline.md` §7). The
`train_loop` axis settles it properly: both loops, one split, one grader.

## 5. Sweeps

### 5.1 Design

One harness, `analysis/sweep.py`, scores every model through the same call the
evolve loop uses (`evaluator._score_forecaster`), on the same 24 validation
windows. Five separate scripts would drift; a sweep is only interpretable if
every point in it was measured with an identical ruler.

Each model contributes a **grid** — a list of parameter points — and, if it
needs fitting, a **prepare** hook. The harness loads the baseline into a fresh
module object per point and sets module-level constants on it. No baseline
source is modified, and fitted artifacts are cached on a key derived from the
fit parameters, so the `scale` and `apply_steps` axes reuse a single fit.

Sweeps are **coordinate sweeps around a reference configuration**, not full
factorials. The GBDT grid as a factorial would be 576 fits; as a coordinate
sweep it is 32, and it answers the question this report asks — *which knobs move
the score at all* — rather than locating a global optimum that would in any case
be selected on validation.

**Monitoring is event-triggered.** After each point the harness appends its row
to `runs/sweeps/sweep_<model>.tsv` and fsyncs, prints a one-line event with the
score, the delta against the reference, and an ETA, and rewrites
`sweep_<model>.status.json`. `tail -f` on either shows progress the instant a
point lands, and a killed sweep loses at most the point in flight. The reference
configuration is always run first, so every delta is anchored to the same
number.

### 5.2 The noise floor comes first

Where a model has a seed, the grid opens with a `seed` axis: three or five
repeats of the *identical* configuration, differing only in the RNG. **No other
difference in any sweep below is meaningful unless it exceeds that spread.** The
entire top ten of the original KDD Cup spanned about 0.4 MW, so an unmeasured
noise floor can easily be larger than the effect being studied.

- **XGBoost:** the seed governs origin sampling and tree construction.
- **GRU:** dropout is live at inference, so even *scoring fixed checkpoints* is
  stochastic. Five inference seeds; this is the queued noise-floor experiment.
- **Naive, climatology, ARMA:** deterministic given the split. Any difference is
  real — though still specific to this split, which is a separate caveat.

`analysis/plots.py sweeps` draws the measured floor as a grey band across every
sweep panel, so a curve cannot be over-read.

### 5.3 Results

All scores are `0.5*(RMSE+MAE)` in MW on the 197/48 validation split, 24 windows,
official grader. Lower is better. **The only measured noise floor in the whole
suite is XGBoost's seed axis at ±0.396 MW (1 sd);** it is the yardstick for every
claim below, including for the deterministic models, whose differences are exact
on this split but still specific to it.

---

#### Seasonal naive — `LAG`, `scale`

| axis | setting | score | RMSE | MAE | valid |
|---|---|---|---|---|---|
| lag | 2 | — | — | — | 76.87% turbines with zero predicted values |
| lag | 6 | — | — | — | 54.48% turbines with zero predicted values |
| lag | 18 | — | — | — | 100.00% turbines with identical predicted values |
| lag | 36 | — | — | — | 19.40% turbines with zero predicted values |
| lag | 72 | — | — | — | 19.40% turbines with zero predicted values |
| lag | **144** | — | — | — | **66.42% turbines with zero predicted values** |
| lag | 288 | 68.653 | 75.380 | 61.926 | yes |
| lag | **432** | **59.424** | 66.321 | 52.527 | yes |
| lag | 1008 | 63.115 | 70.587 | 55.642 | yes |
| lag | 2016 | 62.760 | 68.957 | 56.564 | yes |
| scale | all nine, at lag=144 | — | — | — | rejected, as lag=144 itself is |

**Daily seasonal naive is unscoreable on this dataset.** `lag=144` — the textbook
"tomorrow looks like today" — is rejected by the official
`is_zero_prediction` guard: copying the last 24 hours forward gives an all-zero
forecast for any turbine that was parked that day, and 66% of turbines were. The
regime split below puts 53% of all turbine-windows in the parked bin, so this is
not an unlucky window, it is the shape of the farm.

Six of ten lags fail this way, and the failure mode is informative: short lags
(2, 6) fail immediately at horizon 0; `lag=18` fails the *identical*-prediction
guard instead, because a 3-hour block tiled 16 times is nearly flat. Only lags of
two days or more survive, and they survive by being long enough to contain some
non-zero history.

The surviving curve is **non-monotonic** — 288 → 68.65, 432 → 59.42, 1008 →
63.11, 2016 → 62.76 — which should not be read as an optimal memory length. A
longer block is more likely to dodge the zero guard *and* more likely to average
over the diurnal cycle; the 432 minimum is those two effects crossing, not a
statement about wind.

**Reference configuration changed to `lag=432`.** `lag=144` cannot produce a
score, so it cannot be the reference. This is a finding about the metric, not a
defect in the baseline, and it belongs in the write-up: the competition's own
guards make the most standard naive forecaster inadmissible on a heavily
curtailed farm.

---

#### Climatology — `N_DAYS`, `HALFLIFE_DAYS`, `AGG`, `scale`

| axis | setting | score | RMSE | MAE |
|---|---|---|---|---|
| n_days | 1 | — | — | rejected: 66.42% zero predictions |
| n_days | 2 | 62.538 | 68.025 | 57.051 |
| n_days | 3 | 54.079 | 59.646 | 48.512 |
| n_days | 5 | 50.553 | 55.855 | 45.251 |
| n_days | 7 | 49.148 | 54.444 | 43.853 |
| n_days | **10** | **47.047** | 52.064 | 42.031 |
| n_days | 14 (reference) | 47.768 | 52.665 | 42.871 |
| halflife | 0.5 d | 64.147 | 70.052 | 58.242 |
| halflife | 1 d | 56.687 | 62.191 | 51.182 |
| halflife | 2 d | 51.134 | 56.444 | 45.823 |
| halflife | 3 d | 49.345 | 54.549 | 44.141 |
| halflife | 5 d | 48.248 | 53.327 | 43.170 |
| halflife | 10 d | 47.790 | 52.768 | 42.813 |
| agg | median, n_days=3 | 57.218 | 63.624 | 50.813 |
| agg | median, n_days=7 | 52.092 | 57.932 | 46.252 |
| agg | median, n_days=14 | 50.455 | 56.177 | 44.732 |
| scale | 1.0 | 47.768 | 52.665 | 42.871 |
| scale | 0.9 | 47.167 | 52.221 | 42.114 |
| scale | **0.8** | **46.940** | 52.163 | 41.717 |
| scale | 0.7 | 47.130 | 52.533 | 41.726 |
| scale | 0.6 → 0.2 | 47.787 → 55.708 | | rises monotonically |

**More history is uniformly better, and both axes agree.** `n_days` improves
monotonically from 2 to 10 (62.54 → 47.05), and the half-life curve traces the
same shape from 0.5 days to 10 (64.15 → 47.79). The 14-day point is 0.72 worse
than 10, which is within about 2σ of the XGBoost floor and should not be read as
a real optimum.

The conclusion is the important part: **there is no evidence the recent weather
regime carries extra information at a 48-hour horizon.** The long-run daily shape
wins, and both curves are still improving at the edge of what is available.
Climatology only sees the window's own 14 days, so this is a *bound*, not a
measurement — whether 30 days would be better is untestable with this model as
written, and worth saying rather than implying.

`n_days=1` is rejected for the same reason `lag=144` is: one day of history from
a parked turbine is all zeros.

**The median loses everywhere**, by 3.1 MW at n_days=3 and 2.7 at n_days=14. The
reasoning that motivated the axis was that parked days drag a mean profile down
while leaving a median untouched. That mechanism is real — the mean *is* biased
low — but the sign of its effect on the score is the opposite of what was
expected. The next subsection explains why, and the two results are the same
finding seen twice.

---

#### The shrinkage control — why a constant multiplier is worth as much as a sweep

Every model improves at `scale = 0.8`, by 0.46 to 0.83 MW. Decomposing the
metric shows exactly what is being bought and sold:

| Model | RMSE 1.0 → 0.8 | MAE 1.0 → 0.8 | Score change |
|---|---|---|---|
| ARMA | 48.853 → 48.980 (**+0.127**) | 41.099 → 40.053 (−1.046) | −0.459 |
| GRU | 49.038 → 49.122 (**+0.084**) | 41.233 → 40.163 (−1.070) | −0.493 |
| XGBoost | 50.448 → 50.156 (−0.292) | 41.830 → 40.897 (−0.933) | −0.613 |
| Climatology | 52.665 → 52.163 (−0.502) | 42.871 → 41.717 (−1.154) | −0.828 |

**Shrinkage is MAE optimisation, paid for in RMSE.** MAE is minimised by the
conditional *median*, and SDWPF power is heavily zero-inflated, so the median
sits well below the mean. Scaling every prediction down moves it toward the
median. For the two best models the RMSE actually gets *worse*; the score
improves anyway because the metric averages the two halves and the MAE gain is
roughly ten times the RMSE loss.

This also explains the climatology median result. A median-over-days profile is
*higher* in level than the parked-day-depressed mean profile, so it moves away
from the quantity MAE rewards.

The practical consequence for this project: **a one-line multiplication is worth
as much as an entire hyperparameter sweep on every baseline.** An evolutionary
search will find it within a handful of generations, and it is not model
discovery — it is an exploitation of the fact that the scoring metric is half
MAE. Reporting the shrinkage-optimised baseline alongside the champion is what
keeps that credit in the right place.

---

#### ARMA / ARIMA — `order`, `fit_days`, `APPLY_STEPS`, `scale`

| axis | setting | score | RMSE | MAE |
|---|---|---|---|---|
| order | (1,0,0) | 45.063 | 48.945 | 41.181 |
| order | (1,0,1) — reference | 44.976 | 48.853 | 41.099 |
| order | (2,0,1) | 45.537 | 49.499 | 41.575 |
| order | **(2,0,2)** | **44.883** | 48.751 | 41.015 |
| order | (1,1,0) | — | — | rejected: 92.54% identical predictions |
| order | **(1,1,1)** | **56.631** | 61.433 | 51.829 |
| fit_days | **30** | **44.324** | 48.705 | 39.943 |
| fit_days | 90 | 44.681 | 48.670 | 40.691 |
| fit_days | 197 | 44.976 | 48.853 | 41.099 |
| apply_steps | 144 / 288 / 432 / 1008 / 2016 | **44.9758 for all five** | 48.853 | 41.099 |
| scale | 0.9 | 44.561 | 48.712 | 40.410 |
| scale | **0.8** | **44.517** | 48.980 | 40.053 |
| scale | 0.7 → 0.2 | 44.880 → 54.814 | | rises monotonically |

Three findings, all clean.

**1. `d=1` is catastrophic, and the ADF test called it.** ARIMA(1,1,1) scores
56.63 against ARMA(1,0,1)'s 44.98 — **11.7 MW worse**, roughly 30σ. ARIMA(1,1,0)
is rejected outright, with 92.5% of turbines producing almost-identical
predictions: differencing an already-stationary series leaves something close to
a random walk, whose multi-step forecast is a flat line. Demoting ARIMA(1,1,1)
from a headline baseline to a point on this axis was the right call — it is a
result about the data, not a model worth reporting separately.

**2. Less training history is better.** 30 days beats 197 by 0.65 MW,
monotonically. Set this beside the climatology result, where *more* history won
monotonically, and the pair is genuinely informative: **the daily level profile
is stable and wants long history, while the dynamics are non-stationary and want
recent history.** Two different quantities estimated from the same data, opposite
answers. This is the most interesting thing in the classical sweeps and it is
worth a sentence in the abstract.

**3. `APPLY_STEPS` does nothing whatsoever** — all five settings return
identically 44.9758 to four decimals. That exact equality is the giveaway that
it is structural rather than coincidental: with coefficients held fixed, a
stationary ARMA's Kalman filter converges to steady state within a few dozen
steps, so anything before that washes out entirely. **Anything above ~144 steps
is wasted compute**, a free speedup with no score cost.

Order barely matters. (2,0,2) beats (1,0,1) by 0.09 MW — well inside noise — so
AR(1) captures essentially all of the available autocorrelation.

---

#### XGBoost — `seed`, `features`, `trees`, `depth`, `lr`, `objective`, `scale`

**Noise floor first.** Three fits of the identical configuration:

| seed | 42 | 43 | 44 | mean | 1 sd |
|---|---|---|---|---|---|
| score | 46.139 | 46.922 | 46.426 | 46.496 | **0.396** |

A spread of 0.78 MW between two identical configurations. **This is larger than
almost every effect measured anywhere in this report**, and it is the number
against which everything else must be read.

| axis | setting | score | RMSE | MAE |
|---|---|---|---|---|
| features | power (9 feats) | **45.834** | 50.155 | 41.512 |
| features | base (12) — reference | 46.139 | 50.448 | 41.830 |
| features | turbine (22) | 48.151 | 52.550 | 43.753 |
| features | farm (26) | 47.582 | 51.957 | 43.207 |
| features | neighbour (29) | 47.789 | 52.103 | 43.474 |
| features | xy (31) | 47.609 | 51.916 | 43.303 |
| trees | **200** | **45.445** | 49.692 | 41.198 |
| trees | 400 | 46.139 | 50.448 | 41.830 |
| trees | 800 | 46.683 | 51.039 | 42.327 |
| depth | **4** | **45.512** | 49.724 | 41.300 |
| depth | 6 | 46.139 | 50.448 | 41.830 |
| depth | 9 | 47.180 | 51.645 | 42.714 |
| lr | **0.03** | **45.759** | 49.966 | 41.552 |
| lr | 0.05 | 46.139 | 50.448 | 41.830 |
| lr | 0.1 | 47.231 | 51.603 | 42.858 |
| objective | squarederror | 46.139 | 50.448 | 41.830 |
| objective | absoluteerror | 49.104 | 54.243 | 43.965 |
| scale | **0.8** | **45.526** | 50.156 | 40.897 |

**Every capacity axis says the model is overfitting.** Trees, depth and learning
rate are each monotonic in the direction of *less* model, each at 3–4σ, and
**each optimum sits at the edge of the grid.** The sweep did not go low enough.
The obvious follow-up is cheap CPU work: `trees` 50/100, `depth` 2/3, `lr`
0.01/0.02.

**The feature ladder goes the wrong way, and that is the headline.**

| step | Δ score | vs 0.396 floor | reading |
|---|---|---|---|
| power → base | +0.31 | within noise | **wind speed buys nothing measurable** |
| base → turbine | **+2.01** | ~5σ | the extra channels actively hurt, badly |
| turbine → farm | −0.57 | ~1.4σ | marginal recovery |
| farm → neighbour | +0.21 | within noise | **no evidence the layout helps** |
| neighbour → xy | −0.18 | within noise | **turbine identity buys nothing** |

*Wind speed.* Adding `wspd_last`, `wspd_mean_6h`, `wspd_mean_24h` changes nothing
detectable. This is consistent rather than surprising: the model has wind speed
at the *origin*, and a 48-hour-ahead target is nearly independent of it. Without
future weather, present wind speed is close to useless. This is the
no-future-weather wall showing up as a feature ablation.

*The `turbine` tier.* A 2.01 MW penalty, the largest single effect in the sweep.
Two mechanisms, both consistent with the capacity results: the added features are
origin-time snapshots that stay constant across all 288 horizons, so they let the
model fit the origin state when the distant target does not depend on it; and
`colsample_bytree=0.8` now dilutes the few informative features among 22. The
hypothesis that pitch angle would identify parked turbines was reasonable and did
not survive. Worth retesting at the *tuned* capacity (`trees=200, depth=4`)
before concluding the features themselves are worthless rather than the
combination.

*The spatial tiers.* Neither farm aggregates nor nearest-neighbour features
produce a detectable gain. There is no evidence here that turbine layout carries
information the temporal features do not already have.

*`xy` — the result that matters most for this project.* Giving the global model
each turbine's unique coordinates is the strongest possible invitation to
memorise per-turbine behaviour, and it produces a change of −0.18 MW, well inside
the noise floor. **There is little turbine-specific structure to exploit.** That
is an early, direct answer to the project's central question, and it points away
from needing an instance-specific portfolio. It must be caveated exactly as
measured: at this hyperparameter setting, within a ±0.396 MW floor, on these 24
windows, with a *global* model rather than 134 private ones.

**`objective=absoluteerror` is 2.97 MW worse.** Fitting the conditional median
damages RMSE (50.45 → 54.24) far more than it helps MAE (41.83 → 43.97 — it
does not help it at all here). Note this is the opposite of what the `scale` axis
does, even though both move predictions toward the median: shrinkage applies a
single global multiplier at prediction time, while an absolute-error objective
changes what every tree fits. The metric rewards the first and punishes the
second.

---

#### GRU — `capacity`, `input_len`, `optim`, `budget`, `provenance`, `scale`

| axis | setting | score | RMSE | MAE |
|---|---|---|---|---|
| capacity | hidden=48, layers=2 — reference | 45.136 | 49.038 | 41.233 |
| capacity | hidden=96 | 44.833 | 48.763 | 40.902 |
| capacity | **hidden=128** | **44.745** | 48.683 | 40.806 |
| capacity | layers=1 | 45.118 | 48.999 | 41.238 |
| capacity | hidden=96, layers=1 | 44.927 | 48.853 | 41.000 |
| input_len | 288 | 45.117 | 49.042 | 41.192 |
| input_len | 432 | 45.093 | 49.021 | 41.165 |
| optim | lr=3e-4 | 44.642 | 48.576 | 40.708 |
| optim | lr=1e-3 | 44.667 | 48.666 | 40.667 |
| optim | **lr=1e-4 flat (no decay)** | **44.604** | 48.558 | 40.649 |
| optim | lr=1e-3, decay=0.9 | 44.778 | 48.807 | 40.749 |
| budget | 20 epochs, patience 5 | 44.597 | 48.552 | 40.642 |
| budget | **40 epochs, patience 8** | **44.537** | 48.499 | 40.576 |
| provenance | **official loop, all four defects** | **44.802** | 48.818 | 40.786 |
| scale | **0.8** | 44.643 | 49.122 | 40.163 |

**`input_len` does nothing.** 144 → 45.136, 288 → 45.117, 432 → 45.093. This was
predicted to be the most consequential axis in the sweep — the model reads 24
hours to forecast 48 — and it is the *smallest* effect measured. Giving the
network three days of history instead of one changes the score by 0.04 MW.
Prediction wrong, and instructively so: the limit is not how much past the model
can see.

**Capacity helps, modestly, and is still improving at the grid edge.** 48 → 96 →
128 gives 45.136 → 44.833 → 44.745, monotonic and roughly 1σ of the XGBoost
floor. Note this runs the opposite way to XGBoost, which wanted *less* capacity
on every axis.

**The optimiser was the actual problem.** Every `optim` point beats the reference
by about 0.5 MW, and the best of them is `lr=1e-4 flat` — the same learning rate
as the default with the *decay switched off*. So the issue was never the rate; it
was the schedule. `budget` confirms it: 40 epochs with patience 8 gives the best
GRU score in the sweep, 44.537. **The model was stopping before it had
converged.**

**The finding worth the whole axis: the official broken loop scores 44.802,
better than the corrected default at 45.136.**

This is not an argument for keeping the defects. It is an explanation of them.
Never zeroing gradients accumulates a running sum that behaves like momentum, and
applying the optimizer twice per batch doubles the effective step size. Both
compensate for a learning rate that decays to roughly 1/500th of its starting
value by epoch 10. **The defects were an accidental workaround for an
under-tuned optimiser.** Fix the schedule honestly and the corrected loop reaches
44.537, beating the broken one by 0.27 — and this is only visible because both
were measured on the same split with the same grader.

**The GRU has no measured noise floor.** Inference is deterministic now, so
repeated scoring gives identical numbers, but *training* is not: initialisation,
batch order, and dropout masks all vary with the seed. Its 0.6 MW headroom is the
same size as XGBoost's noise. **Three retrains of the reference configuration at
different seeds are needed before any GRU difference below ~0.5 MW can be
claimed** — about 30 minutes of GPU, and the most valuable half-hour remaining in
this study.

### 5.4 Cross-model summary

Regenerate with `python analysis/plots.py summary`, which writes
`runs/plots/summary_table.md` directly from the sweep TSVs so this table can
never disagree with the results it summarises.

| Model | Reference config | Reference | Best tuned config | Best tuned | Tuning headroom | Best shrinkage | Shrinkage gain | Noise floor (1 sd) |
|---|---|---|---|---|---|---|---|---|
| ARMA | `order=(1,0,1)` | 44.976 | `fit_days=30` | **44.324** | 0.652 | `scale=0.8` → 44.517 | 0.459 | deterministic |
| GRU | `hidden=48, layers=2` | 45.136 | `40 epochs, pat=8` | **44.537** | 0.598 | `scale=0.8` → 44.643 | 0.493 | **not measured** |
| XGBoost | `seed=42`, base features | 46.139 | `trees=200` | **45.445** | 0.694 | `scale=0.8` → 45.526 | 0.613 | **0.396** |
| Climatology | `N_DAYS=14, mean` | 47.768 | `n_days=10` | **47.047** | 0.721 | `scale=0.8` → 46.940 | 0.828 | deterministic |
| Seasonal naive | `LAG=432` | 59.424 | `lag=432` | 59.424 | 0.000 | not scoreable | — | deterministic |

102 grid points scored; 17 rejected by the official guards.

#### Reading this table

**1. The tuning headroom is not distinguishable from noise.** Every value in that
column — 0.652, 0.598, 0.694, 0.721 — sits within about 1.5–2σ of the one
measured noise floor. Worse, "best tuned" is a *minimum over roughly ten points*,
and taking the smallest of ten draws from a distribution with sd 0.396
manufactures an apparent gain of that order **even if every configuration were
identical**. The observed headroom is almost exactly the size that selection
alone would produce.

The deterministic models are a partial exception: naive, climatology and ARMA
give exact, repeatable numbers, so their differences are real *on these 24
windows*. That is not the same as generalising, and the write-up should say so.

The consequence for the project is direct: **a search that returns 0.5 MW better
than a baseline has demonstrated nothing.** The bar is higher than the headroom
column suggests, and this is the single most important number to carry into the
arm comparison.

**2. Shrinkage is worth as much as all the tuning, on every model.** The
shrinkage gain column (0.46–0.83) matches the tuning headroom column
(0.60–0.72) model for model. One multiplication is worth an entire sweep — and it
is metric calibration, not forecasting. See the shrinkage subsection above for
the RMSE/MAE decomposition that explains why.

**3. The ordering changed from the earlier measurements, and ARMA now leads.**
ARMA 44.98, GRU 45.14, XGBoost 46.14 at reference; ARMA 44.32, GRU 44.54,
XGBoost 45.45 tuned. A univariate linear model reading nothing but past power
matches a tuned neural network and beats a gradient-boosted tree that reads nine
additional channels.

**4. These numbers are NOT comparable with the pre-split measurements.** Every
model scores better here than in the earlier round despite having *less* training
data (197 days rather than 245). The parsimonious explanation is that these
validation windows are easier than `final_phase_test`, not that the models
improved. The two sets must not be differenced anywhere in the write-up. What
*is* comparable is the internal ordering, and it moved.

#### The wall

The earlier round put ARMA, XGBoost and the GRU within about 1 MW of each other
at ~47 MW. On the clean split the wall is still there and has moved down to
**~44.5 MW**: ARMA and the GRU are 0.21 apart, with XGBoost 0.91 behind.

That a univariate linear model on power alone is competitive with a tuned
recurrent network — and better than a tree with access to wind speed, direction,
temperatures, pitch angles and neighbouring turbines — is a strong statement that
**the binding constraint is information, not model capacity.** The XGBoost
feature ladder says the same thing from the other direction: adding nine channels
of data made it worse, not better.

#### Follow-ups this table implies

1. **Extend the XGBoost sweep downward.** `trees` 50/100, `depth` 2/3, `lr`
   0.01/0.02. Every capacity optimum is at the grid edge. Cheap CPU work.
2. **Measure the GRU training noise floor.** Three retrains of the reference
   configuration at different seeds, ~30 minutes of GPU. Without it, half the GRU
   column cannot be read.
3. **Retest the `turbine` feature tier at tuned capacity.** The 2.01 MW penalty
   may be an artefact of over-capacity rather than of the features themselves.
4. **Report `lag=432`, not `lag=144`, as the naive baseline,** and state why.

## 6. Figures

Generated by `analysis/plots.py`, which caches each model's raw forecasts once
and draws every figure from that single set. This matters for the GRU: caching
means the GRU curve in the per-turbine plot and the GRU curve in the horizon
plot are the *same* forecast, not two draws from a stochastic model.

| Figure | File | What it shows |
|---|---|---|
| Single-panel prediction | `pred_single.png` | one turbine, one window, all models against truth |
| Prediction grid | `pred_grid.png` | turbines down the rows × windows across the columns |
| Farm aggregate | `pred_farm.png` | all 134 turbines summed, per window |
| Error vs. horizon | `error_vs_horizon.png` | mean per-turbine \|error\| against horizon step |
| Sweep curves | `sweep_<model>.png` | score against each parameter, with the noise-floor band |
| Regime error | `regime_error.png` | error vs horizon, one panel per origin-power regime |
| Regime winner | `regime_winner.png` | best model per (regime, horizon) cell, with its margin |
| Regime decay | `regime_decay.png` | mean forecast trajectory from each starting regime |

Three notes on reading them.

**The mask is drawn, not hidden.** The official metric excludes ground-truth
steps that are missing or invalid — NaN anywhere in the row, negative power, a
turbine at zero power while the wind exceeds 2.5 m/s, a pitch angle above 89°,
an out-of-range direction sensor. Those stretches are shaded grey in every
prediction plot. A model can look badly wrong across a stretch that was never
scored, and a reader who does not know which stretches those are will draw the
wrong conclusion from the picture.

**The farm-aggregate plot is the evidence; single-turbine panels are
illustration.** The official score sums per-turbine errors across all 134
turbines, so farm total is the quantity being forecast. Any single turbine can
be cherry-picked. Use the grid (multiple turbines × multiple windows) when a
per-turbine view is wanted, and say which turbines and why.

**Error-vs-horizon is what explains a sweep.** Two configurations with the same
total score can fail in completely different places, and a change that improves
the first six hours while wrecking the last twelve shows up as no change at all
in the aggregate. When a sweep axis moves the score, draw this figure at both
ends of the axis and the mechanism is usually visible immediately.

---

## 6b. Regime analysis: where does each model actually win?

A single headline score can hide the possibility that two models with nearly
identical totals are winning in completely different places. Given that ARMA,
XGBoost and the GRU all land within about a megawatt of each other, that is not
a hypothetical concern here -- it is the most likely explanation for the flat
region at ~47 MW.

The literature suggests a specific shape for this. A neural net's short-horizon
forecast is reported to decay too steeply when the turbine is currently at very
high or very low power, with tree models handling those regimes better. The
mechanism is plausible: a network trained on mostly mid-range states learns an
average reversion rate and applies it everywhere, while a tree can split the
power axis and learn a different rate in each region.

`python analysis/plots.py regimes` tests this directly. Every turbine-window is
assigned to a regime by its power at the forecast origin -- the mean of the last
hour of input, rather than a single 10-minute reading, because one sample is
noisy and occasionally a dropout:

| Regime | Origin power |
|---|---|
| parked | < 30 kW |
| low | 30-300 kW |
| mid | 300-800 kW |
| high | 800-1200 kW |
| near rated | > 1200 kW |

Bins are physical rather than quantile-based, so the boundaries mean the same
thing in every run and two reports stay comparable.

### The three outputs

**`regime_error.png`** -- error against horizon, one panel per regime. If the
claim holds, the GRU and XGBoost curves will cross, and cross at *different*
horizons in different panels. Curves are smoothed with a one-hour moving average
(`--smooth 0` to disable); the tables are never smoothed.

**`regime_winner.png`** -- which model wins each (regime, horizon) cell, and its
margin over the runner-up. **Read the margin, not the colour.** A cell won by
0.001 MW is a tie, and the noise floor from the XGBoost seed axis is the right
yardstick for deciding which margins are real.

**`regime_decay.png`** -- the mean forecast *trajectory* from each starting
regime, with the actual overlaid. This is the direct test of the steepness
claim: a model that reverts toward the middle faster than reality does will pull
away from the black line in the first few hours. Note this asks a different
question from the error plot -- a model can have a good mean absolute error while
systematically mis-shaping its trajectory, because over- and under-shoots cancel
in the mean prediction but not in the error.

`regime_tables.md` carries the numbers, including a **crossover table** (the
horizon at which the leader at h=1 is overtaken, per regime) and a
**trajectory-bias table** (mean predicted minus mean actual over 0-6h; negative
means decaying too fast, positive means holding on too long).

### Results

Regime membership, over 402 turbine-windows (3 windows x 134 turbines):

| Regime | Origin power | n | share |
|---|---|---|---|
| parked | < 30 kW | 213 | 53.0% |
| low | 30–300 | 50 | 12.4% |
| mid | 300–800 | **13** | **3.2%** |
| high | 800–1200 | 54 | 13.4% |
| near rated | > 1200 | 72 | 17.9% |

**The farm is bimodal.** Turbines are almost always either off or running hard;
only 3.2% of turbine-windows start in the middle of the power curve. Two
consequences follow immediately. The `mid` panel has n=13 and **nothing in it
should be reported as a result**. And the farm-level score is dominated by how
well models handle *parked* turbines — which is a question about curtailment and
maintenance, not about wind.

#### Mean |error| (MW) by regime and horizon

| regime | model | 0–6h | 6–12h | 12–24h | 24–36h | 36–48h |
|---|---|---|---|---|---|---|
| parked (n=213) | Seasonal naive | **0.202** | 0.287 | 0.726 | 0.749 | 0.440 |
| | Climatology | 0.336 | 0.348 | 0.668 | 0.640 | 0.346 |
| | ARMA | 0.285 | **0.237** | 0.624 | **0.586** | 0.314 |
| | XGBoost | 0.384 | 0.285 | 0.743 | 0.673 | **0.306** |
| | GRU | 0.273 | **0.236** | **0.638** | 0.607 | 0.318 |
| low (n=50) | Seasonal naive | **0.220** | 0.384 | 0.746 | 0.675 | 0.362 |
| | Climatology | 0.508 | 0.503 | 0.714 | 0.768 | **0.293** |
| | ARMA | 0.418 | **0.356** | **0.661** | **0.653** | **0.265** |
| | XGBoost | 0.493 | 0.462 | 0.817 | 0.762 | 0.293 |
| | GRU | 0.427 | 0.368 | 0.680 | 0.672 | 0.269 |
| mid (n=13) | Seasonal naive | 0.308 | 0.425 | 0.749 | **0.644** | 0.352 |
| | Climatology | 0.575 | 0.710 | 0.653 | 0.886 | 0.398 |
| | ARMA | 0.358 | 0.456 | 0.611 | 0.810 | **0.317** |
| | XGBoost | 0.410 | 0.463 | 0.760 | 0.952 | 0.347 |
| | GRU | 0.360 | **0.399** | **0.636** | 0.842 | 0.325 |
| high (n=54) | Seasonal naive | 0.902 | 0.830 | 0.683 | 0.197 | 0.500 |
| | Climatology | 1.092 | 1.143 | **0.307** | **0.057** | 0.380 |
| | ARMA | 0.490 | 0.786 | 0.310 | 0.187 | 0.290 |
| | XGBoost | **0.310** | **0.542** | 0.438 | 0.313 | **0.259** |
| | GRU | 0.396 | **0.542** | 0.354 | 0.243 | 0.281 |
| near rated (n=72) | Seasonal naive | 1.064 | 0.918 | 0.816 | 0.239 | 0.583 |
| | Climatology | 1.298 | 1.292 | **0.333** | **0.049** | 0.373 |
| | ARMA | 0.497 | 0.867 | 0.335 | 0.204 | 0.284 |
| | XGBoost | **0.387** | 0.750 | 0.421 | 0.219 | **0.261** |
| | GRU | 0.512 | **0.612** | 0.407 | 0.280 | 0.271 |

#### The steepness claim: half confirmed

**At high power and short horizon, the tree does take over.**

| 0–6h | high (800–1200) | near rated (>1200) |
|---|---|---|
| XGBoost | **0.310** | **0.387** |
| GRU | 0.396 | 0.512 |
| ARMA | 0.490 | 0.497 |
| Climatology | 1.092 | 1.298 |

XGBoost beats the GRU by 0.086 and 0.125 MW in the two high-power regimes at
short horizon. That is the effect the literature describes, and it is the only
place in the whole table where XGBoost is the clear winner.

**But the low extreme inverts it.** In the parked regime at 0–6h, XGBoost is the
*worst* of the five (0.384) and seasonal naive is the best (0.202) — copying
zeros forward is exactly right for a turbine that is off and staying off. So the
tree takes over at the top of the power curve, not at both ends. The claim as
stated does not fully transfer to this dataset.

The crossover table shows how quickly the advantage evaporates: outside the high
regime, the leader at h=1 is overtaken within an hour or two. Only in the `high`
regime does XGBoost hold its lead for a full 4 hours.

#### Every model under-predicts, everywhere

The trajectory-bias table (mean predicted minus mean actual over 0–6h, kW) is
**negative in all 25 cells**:

| regime | Naive | Climatology | ARMA | XGBoost | GRU |
|---|---|---|---|---|---|
| parked | −33 | −286 | −267 | −376 | −257 |
| low | −72 | −425 | −397 | −480 | −411 |
| mid | −157 | −568 | −294 | −178 | −243 |
| high | −902 | −1092 | −485 | −272 | −381 |
| near rated | −1064 | −1298 | −488 | −368 | −512 |

Every model reverts toward the mean faster than reality does, and the effect is
enormous at high power — climatology under-predicts by 1.3 MW per turbine in the
first six hours after a near-rated start, because it has no mechanism to
represent "this turbine is currently running hard" at all.

There is an apparent contradiction with the shrinkage result: if every model
already under-predicts, why does multiplying by 0.8 help? Because the two
statements are about different parts of the distribution. The bias above is a
*conditional mean* on high-power starts, which are 31% of turbine-windows; the
score is dominated by the zero-inflated bulk, where MAE rewards predicting low.
Shrinkage makes the high-power bias worse and the overall score better. **This is
the clearest evidence in the report that the metric and forecast quality are not
the same thing**, and it belongs in the write-up next to the shrinkage
decomposition.

#### Caveats

- `mid` has n=13. Do not report anything from that row.
- Climatology's 0.049–0.057 MW at high power, 24–36h is almost certainly a single
  weather event that happened to align with its diurnal profile across three
  windows, not a property of the model. Its 0–6h error in the same regime is the
  worst of any model.
- These figures use 3 windows of the 24. Re-run with `--windows` covering more
  before anything here goes in a paper.

### Why this matters beyond the baselines

Three consequences follow from the results above.

**1. The ~44.5 MW wall is not an averaging artefact — but it is not uniform
either.** Models that fail in different places can produce near-identical totals,
which would have made the wall a property of the regime mix rather than an
information limit. That is not what happened: ARMA and the GRU are close in
*every* regime, not just on average. But XGBoost's win at high power and short
horizon is real and localised, so the models are not interchangeable even where
their totals nearly agree.

**2. There is a per-regime selector worth building.** Taking the best model in
each (regime, horizon) cell costs one lookup table and no new modelling. From the
table above it would use seasonal naive for parked short-horizon, XGBoost for
high-power short-horizon, ARMA or the GRU elsewhere. Whether that beats 44.32 is
untested and should be tested — if it does, it belongs in the baseline table as
an honest comparator, and the evolutionary search has to beat it too. This is the
single cheapest remaining improvement in the study.

**3. It sharpens the generalisation question, and points the same way the `xy`
tier does.** Regime specialisation is *turbine-independent* structure: it
transfers to any turbine that happens to be at that power level. Turbine
*identity* is not — and the `xy` feature tier found no measurable value in it
(−0.18 MW, inside the noise floor). Taken together, these two results suggest the
exploitable structure in SDWPF is about **state**, not about **which turbine**,
which argues against needing an instance-specific portfolio and in favour of a
single evolved program that switches on regime. If evolution discovers regime
switching on its own, that is a genuinely positive result, and this section is
what makes it recognisable as one rather than as noise.

## 7. Reproduction

From the package root (`windevolve_sdwpf/`), with `SDWPF_DATA_DIR` exported.

```bash
# 0. confirm every path resolves
python paths.py

# 1. build the split (once; skip if evaluation/splits/ already exists)
python evaluation/val_windows.py --dry-run
python evaluation/val_windows.py
python evaluation/val_windows.py --verify --verify-all

# 2. plan the sweeps
python analysis/sweep.py --list
python analysis/sweep.py gbdt --dry-run

# 3. smoke test on 2 windows before committing to the full run
python analysis/sweep.py naive --windows 2

# 4. the sweeps, cheapest first
python analysis/sweep.py naive                     # ~20 min
python analysis/sweep.py climatology               # ~30 min
python analysis/sweep.py arma                      # ~1-2 h,  CPU
python analysis/sweep.py gbdt                      # ~2-3 h,  CPU
python analysis/sweep.py gru --only seed,infer_mode,clip,scale   # ~40 min, reuses checkpoints

# 4b. the GRU retraining axes -- OPT IN, ~1 GPU-hour per point.
#     Run train_loop first; it answers the "why did the old one score
#     better" question and it is only two points.
python analysis/sweep.py gru --only train_loop     # ~1 h  (the official arm reuses cache)
python analysis/sweep.py gru --only architecture   # ~4 h
python analysis/sweep.py gru --only optim          # ~2 h

# 5. figures and tables
python analysis/plots.py predict --models naive,climatology,arma,gbdt,gru
python analysis/plots.py figures --windows 1,12,24 --turbines 0,42,97
python analysis/plots.py sweeps
python analysis/plots.py regimes      # writes section 6b
python analysis/plots.py summary      # writes section 5.4

# optional: plot each model at its best VALIDATION configuration instead
python analysis/plots.py predict --models naive,climatology,arma,gbdt,gru --best
```

Outputs land in `runs/sweeps/` (TSVs, status JSON, cached fit artifacts) and
`runs/plots/` (PNGs, cached forecasts, `sweep_tables.md`, `summary_table.md`).

Note that `analysis/plots.py predict` reloads all 134 GRU checkpoints once per
model, and `sweep.py gru` reloads them once per grid point — about twenty times
across that sweep. That is the price of giving every point a clean module and a
clean cache, and it is a few seconds each, not minutes.

---

## 8. What this feeds into

- **The seed program's initial score.** The reference numbers in §5.4 are what
  the OpenEvolve and agentic-loop arms start from, measured with the identical
  call those loops make. ARMA(1,0,1) at 44.976 is the strongest reference; the
  climatology at 47.768 is the more forgiving starting point if a seed program
  needs room to improve.
- **The bar for "discovery," and it is higher than it looks.** The tuning
  headroom column reads 0.60–0.72 MW, but the only measured noise floor is
  ±0.396 MW, and best-of-ten selection manufactures a gain of that size on its
  own. **A search that returns 0.5 MW better than a baseline has demonstrated
  nothing.** Any claimed improvement needs either a margin well above 1 MW or a
  repeated-seed measurement of its own.
- **The shrinkage control.** Every baseline gains 0.46–0.83 MW from a single
  multiplication by 0.8, which is as much as its entire hyperparameter sweep.
  When the search finds this — and it will, early — the shrinkage-optimised
  baseline is what it must be compared against.
- **The noise floor.** ±0.396 MW from the XGBoost seed axis is the resolution
  limit for every subsequent ablation: islands, iteration budget, and the
  HPO-versus-model-discovery search-space split. The GRU's floor is still
  unmeasured and should be filled in.
- **An early answer on generalisation.** Turbine identity (`xy`) bought nothing
  measurable, and neither did farm layout. The exploitable structure appears to
  be about power *state* rather than about *which turbine* — which points toward
  a single generalising program rather than a per-turbine portfolio, and is worth
  raising with Prof. Shi before search budget is spent discovering it the
  expensive way.

## 9. Open items

| Item | Cost | Why |
|---|---|---|
| Extend the XGBoost sweep downward (`trees` 50/100, `depth` 2/3, `lr` 0.01/0.02) | ~30 min CPU | every capacity optimum sits at the grid edge; the true optimum is outside the grid |
| Three GRU retrains at different seeds | ~30 min GPU | the GRU has no measured noise floor, so half its column cannot be read |
| Retest the `turbine` feature tier at `trees=200, depth=4` | ~10 min CPU | the 2.01 MW penalty may be over-capacity rather than the features |
| Build and score the per-regime selector | ~1 h | cheapest remaining improvement; would become a baseline in its own right |
| Re-run the regime analysis over all 24 windows | ~10 min | current figures use 3 windows; `mid` has n=13 |
| Score every baseline once on `final_phase_test` | ~1 h | at project end, alongside the champions, in a single pass |