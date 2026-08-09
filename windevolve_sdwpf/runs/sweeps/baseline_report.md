# Baseline Forecasters for the SDWPF Wind Power Prediction Task: Characterisation and Sensitivity Analysis

## Abstract

Automated research loops require a reference point: a set of forecasters whose
performance is known, whose sensitivity to their own hyperparameters has been
measured, and whose scores were obtained through the identical evaluation call
the search itself will make. Without this, an improvement discovered by search
cannot be distinguished from ordinary tuning, from metric exploitation, or from
noise.

This report characterises five baseline forecasters on the SDWPF benchmark (134
turbines, 48-hour horizon, 10-minute resolution) under a 197/48-day
train/validation protocol that leaves the competition test set untouched. It
reports 102 scored configurations across five parameter sweeps, a measured
evaluation noise floor, a six-tier feature ablation for the tree model, and a
regime-conditional error analysis.

## 1. Introduction

### 1.1 Contributions

1. A characterisation of five baseline forecasters spanning four model families
   under a leakage-free train/validation protocol (§3.1, §4).
2. A unified sweep harness scoring 102 configurations through the official
   competition grader (§3.3).
3. A measured evaluation noise floor, and the observation that reported tuning
   headroom is not distinguishable from selection bias (§4.1, §4.7).
4. A nested six-tier feature ablation for the tree model, isolating the
   contribution of wind speed, auxiliary sensor channels, farm-wide aggregates,
   spatial neighbourhood, and turbine identity (§4.5).
5. A decomposition showing that uniform output shrinkage is mean-absolute-error
   optimisation rather than forecasting improvement (§5.2).
6. A regime-conditional error analysis partially confirming a reported
   short-horizon failure mode of neural forecasters at extreme power states
   (§4.8).

---

## 2. Problem Setting

### 2.1 Dataset

The Spatial Dynamic Wind Power Forecasting (SDWPF) dataset comprises 245 days of
10-minute SCADA records from 134 turbines at a single wind farm, released for the
Baidu KDD Cup 2022. Each record carries 13 columns: turbine identifier, day,
timestamp, wind speed (`Wspd`), wind direction (`Wdir`), external and internal
temperature (`Etmp`, `Itmp`), nacelle direction (`Ndir`), three blade pitch
angles (`Pab1–3`), reactive power (`Prtv`), and active power (`Patv`), the
forecast target. A separate file gives each turbine's planar coordinates.

Two properties of the data govern much of what follows. Active power is heavily
**zero-inflated** — turbines are frequently parked for curtailment, maintenance
or icing — and the dataset contains **no forecast weather**, only historical
observations, so a 48-hour-ahead prediction cannot be conditioned on future wind.

### 2.2 Evaluation metric

For each turbine, MAE and RMSE are computed in MW over the unmasked timesteps of
one window. Both are **summed across the 134 turbines**, then **averaged across
the 24 windows**, and the score is half their sum:

<p align="center"><code>score = (RMSE + MAE) / 2</code></p>

Lower is better. Scoring is delegated to the competition's own evaluation code
(`kddcup22-sdwpf-evaluation/paddlepaddle/metrics.py`), verified to a delta of
0.00e+00 against the reference evaluator.

The grader applies a **validity mask** to ground truth, excluding timesteps that
are missing, carry negative power, show zero power while wind exceeds 2.5 m/s,
have a pitch angle above 89°, or contain out-of-range direction readings.

It also enforces two **run-level guards** that can reject a whole window
outright. The *zero-prediction* guard flags any turbine whose entire 288-step
forecast is zero, and rejects the window if the share of non-zero turbines falls
below a threshold. The *identical-prediction* guard does the same for turbines
whose forecasts are nearly constant. A separate accuracy gate derived from the
final-24-hour RMSE also exists but never fired in this study. These guards prove
consequential in §4.2.

### 2.3 Program contract

Every baseline implements a single function:

```python
forecast(envs) -> np.ndarray, shape (134, 288, 1), active power in kW
```

`envs["path_to_test_x"]` names one window's input file: 270,144 rows, comprising
134 turbines stacked turbine-major with 2,016 rows each (14 days at 10-minute
resolution). The program reads this file itself, and `forecast()` is invoked once
per window.

---

## 3. Experimental Setup

### 3.1 Data split

All results use a **197-day training / 48-day validation split** of the 245-day
file. The competition test set (`final_phase_test`) is not read anywhere in this
study.

The 48-day validation period is 6,912 timesteps, exactly 24 × 288, so at 24
windows the origins are spaced 288 steps apart and the windows **tile** the
period: every ground-truth timestep is scored exactly once. Origins are evenly
spaced with no randomisation, so the split is reproducible from its arguments
alone.

### 3.2 Baseline suite

| Model | Family | Fitted | Trains on | Reads at prediction | Shared across turbines |
|---|---|---|---|---|---|
| Seasonal naive | copy-forward | no | — | window `Patv` | n/a |
| Climatology | time-of-day average | no | — | window `Patv` | n/a |
| ARMA / ARIMA | linear, univariate | 134 models | `Patv` only | last 432 steps of `Patv` | no |
| XGBoost | gradient-boosted trees | 1 model | all channels | window history, all channels | **yes** |
| GRU | recurrent network | 134 models | 10 channels | last 144 steps × 10 channels | no |

### 3.3 How the sweeps work

Every configuration in this report is scored by the same call the search loops
make — `evaluator._score_forecaster`, on the same 24 validation windows. A single
harness (`analysis/sweep.py`) loads each baseline fresh, overrides its
module-level parameters, scores it, and appends one row per configuration to a
TSV. No baseline source is modified at run time, and fitted models are cached so
that inference-only parameters reuse a single fit.

Sweeps vary **one parameter at a time** around a reference configuration rather
than exploring a full grid. A full grid over the tree model's parameters alone
would need several hundred fits; varying one at a time answers the question this
study asks — which parameters move the score at all — in 29.

Each parameter is one of four kinds, and this determines what may be concluded
from it:

| Kind | Meaning |
|---|---|
| noise | the same configuration repeated; sets the smallest difference the study can resolve |
| ablation | answers a question about the data or method; reportable either way |
| control | not a hyperparameter — a check on the metric itself |
| tuning | seeks a better number only |

### 3.4 Changes made to the baselines

Three baselines needed parameters exposed before they could be swept. In each
case the defaults reproduce the prior behaviour, verified by comparison rather
than inspection.

**Climatology** gained `N_DAYS` (how many trailing days enter the average),
`HALFLIFE_DAYS` (weight days by recency instead of cutting them off), `AGG` (mean
or median across days) and `SMOOTH_STEPS`. Defaults produce bit-identical
forecasts to the original (max absolute difference 0.0).

**XGBoost** gained six nested feature tiers, a neighbourhood size `k`, and a
choice of training objective. The `base` tier reproduces the original 12 features
in the original column order, verified bit-identical.

**The GRU** was changed more substantially. Its training loop previously
reproduced four defects of the official Baidu code; those are now corrected by
default, with the official behaviour available under `--official` and reported as
one sweep point (§4.6). Seeding also moved from a single stream to per-turbine,
which makes partial runs reproducible and lets turbines train in parallel.

**Seasonal naive** was not modified, but its reference configuration moved from
`LAG=144` to `LAG=432`, because `LAG=144` cannot produce a score at all (§4.2).

---

## 4. Results

All scores are `(RMSE + MAE)/2` in MW on the 197/48 validation split, 24 windows,
official grader. Lower is better.

### 4.1 The evaluation noise floor

Three fits of an identical XGBoost configuration, differing only in the random
seed governing forecast-origin sampling and tree construction:

**Table 1.** Evaluation noise floor.

| seed | 42 | 43 | 44 | mean | 1 sd |
|---|---|---|---|---|---|
| score | 46.139 | 46.922 | 46.426 | 46.496 | **0.396** |

The spread between two nominally identical configurations is 0.78 MW. **This
exceeds nearly every effect measured elsewhere in this study**, and is the
yardstick for every difference reported below. Naive, climatology and ARMA are
deterministic, so their differences are exact — though still specific to these 24
windows. The GRU's training is stochastic but was not replicated.

### 4.2 Seasonal naive

**What it does.** Take the most recent `LAG` timesteps of a turbine's history and
repeat that block until the 288-step horizon is filled.

**What `LAG` actually controls.** For `LAG < 288` this is straightforward
repetition: `LAG=144` copies yesterday twice. For `LAG ≥ 288` the block is longer
than the horizon, so only its **first 288 steps** are used — which makes long lags
a *time-shifted copy* rather than a longer memory:

| LAG | what is copied forward |
|---|---|
| 288 | the most recent 48 h |
| 432 | the 48 h ending **1 day** ago |
| 1008 | the 48 h ending **5 days** ago |
| 2016 | the **oldest** 48 h in the 14-day input, ending 12 days ago |

**Table 2.** Seasonal naive sweep.

| axis | setting | score | RMSE | MAE | status |
|---|---|---|---|---|---|
| lag | 2 | — | — | — | rejected: 76.87% turbines all-zero |
| lag | 6 | — | — | — | rejected: 54.48% turbines all-zero |
| lag | 18 | — | — | — | rejected: 100.00% turbines near-identical |
| lag | 36 | — | — | — | rejected: 19.40% turbines all-zero |
| lag | 72 | — | — | — | rejected: 19.40% turbines all-zero |
| lag | **144** | — | — | — | **rejected: 66.42% turbines all-zero** |
| lag | 288 | 68.653 | 75.380 | 61.926 | valid |
| lag | **432** | **59.424** | 66.321 | 52.527 | valid |
| lag | 1008 | 63.115 | 70.587 | 55.642 | valid |
| lag | 2016 | 62.760 | 68.957 | 56.564 | valid |
| scale | nine points at lag=144 | — | — | — | rejected, as lag=144 is |

**Daily seasonal naive cannot be scored on this dataset.** `LAG=144` — the
canonical "tomorrow resembles today" — is rejected by the zero-prediction guard:
copying the preceding 24 hours forward gives an all-zero forecast for any turbine
parked during that day, and 66.42% of turbines were. The regime analysis (§4.8)
places 53% of all turbine-windows in the parked state, so this reflects the
operating profile of the farm rather than an unlucky window.

Six of ten settings fail, and the failure modes differ. Short lags produce
all-zero forecasts; `LAG=18` instead trips the *identical*-prediction guard,
because a three-hour block repeated sixteen times is nearly flat.

**Among the settings that survive, yesterday beats today.** `LAG=432` — the 48
hours ending one day ago — scores 59.42, better than `LAG=288`, which copies the
most recent 48 hours, at 68.65. Going further back is worse again (1008 → 63.11,
2016 → 62.76). Two effects are confounded here and cannot be separated with this
model: an earlier stretch of history is both differently aligned with the diurnal
cycle *and* more likely to contain non-zero readings, which is what lets it pass
the guard at all. The 432 result should therefore not be read as evidence about
wind persistence.

![Seasonal naive sweep](plots/sweep_naive.png)

***Figure 1.** Seasonal naive: score against lag and against the shrinkage
control. Rejected configurations are absent from the curve.*

### 4.3 Climatology

**What it does.** Reshape a turbine's 14-day history into (days × time-of-day),
average down the day axis to get one typical day — 144 values, one per
10-minute slot — and repeat it twice to cover 48 hours. The model has no memory
across windows; it rebuilds its profile from each window's own history.

**What was swept.**

| Parameter | What it changes |
|---|---|
| `n_days` | how many of the most recent days enter the average (hard cutoff) |
| `halflife` | weight days by recency instead: yesterday counts 1, a day `halflife` earlier counts 0.5 |
| `agg` | mean or median across days at each time-of-day slot |
| `scale` | the shrinkage control — multiply the final output by a constant |

**Table 3.** Climatology sweep.

| axis | setting | score | RMSE | MAE |
|---|---|---|---|---|
| n_days | 1 | — | — | rejected: 66.42% all-zero |
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
| scale | 0.9 | 47.167 | 52.221 | 42.114 |
| scale | **0.8** | **46.940** | 52.163 | 41.717 |
| scale | 0.7 | 47.130 | 52.533 | 41.726 |
| scale | 0.6 → 0.2 | 47.787 → 55.708 | | monotonically worse |

**More history is better, and both parameterisations agree.** The hard cutoff
improves monotonically from 2 to 10 days (62.54 → 47.05); the recency weighting
traces the same trajectory from 0.5 to 10 days (64.15 → 47.79). The 14-day point
is 0.72 MW worse than 10 days, within about 2σ of the noise floor, and is not
treated as a real optimum.

The conclusion is that **the recent weather regime carries no detectable extra
information at a 48-hour horizon** — the long-run daily shape dominates. Both
curves are still improving at the edge of the available history, so this is a
*bound* rather than a measurement: climatology sees only the window's own 14
days, and whether 30 days would help is untestable with this model as written.

**The median is worse throughout**, by 3.1 MW at 3 days and 2.7 MW at 14. The
motivating argument was that parked days drag a mean profile down while leaving a
median untouched. That mechanism is real, but its effect on the score runs the
opposite way to what was expected, for reasons given in §5.2.

![Climatology sweep](plots/sweep_climatology.png)

***Figure 2.** Climatology: score against days averaged, recency weighting,
aggregator, and the shrinkage control.*

### 4.4 ARMA / ARIMA

**What it does.** For each turbine, fit a non-seasonal ARIMA(p,d,q) to that
turbine's power series alone. At prediction time the fitted coefficients are held
fixed while the Kalman filter is re-run over the window's recent history to
establish the model's current state, and the forecast is projected 288 steps.

**What was swept.**

| Parameter | What it changes |
|---|---|
| `order` (p,d,q) | model structure. `d` is the differencing order: `d=1` models changes in power rather than power itself |
| `fit_days` | how much **training** history the coefficients are estimated from, done once offline |
| `apply_steps` | how much of the window's 14-day history is fed through the filter at **prediction** time to warm it up. Not a training parameter |
| `scale` | the shrinkage control |

**Table 4.** ARMA/ARIMA sweep.

| axis | setting | score | RMSE | MAE |
|---|---|---|---|---|
| order | (1,0,0) | 45.063 | 48.945 | 41.181 |
| order | (1,0,1) — reference | 44.976 | 48.853 | 41.099 |
| order | (2,0,1) | 45.537 | 49.499 | 41.575 |
| order | **(2,0,2)** | **44.883** | 48.751 | 41.015 |
| order | (1,1,0) | — | — | rejected: 92.54% near-identical |
| order | **(1,1,1)** | **56.631** | 61.433 | 51.829 |
| fit_days | **30** | **44.324** | 48.705 | 39.943 |
| fit_days | 90 | 44.681 | 48.670 | 40.691 |
| fit_days | 197 | 44.976 | 48.853 | 41.099 |
| apply_steps | 144 / 288 / 432 / 1008 / 2016 | **44.9758 for all five** | 48.853 | 41.099 |
| scale | 0.9 | 44.561 | 48.712 | 40.410 |
| scale | **0.8** | **44.517** | 48.980 | 40.053 |

**(a) Differencing is catastrophic.** ARIMA(1,1,1) scores 56.631 against
ARMA(1,0,1)'s 44.976 — 11.66 MW worse, roughly 30σ. ARIMA(1,1,0) is rejected
outright, 92.54% of turbines producing near-identical forecasts: differencing an
already-stationary series leaves approximately a random walk, whose multi-step
forecast is a flat line. An augmented Dickey–Fuller test on SDWPF power rejects a
unit root decisively, and this is its behavioural confirmation. Treating
ARIMA(1,1,1) as a point on this axis rather than as a separate baseline is
therefore appropriate — it is a result about the data, not a model worth
reporting on its own.

**(b) Less training history is better.** Thirty days beats 197 by 0.65 MW,
monotonically. Set beside the climatology result, where *more* history won, the
pair is informative: **the daily level profile is stable and rewards long
history, while the dynamics are non-stationary and reward recent history.** Two
quantities estimated from the same data give opposite answers about how much of
it to use.

**(c) The warm-up length has no effect at all.** All five `apply_steps` settings
return 44.9758 to four decimal places. The exact equality shows this is
structural rather than coincidental: with coefficients held fixed, a stationary
ARMA's Kalman filter converges to steady state within a few dozen steps, so
earlier history washes out entirely. Anything above roughly 144 steps is wasted
computation.

Model order barely matters — (2,0,2) improves on (1,0,1) by 0.09 MW, well inside
noise — so an AR(1) term captures essentially all the available autocorrelation.

![ARMA sweep](plots/sweep_arma.png)

***Figure 3.** ARMA/ARIMA: score against model order, training history length,
warm-up length, and the shrinkage control.*

### 4.5 XGBoost

**What it does.** A single global model shared by all 134 turbines, trained by
direct multi-horizon regression: each training row is one (turbine, forecast
origin, horizon step) triple, with the horizon distance `h` supplied as an input
feature. The model is called once per horizon step rather than fed its own
output, avoiding the error compounding of recursive decoding.

**What was swept.** Capacity parameters (`trees`, `depth`, `lr`), the training
objective, the shrinkage control, and — the main experiment — six nested
**feature tiers**, each a strict superset of the previous one, so that each
adjacent difference isolates one block of information.

**Table 5.** XGBoost sweep.

| axis | setting | score | RMSE | MAE |
|---|---|---|---|---|
| seed | 42 / 43 / 44 | 46.139 / 46.922 / 46.426 | | (Table 1) |
| features | power (9 features) | **45.834** | 50.155 | 41.512 |
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

**Every capacity parameter says the model is overfitting.** Tree count, depth and
learning rate are each monotonic toward a *smaller* model, each at 3–4σ, and in
each case **the best value sits at the edge of the swept range** — so the true
optimum lies outside the grid, and the reported best-tuned score is a lower bound
on what this model can do.

**Table 6.** Feature tier increments, against the ±0.396 MW noise floor.

| Step | What is added | Δ score | Reading |
|---|---|---|---|
| power → base | wind speed at the origin | +0.31 | within noise: **wind speed buys nothing** |
| base → turbine | direction, temperatures, pitch angles, reactive power, parked fraction | **+2.01** | ≈5σ: actively harmful |
| turbine → farm | farm-wide aggregates | −0.57 | ≈1.4σ: marginal recovery |
| farm → neighbour | the *k* nearest turbines | +0.21 | within noise: **layout does not help** |
| neighbour → xy | the turbine's own coordinates | −0.18 | within noise: **identity buys nothing** |

*Wind speed.* Adding it produces no detectable change. This is consistent rather
than surprising: the target is 48 hours ahead and nearly independent of present
wind. Without forecast weather, observed wind speed is close to uninformative.

*Auxiliary channels.* The `turbine` tier costs 2.01 MW, the largest single effect
in the sweep. Two mechanisms fit the capacity results: these features are
origin-time snapshots held constant across all 288 horizons, letting the model
fit the origin state when the distant target does not depend on it; and column
subsampling now dilutes the few useful features among 22. The hypothesis that
pitch angle would identify parked turbines was reasonable and did not survive.
Retesting at the tuned capacity is an open item.

*Spatial tiers.* Neither farm aggregates nor nearest-neighbour features produce a
detectable gain. The physical motivation — upwind turbines seeing weather before
downwind ones — does not materialise at this horizon.

*Turbine identity.* Coordinates uniquely identify a turbine, so supplying them is
the strongest available invitation to memorise per-turbine behaviour. The result
is −0.18 MW, well within noise. **There is little turbine-specific structure to
exploit**, which bears on the project's central question (§5.4).

**Objective function.** Fitting the conditional median (`absoluteerror`) is 2.97
MW worse, degrading RMSE from 50.448 to 54.243 while failing to improve MAE. This
is the opposite of the `scale` control's effect, despite both moving predictions
toward the median: shrinkage applies one global multiplier after the fact,
whereas an absolute-error objective changes what every tree fits. The metric
rewards the first and punishes the second.

![XGBoost sweep](plots/sweep_gbdt.png)

***Figure 4.** XGBoost: score against each swept parameter. The grey band is the
measured noise floor (±1 sd).*

### 4.6 GRU

**What it does.** 134 independent models, one per turbine. Each reads 144
timesteps (24 h) across 10 channels, pads to 432 steps, passes through a
two-layer GRU with 48 hidden units, and projects the final 288 steps to a single
output channel.

**Training-loop correction.** The implementation previously reproduced the
official Baidu baseline including four defects, each verified against Paddle
3.3.1: gradients were never zeroed, so they accumulated across every batch *and*
epoch; the optimiser was applied twice per batch; clipping was per-parameter
rather than global; and validation ran with dropout active, making the
early-stopping signal noisy. Since the purpose of this baseline is comparison
rather than reproducing a published number, **the corrected loop is now the
default**, with the official behaviour kept under `--official` and reported as
one sweep point.

**What was swept.**

| Parameter | What it changes |
|---|---|
| `capacity` | hidden units and number of GRU layers |
| `input_len` | how many past timesteps the model reads before forecasting |
| `optim` | learning rate and its per-epoch decay, varied together |
| `budget` | maximum epochs and early-stopping patience |
| `provenance` | the official loop with all four defects restored |
| `scale` | the shrinkage control |

**Table 7.** GRU sweep. All axes except `scale` retrain 134 models.

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
| optim | **lr=1e-4, no decay** | **44.604** | 48.558 | 40.649 |
| optim | lr=1e-3, decay=0.9 | 44.778 | 48.807 | 40.749 |
| budget | 20 epochs, patience 5 | 44.597 | 48.552 | 40.642 |
| budget | **40 epochs, patience 8** | **44.537** | 48.499 | 40.576 |
| provenance | **official loop, all four defects** | **44.802** | 48.818 | 40.786 |
| scale | **0.8** | 44.643 | 49.122 | 40.163 |

**Input length has no effect.** Extending the observation window from 144 to 288
to 432 timesteps changes the score by 0.043 MW. This was expected to be the
dominant parameter — the model forecasts 48 hours from 24 hours of history — and
produced the *smallest* effect measured. How much past the model sees is not the
binding constraint.

**Capacity helps modestly** and is still improving at the edge of the grid (48 →
96 → 128 gives 45.136 → 44.833 → 44.745). This runs opposite to XGBoost, which
wanted less capacity on every axis.

**The optimiser schedule was the real limitation.** Every `optim` point beats the
reference by about 0.5 MW, and the best is `lr=1e-4` with decay disabled — the
same learning rate as the default, with only the schedule removed. The `budget`
axis confirms it: 40 epochs with patience 8 gives the best GRU score in the
study, 44.537. Under the default schedule the learning rate falls to roughly
1/500 of its initial value by epoch 10, and **the model was stopping before it
converged.**

**The official defective loop scores 44.802, better than the corrected default at
45.136.** This is not an argument for keeping the defects but an explanation of
them: never zeroing gradients accumulates a running sum that behaves like
momentum, and stepping twice per batch doubles the effective step size. Both
compensate for a learning rate decaying to negligibility. **The defects worked as
an accidental workaround for an under-tuned optimiser.** Correcting the schedule
honestly reaches 44.537, beating the defective loop by 0.27 MW — visible only
because both were measured on one split with one grader.

![GRU sweep](plots/sweep_gru.png)

***Figure 5.** GRU: score against capacity, input length, optimiser schedule,
training budget, and the shrinkage control.*

### 4.7 Cross-model summary

**Table 8.** Baseline summary. Reference configurations are unselected; best
tuned configurations are selected on validation and therefore optimistically
biased.

| Model | Reference config | Reference | Best tuned config | Best tuned | Tuning headroom | Best shrinkage | Shrinkage gain | Noise floor |
|---|---|---|---|---|---|---|---|---|
| ARMA | `order=(1,0,1)` | 44.976 | `fit_days=30` | **44.324** | 0.652 | `scale=0.8` → 44.517 | 0.459 | deterministic |
| GRU | `hidden=48, layers=2` | 45.136 | `40 epochs, pat=8` | **44.537** | 0.598 | `scale=0.8` → 44.643 | 0.493 | not measured |
| XGBoost | `seed=42`, base features | 46.139 | `trees=200` | **45.445** | 0.694 | `scale=0.8` → 45.526 | 0.613 | **0.396** |
| Climatology | `N_DAYS=14`, mean | 47.768 | `n_days=10` | **47.047** | 0.721 | `scale=0.8` → 46.940 | 0.828 | deterministic |
| Seasonal naive | `LAG=432` | 59.424 | `LAG=432` | 59.424 | 0.000 | not scoreable | — | deterministic |

102 configurations scored; 17 rejected by the official guards.

**Tuning headroom is not distinguishable from selection bias.** Every value in
that column (0.598–0.721) lies within about 1.5–2σ of the measured noise floor.
Moreover, "best tuned" is a *minimum over roughly ten points*, and taking the
smallest of ten draws from a distribution with σ = 0.396 manufactures an apparent
gain of this size **even if all configurations were identical**. The observed
headroom is almost exactly what selection alone would produce.

The deterministic models are a partial exception — their differences are exact
and repeatable *on these 24 windows*, which is not the same as generalising.

![Prediction comparison, farm aggregate](plots/pred_farm.png)

***Figure 6.** Farm-aggregate power: all 134 turbines summed, per window. This is
the quantity the metric scores; single-turbine panels are illustrative.*

![Error against horizon](plots/error_vs_horizon.png)

***Figure 7.** Mean per-turbine absolute error against horizon step, masked
timesteps excluded.*

### 4.8 Regime-conditional analysis

Each turbine-window is assigned to a regime by its mean power over the final hour
of input. A single aggregate score can conceal that two models with near-
identical totals fail in different places, and prior work reports that neural
forecasters degrade at short horizon in extreme current-power states where tree
models remain accurate.

**Table 9.** Regime membership over 402 turbine-windows (3 windows × 134
turbines).

| Regime | Origin power | n | share |
|---|---|---|---|
| parked | < 30 kW | 213 | 53.0% |
| low | 30–300 kW | 50 | 12.4% |
| mid | 300–800 kW | **13** | **3.2%** |
| high | 800–1200 kW | 54 | 13.4% |
| near rated | > 1200 kW | 72 | 17.9% |

**The farm is bimodal.** Turbines are almost always either stopped or running
near capacity; only 3.2% of turbine-windows begin in the middle of the power
curve. Two consequences: the `mid` regime has n = 13 and **nothing in it should
be reported as a result**, and the farm-level score is dominated by how models
handle *parked* turbines — a question about curtailment and maintenance rather
than wind.

**Table 10.** Mean absolute error (MW) by regime and horizon bucket. Bold marks
the best model in each cell.

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

**The reported failure mode is confirmed at high power only.** At short horizon
in the two high-power regimes, XGBoost beats the GRU by 0.086 and 0.125 MW — the
only cells in the whole table where the tree is a clear winner. At the *low*
extreme the ordering inverts: in the parked regime at 0–6 h, XGBoost is the worst
of the five (0.384) and seasonal naive the best (0.202), copying zeros forward
being exactly right for a turbine that is off and staying off. **The tree takes
over at the top of the power curve, not at both extremes.**

![Regime error](plots/regime_error.png)

***Figure 8.** Error against horizon, one panel per origin-power regime. Curves
are smoothed with a one-hour moving average; tables are unsmoothed.*

![Regime winner](plots/regime_winner.png)

***Figure 9.** Best model per (regime, horizon) cell, annotated with margin over
the runner-up in MW. Margins below the 0.396 MW noise floor should be read as
ties.*

**Table 11.** Trajectory bias: mean predicted minus mean actual power (kW),
averaged over 0–6 h. Negative means the model decays away from its starting state
faster than reality does.

| regime | Naive | Climatology | ARMA | XGBoost | GRU |
|---|---|---|---|---|---|
| parked | −33 | −286 | −267 | −376 | −257 |
| low | −72 | −425 | −397 | −480 | −411 |
| mid | −157 | −568 | −294 | −178 | −243 |
| high | −902 | −1092 | −485 | −272 | −381 |
| near rated | −1064 | −1298 | −488 | −368 | −512 |

**Every model under-predicts in every regime** — all 25 cells are negative — and
severely so at high power, where climatology under-predicts by 1.30 MW per
turbine in the first six hours because it has no mechanism to represent that a
turbine is currently running hard.

![Regime decay](plots/regime_decay.png)

***Figure 10.** Mean forecast trajectory from each starting regime, with actual
overlaid. Divergence from the black line in the first hours is the steepness
effect.*

---

## 5. Discussion

### 5.1 The no-future-weather wall

ARMA, the GRU and XGBoost converge to **≈44.5 MW**: ARMA and the GRU are 0.21 MW
apart, with XGBoost 0.91 MW behind.

That a univariate linear model reading nothing but past power is competitive with
a tuned recurrent network — and better than a gradient-boosted tree with access
to wind speed, direction, temperatures, pitch angles and neighbouring turbines —
is strong evidence that **the binding constraint is information, not model
capacity.** The feature ablation says the same thing from the other direction:
adding nine sensor channels made the tree worse, not better.

The mechanism is simple. The dataset contains no forecast weather. At a 48-hour
horizon, conditioned only on history, the best possible predictor approaches a
conditional average, and every sufficiently flexible method finds the same
conditional average. Model class stops mattering once that ceiling is reached.

### 5.2 Shrinkage is metric calibration, not forecasting

Every model improves at `scale = 0.8`, by 0.46–0.83 MW. Decomposing the metric
shows what is being traded.

**Table 12.** Effect of scaling predictions by 0.8.

| Model | RMSE 1.0 → 0.8 | MAE 1.0 → 0.8 | Score change |
|---|---|---|---|
| ARMA | 48.853 → 48.980 (**+0.127**) | 41.099 → 40.053 (−1.046) | −0.459 |
| GRU | 49.038 → 49.122 (**+0.084**) | 41.233 → 40.163 (−1.070) | −0.493 |
| XGBoost | 50.448 → 50.156 (−0.292) | 41.830 → 40.897 (−0.933) | −0.613 |
| Climatology | 52.665 → 52.163 (−0.502) | 42.871 → 41.717 (−1.154) | −0.828 |

**Shrinkage is mean-absolute-error optimisation, paid for in RMSE.** MAE is
minimised by the conditional *median*; SDWPF power is heavily zero-inflated, so
the median lies well below the mean. Scaling predictions down moves them toward
it. For the two strongest models RMSE actually gets worse; the score improves
anyway because the metric averages both halves and the MAE gain is roughly ten
times the RMSE loss.

This also explains the climatology median result (§4.3): a median-over-days
profile is *higher* in level than the parked-day-depressed mean profile, so it
moves away from the quantity MAE rewards.

An apparent contradiction with §4.8 is worth resolving. If every model already
under-predicts in every regime, why does further shrinkage help? Because the two
statements concern different parts of the distribution. Table 11 reports a
*conditional mean* on high-power starts, which are 31% of turbine-windows; the
aggregate score is dominated by the zero-inflated bulk, where MAE rewards
predicting low. Shrinkage worsens the high-power bias and improves the overall
score. **This is the clearest demonstration in the study that the metric and
forecast quality are not the same quantity.**

### 5.3 Implications for automated model discovery

**A search returning 0.5 MW of improvement has demonstrated nothing.** The tuning
headroom column reads 0.60–0.72 MW, but the measured noise floor is ±0.396 MW and
best-of-*n* selection manufactures a gain of that size on its own. Any claimed
improvement needs either a margin well above 1 MW or a repeated-seed measurement
of its own.

**Shrinkage will be discovered early and must not be credited as discovery.** One
multiplication is worth as much as an entire hyperparameter sweep on every
baseline. Because the shrinkage-optimised baseline is now measured, that credit
can be placed correctly when the search finds it.

**Budget accounting must be stated.** The best-tuned column represents roughly
ten evaluations per model. A search granted sixty iterations is not budget-
matched against it, and any comparison should name both the column and the
evaluation count.

### 5.4 Generalisation across turbines

Two independent results point the same way. Supplying the global tree model with
each turbine's coordinates — the strongest available signal for per-turbine
memorisation — produces −0.18 MW, within noise. Separately, the regime analysis
finds that which model wins depends on whether a turbine is parked or near rated,
and that dependency transfers to any turbine in that state.

Together these suggest **the structure worth discovering is a function of
operating state, not of turbine identity.** That argues against needing an
instance-specific portfolio and in favour of a single evolved program that
switches behaviour on regime. It is an early, partial answer and should be
treated as a hypothesis for the search to test rather than a settled result.

A corollary: a **per-regime selector** — choosing the best model in each (regime,
horizon) cell — costs one lookup table and no new modelling. From Table 10 it
would use seasonal naive for parked short-horizon predictions, XGBoost for
high-power short-horizon, and ARMA or the GRU elsewhere. Whether it improves on
44.32 MW is untested; if it does, it belongs in the baseline table as a
comparator the search must also beat.

---

## 6. Threats to Validity

**Single split.** All results come from one 197/48 partition. The deterministic
models' differences are exact on these 24 windows but were not tested for
stability across alternative splits.

**One measured noise floor.** Only XGBoost's seed axis was replicated. The GRU's
training is stochastic and its 0.598 MW headroom is the same size as XGBoost's
noise, so GRU differences below roughly 0.5 MW should not be claimed.

**Boundary optima.** Every XGBoost capacity optimum, and the GRU's capacity
optimum, lie at the edge of their swept ranges. Reported best-tuned values are
lower bounds on achievable performance.

**Confounded lag result.** The `LAG=432` result (§4.2) mixes two effects —
alignment with the diurnal cycle and evasion of the zero-prediction guard — that
this model cannot separate.

**Regime analysis coverage.** Tables 9–11 and Figures 8–10 use 3 of the 24
windows. The `mid` regime has n = 13 and supports no conclusion. Climatology's
0.049–0.057 MW at high power, 24–36 h is almost certainly a single weather event
aligning with its diurnal profile rather than a property of the model; its 0–6 h
error in the same regime is the worst of any model.

**Not comparable with earlier measurements.** Every model scores better here than
in the previous round despite less training data. The likeliest explanation is
that these validation windows are easier than `final_phase_test`, not that the
models improved. The two sets of figures must not be differenced; only the
internal ordering is comparable, and it changed.

---

## 7. Conclusion and Future Work

Five baseline forecasters spanning four model families converge to approximately
44.5 MW on the SDWPF 48-hour task, with a univariate linear model marginally
leading. That convergence, the null result from adding sensor channels, and the
null results from spatial and identity features together indicate that the
binding constraint is the absence of forecast weather rather than model capacity.
The measured noise floor of ±0.396 MW is comparable to the entire tuning headroom
of every model, setting a resolution limit for all subsequent ablations. A single
constant multiplier improves every baseline by as much as its full sweep, and is
shown to be metric exploitation rather than forecasting improvement.

**Table 13.** Open items.

| Item | Cost | Rationale |
|---|---|---|
| Extend XGBoost sweep downward (`trees` 50/100, `depth` 2/3, `lr` 0.01/0.02) | ~30 min CPU | every capacity optimum lies at the grid boundary |
| Extend GRU capacity upward (hidden 192/256) | ~1 h GPU | 128 is the best value and sits at the edge |
| Retest `turbine` feature tier at `trees=200, depth=4` | ~10 min CPU | the 2.01 MW penalty may reflect over-capacity rather than the features |
| Build and score the per-regime selector | ~1 h | cheapest remaining improvement; would become a baseline in its own right |
| Re-run regime analysis over all 24 windows | ~10 min | present figures use 3 windows; `mid` has n = 13 |
| Score all baselines once on `final_phase_test` | ~1 h | at project end, alongside champions, in a single pass |

---

## Appendix A: Reproduction

From the package root, with `SDWPF_DATA_DIR` exported.

```bash
# Verify paths and build the split (deterministic; skip if present)
python paths.py
python evaluation/val_windows.py --dry-run
python evaluation/val_windows.py
python evaluation/val_windows.py --verify --verify-all

# Inspect what each sweep will do, before running it
python analysis/sweep.py --list
python analysis/sweep.py <model> --explain

# Sweeps, cheapest first
python analysis/sweep.py naive                 # ~20 min, CPU
python analysis/sweep.py climatology           # ~30 min, CPU
python analysis/sweep.py arma                  # ~1-2 h,  CPU
python analysis/sweep.py gbdt                  # ~2 h,    CPU
python analysis/sweep.py gru                   # GPU; retrains 134 models per point

# Figures and tables
python analysis/plots.py predict --models naive,climatology,arma,gbdt,gru
python analysis/plots.py figures --windows 1,12,24 --turbines 0,42,97
python analysis/plots.py regimes --windows 1,12,24
python analysis/plots.py sweeps
python analysis/plots.py summary
```

Outputs are written to `runs/sweeps/` (TSVs, status JSON, cached fit artifacts)
and `runs/sweeps/plots/` (figures, `sweep_tables.md`, `summary_table.md`,
`regime_tables.md`).

## Appendix B: Figure Index

| Figure | File | Content |
|---|---|---|
| 1 | `sweep_naive.png` | seasonal naive parameter curves |
| 2 | `sweep_climatology.png` | climatology parameter curves |
| 3 | `sweep_arma.png` | ARMA/ARIMA parameter curves |
| 4 | `sweep_gbdt.png` | XGBoost parameter curves with noise-floor band |
| 5 | `sweep_gru.png` | GRU parameter curves |
| 6 | `pred_farm.png` | farm-aggregate power, predicted vs actual |
| 7 | `error_vs_horizon.png` | mean absolute error against horizon |
| 8 | `regime_error.png` | error against horizon by power regime |
| 9 | `regime_winner.png` | best model per (regime, horizon) cell |
| 10 | `regime_decay.png` | mean forecast trajectory by starting regime |
| — | `pred_single.png` | single turbine, single window, all models |
| — | `pred_grid.png` | turbines × windows panel grid |

## References

1. Zhou, J. et al. SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting
   over a Large Turbine Array. *arXiv:2208.04360* (2022); *Scientific Data* 11:649
   (2024), doi:10.1038/s41597-024-03427-5.
2. Baidu KDD Cup 2022 SDWPF official baseline and evaluation code.
   PaddlePaddle/PaddleSpatial, `apps/wpf_baseline_gru/`.
3. Eggensperger, K., Lindauer, M. & Hutter, F. Pitfalls and Best Practices in
   Algorithm Configuration. *Journal of Artificial Intelligence Research* 64
   (2019).
4. Toledo, C. et al. AIRA-dojo. *arXiv:2507.02554* (2025).