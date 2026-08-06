# Gradient Boosted Trees Baseline (XGBoost)

Reference for `baseline_gbdt.py`

##  gradient boosting 

It builds trees **in sequence, each correcting the previous ones' errors**:

1. Tree 1 makes a rough prediction.
2. Look at its errors (where it was too high or too low).
3. Tree 2 is trained to predict *those errors* — the correction.
4. Add tree 2's correction on top of tree 1.
5. Look at the remaining errors, train tree 3 on those, and so on.

Each tree only nudges the prediction a little (the "learning rate" shrinks its contribution). After a few hundred trees, the sum is highly accurate. The "gradient" in the name comes from the fact that fitting each new tree to the leftover errors is a gradient-descent step on the loss.

So **GBDT = Gradient Boosted Decision Trees** = hundreds of small trees, each correcting the last, summed together. This implementation uses **XGBoost**, the library most common in wind-power-forecasting and competition literature.

(Contrast with a random forest, which builds many trees *independently* and averages them. Boosting builds them *sequentially*, each depending on the last — usually more accurate.)

## How forecasting becomes regression

A tree model has no notion of time — it maps a feature vector to a number. So forecasting has to be recast as regression, via **direct multi-horizon regression**:

Each training example is one `(turbine, forecast origin, horizon step)` triple:

- **Target:** the actual power at `origin + h`.
- **Features:** everything known *at the origin*, plus how far ahead we are predicting (`h`).

So the model learns: "given the recent state, and that I am predicting *h* steps ahead, what is the power?" Including `h` as a feature lets **one model** answer for any horizon.

**"Direct" matters.** To predict 48 h ahead, the model builds features and predicts that step in one shot. It does *not* predict step 1, feed that back to predict step 2, and so on (that is "recursive" forecasting, which compounds errors). Direct forecasting has no error snowball.

## How the training data is built (step by step)

The model learns from thousands of question-and-answer examples, all taken from the training CSV — which by default is `val_train.csv`, the file truncated to end before the validation period (days 1–197), **not** the full 245-day file. Fitting on the full file would train the model on its own validation targets. One example is:

- **Question (features):** numbers describing "now" (the origin) plus how far ahead we are asking (the horizon `h`).
- **Answer (target):** the actual power `h` steps after "now" — which we can simply look up, because it is historical data.

**One origin makes many rows.** Pick a random moment ("now"). The "now" numbers — power now, recent averages, wind now — are computed once. Then we vary `h` and make one row per value:

```
"now" numbers fixed:  power=500, last-hour-avg=480, wind=7, ...

row h=1:   [ ...now numbers..., h=1  ]   answer: power 1 step later
row h=2:   [ ...now numbers..., h=2  ]   answer: power 2 steps later
row h=3:   [ ...now numbers..., h=3  ]   answer: power 3 steps later
...
```

The "now" numbers repeat across these rows; only `h` (and the two features that depend on the predicted step, `hour_of_day` and `patv_yesterday`) change. So **one origin fans out into a block of rows, one per horizon step.**

**Many origins, all turbines, piled together.** We repeat this at 150 random origins, for each of the 134 turbines, and stack every resulting row into one big matrix. That matrix is the training set handed to XGBoost.

**Striding, for volume.** Making a row for every `h` from 1 to 288 would give 150 × 134 × 288 ≈ 5.8 M rows. `--stride 12` keeps only every 12th horizon (`h = 1, 13, 25, ..., 277` — 24 of the 288), cutting this to ~480 K rows, which trains far faster. The model does not lose the skipped steps: `h` is just a number it learns a smooth relationship over, so it interpolates to `h = 7` fine even having trained only on `h = 1` and `h = 13` — like fitting a smooth curve from 24 points instead of 288.

**One asymmetry to remember:** training is *strided* (every 12th `h`, for speed), but prediction is *not* — `forecast` uses every `h` from 1 to 288, because there we want the complete 48 h trajectory, not a training sample.

The three terms, plainly:
- **origin** — one "now", a random moment we build examples from.
- **horizon (`h`)** — how far ahead the question asks.
- **feature** — one of the numbers in the question (power now, wind now, `h`, ...).

## Algorithm note: how it trains, how it predicts

### Training (fitting the trees to residuals)

Gradient boosting builds the model as a **sum of trees**, each one predicting the leftover error of the trees before it. With squared-error loss (`objective="reg:squarederror"`), that leftover error is exactly the **residual** = actual − current prediction.

```
start:      prediction P0 = a constant (the mean target)
round 1:    residual r1 = actual − P0
            fit tree T1 to predict r1
            P1 = P0 + lr · T1            (lr = learning rate, e.g. 0.05)
round 2:    residual r2 = actual − P1
            fit tree T2 to predict r2
            P2 = P1 + lr · T2
...
round N:    residual rN = actual − P(N−1)
            fit tree TN to predict rN
            PN = P(N−1) + lr · TN

final model:  prediction = P0 + lr · (T1 + T2 + ... + TN)
```

Each tree only nudges the prediction by a small fraction `lr`, so no single tree dominates and the sum converges gradually. Fitting each tree to the residual is a gradient-descent step on the squared-error loss — hence "gradient" boosting. In this baseline `N = --trees` (default 400), each tree at most `--depth` deep (default 6).

Building the training rows (origins × horizons, described above) happens once, up front; then all `N` rounds run on that fixed matrix.

### Prediction (summing the trees)

At forecast time the trees are fixed. For one input row, prediction is just the constant plus every tree's shrunk output added up:

```
prediction = P0 + lr · (T1(row) + T2(row) + ... + TN(row))
```

For a full forecast, one turbine's `(288, n_features)` matrix (every horizon `h = 1..288`, no striding) is passed through all `N` trees at once, giving 288 power values. `n_features` depends on the feature tier the model was trained with — 12 for the default `base` tier, up to 31 for `xy`; see "Feature tiers" below. Repeat for 134 turbines, clip negatives to zero, and return `(134, 288, 1)`.

## Features

All are computed from history at or before the origin — never the future.

| Feature | Why it is there |
|---|---|
| `h` | the horizon step; lets one model handle all 288 steps |
| `hour_of_day` | time of day at the predicted step, for any diurnal shape |
| `patv_last` | power right at the origin |
| `patv_mean_1h`, `patv_mean_6h`, `patv_mean_24h` | recent power level at three timescales |
| `patv_std_24h` | recent variability |
| `wspd_last`, `wspd_mean_6h`, `wspd_mean_24h` | **wind speed** — drives the power curve |
| `patv_yesterday` | power one day before the predicted step |
| `patv_turbine_mean` | the turbine's typical output (capacity proxy) |

This is the first baseline that reads **wind speed**. On data with a real power curve, feature importance splits sensibly between recent power (the strongest short-term predictor) and wind speed (the power-curve signal).

## Parameters

| Setting | Default | Meaning |
|---|---|---|
| `--origins` | 150 | forecast origins sampled per turbine |
| `--stride` | 12 | keep every 12th horizon step (24 of 288) when building training rows |
| `--trees` | 400 | number of boosting rounds (trees) |
| `--depth` | 6 | maximum depth of each tree |
| `--lr` | 0.05 | learning rate (shrinkage per tree) |
| `--seed` | 42 | RNG seed for sampling and the model |
| `--model` | `gbdt_model.joblib` | output/input path |

Subsampling is necessary: 134 turbines × ~30,000 origins × 288 horizons is far too many rows. The defaults give a few hundred thousand training rows — fits in memory, trains in minutes on CPU. The fitted model is cached at module level, so it loads once rather than 142 times.


## Usage

```bash
# Train the global model (writes gbdt_model.joblib beside the .py)
python baselines/tree/baseline_gbdt.py --origins 150

# Score with the official metric
python evaluation/evaluator.py baselines/tree/baseline_gbdt.py

# A quicker, smaller model for testing
python baselines/tree/baseline_gbdt.py --origins 50 --trees 100
```

## Weaknesses

1. **No temporal structure inside the model.** Trees see a fixed feature vector; any notion of sequence must be hand-built as lag features. A poor feature set caps performance.
2. **Manual feature engineering.** The choice of lags and windows is a human modeling decision — precisely the gap an evolutionary search is meant to close automatically. (A pointed irony worth noting in the writeup.)
3. **No spatial information.** Turbine coordinates are available in `envs["path_to_locations"]` but unused.
4. **Sampling variance.** Origins are randomly sampled, so results shift slightly with `--seed`. Average over seeds before claiming a difference from another method.

---

## Parameters added for the sweep

### Feature tiers

The feature list is no longer fixed. `FEATURE_SET` (or `--features`) selects one
of six **nested** tiers, each a strict superset of the one before, so each
adjacent pair isolates exactly one block of information.

| Tier | Adds | n | The question it answers |
|---|---|---|---|
| `power` | — (`base` minus the wind-speed columns) | 9 | |
| `base` | `wspd_last`, `wspd_mean_6h`, `wspd_mean_24h` | 12 | **what is wind speed worth?** |
| `turbine` | `wdir_sin`, `wdir_cos`, `etmp_last`, `itmp_last`, `pab_max_last`, `prtv_last`, `zero_frac_24h`, `wspd_std_24h`, `patv_mean_48h`, `wspd_yesterday` | 22 | what are the turbine's other channels worth? |
| `farm` | `farm_patv_now`, `farm_wspd_now`, `farm_patv_24h`, `turbine_share` | 26 | what is farm-wide state worth? |
| `neighbour` | `nbr_patv_last`, `nbr_wspd_last`, `nbr_wspd_mean_6h`, over the *k* nearest turbines | 29 | **what is the farm layout worth?** |
| `xy` | `turb_x`, `turb_y` | 31 | what is turbine *identity* worth? |

`base` reproduces the original 12 features **in the original column order**,
verified bit-for-bit against the previous implementation across several
(turbine, origin) pairs. The order matters as well as the set: reordering would
change which columns `colsample_bytree` draws and move the score for no reason.

### Notes on three of the tiers

**`turbine` gives the model a way to tell *why* a turbine is at zero.** The
original feature set reads power and wind speed and nothing else. `pab_max_last`
— the largest of the three pitch angles — says the blades are feathered, i.e.
the turbine is deliberately spilling wind and will probably still be doing so in
an hour. `zero_frac_24h` says how much of the last day it spent at zero. Given
how much of SDWPF is zero-inflated, this is the tier with the most obvious room
to help. `Wdir` enters as sin/cos rather than degrees because direction is
circular: 359° and 1° are one degree apart but 358 apart as raw numbers, and a
tree would otherwise spend splits rediscovering that.

**`farm` and `neighbour` are spatial but carry no identity.** Both are aggregates
over *other* turbines — the farm mean at the origin, and the mean power and wind
speed of the *k* nearest turbines from
`sdwpf_baidukddcup2022_turb_location.csv`. The physical story is that upwind
turbines see weather before downwind ones do, which is what the competition's
spatial entrants were after. `k` defaults to 5 and is itself sweepable.

**`xy` is a turbine identifier in disguise, and is reported separately.** Each
turbine has unique coordinates, so a tree given x/y can memorise per-turbine
behaviour. That quietly converts the *global* model into 134 private models and
destroys the one property that makes this baseline the right comparator for the
generalisation question. It is included because the `neighbour` → `xy` gap is a
direct estimate of how much of the achievable score is turbine-specific rather
than transferable — the project's central question in miniature. It is **not**
part of "the tuned XGBoost".

### Other parameters

| Parameter | Default | Notes |
|---|---|---|
| `k` / `--k` | 5 | Nearest turbines aggregated by the `neighbour` tier. |
| `objective` / `--objective` | `reg:squarederror` | The score is `0.5*(RMSE+MAE)`, so squared error over-weights the tail relative to what is scored and `reg:absoluteerror` under-weights it. Neither is the right loss, which is what makes measuring both informative. |
| `subsample`, `colsample_bytree`, `min_child_weight` | 0.8, 0.8, 1 | Previously hardcoded; now arguments to `train()`. Defaults unchanged. |

The saved `.joblib` now records `feature_set` and `k` alongside the model, and
`forecast()` reads the tier **from the checkpoint** rather than from the module
constant — so a model trained on one tier can never be scored with another
tier's features.
