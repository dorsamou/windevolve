# XGBoost Baseline

Reference for `baselines/tree/baseline_gbdt.py`

> **CHAMPION — `feature_set = base+nbr`, 44.079 MW (5-fold CV).**

---

## The whole model

**One global model** shared by all 134 turbines

**Direct multi-horizon.** Each training row is one `(turbine, forecast origin,
horizon step)` triple, and the horizon distance `h` is an **input feature**. So
the model is called once per step.

Called **134 × 288 = 38,592 times per window.**

### How the ensemble is built

200 rounds of the same three steps:

1. Start from a constant prediction — the mean
2. Compute each row's **residual**: actual − current prediction
3. Fit a depth-4 tree to *those residuals*, add 3% of it, repeat

Each tree is trained on **what is still wrong**, not on the original target.
Round 2's tree never sees 480 kW; it sees the 157 kW the ensemble is still short
by. The learning rate keeps each correction small so 200 noisy estimates can
average into something reliable.

---

## The feature tiers — the headline result

Six tiers are **nested**, each a strict superset of the one before. `base+nbr` is
**non-nested**: `base` plus only the three neighbour columns.

| tier | cols | score (MW) | vs base | **increment** | `fold_std` |
|---|---|---|---|---|---|
| power | 9 | 44.683 | +0.461 | | 5.23 |
| **base** | 12 | 44.222 | — | **−0.461** | 5.03 |
| **base+nbr** | **15** | **44.079** | **−0.143** | | **5.16** |
| turbine | 22 | 45.468 | +1.246 | **+1.246** | 3.73 |
| farm | 26 | 45.512 | +1.290 | +0.044 | 3.76 |
| neighbour | 29 | 45.387 | +1.165 | −0.125 | 3.78 |
| xy | 31 | 45.226 | +1.004 | −0.161 | 3.85 |

The **increment** column is what each group adds on top of the tier before it —
the only interpretable quantity in a nested chain.

### Four readings

**Wind speed is worth 0.461 MW.** That is the `power → base` step, and it is the
only clean measurement of what reading wind speed at all buys. Smaller than most
people expect.

**The sensor columns cost 1.246 MW** — ten columns (`wdir_sin/cos`, `etmp_last`,
`itmp_last`, `pab_max_last`, `prtv_last`, `zero_frac_24h`, `wspd_std_24h`,
`patv_mean_48h`, `wspd_yesterday`) costing nearly three times what wind speed
gains. This is the entire "2.01 MW penalty" from the earlier work, now localised.

**Farm aggregates do nothing: +0.044 MW.**

**Neighbour and xy help slightly** inside the nested chain (−0.125, −0.161), but
they are recovering ground in an already-damaged model and cannot undo the
sensor cost.


---

**All of the tier difference comes from one fold.**

| fold | base | base+nbr | turbine | farm | neighbour | xy | gap |
|---|---|---|---|---|---|---|---|
| F1 | 47.3 | 47.2 | 47.8 | 47.6 | 47.5 | 47.4 | 0.6 |
| F2 | 47.8 | 47.7 | 47.2 | 47.5 | 47.4 | 47.3 | 0.6 |
| F3 | 40.1 | 40.1 | 39.9 | 39.8 | 39.8 | 39.7 | 0.4 |
| F4 | 49.4 | 49.4 | 50.0 | 50.1 | 50.0 | 50.0 | 0.7 |
| **F5** | **36.4** | **36.0** | **42.4** | **42.6** | **42.2** | **41.8** | **6.6** |

On four folds every tier scores within 0.7 MW. On **F5** they split by 6.6 MW.

And the arithmetic lands exactly: **6.0 MW ÷ 5 folds = 1.2 MW**, which is the
reported penalty. 


## Capacity sweep

| axis | values | range |
|---|---|---|
| trees | 100 / **200** / 400 | **0.008 MW** |
| learning_rate | 0.01 / **0.03** / 0.1 | **0.010 MW** |
| max_depth | 3 / **4** / 6 / 8 | 0.455 MW |



## Parameters

| parameter | default | what it is |
|---|---|---|
| `FEATURE_SET` | **`base+nbr`** | Which column set is exposed|
| `N_ESTIMATORS` | 200 | Boosting rounds, one tree per round |
| `MAX_DEPTH` | 4 | Up to 4 features interacting per path |
| `LEARNING_RATE` | 0.03 | Only 3% of each tree's correction is added |
| `K_NEIGHBOURS` | 5 | Turbines the neighbour columns aggregate over |
| scope | global | One model for all 134 turbines |


## Cost

CPU-only

Under folds each fold needs its own model, fitted on that fold's own
`fold_train.csv` — reusing one across folds would mean four of five scored a
model trained on their own validation targets.
