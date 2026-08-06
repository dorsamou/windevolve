# Diurnal Climatology Baseline

Reference for `baseline_climatology.py`.

## Method

For each turbine, reshape its 14 days of history into a `(14, 144)` grid — rows are days, columns are time of day — and average down the columns. That gives a **mean 24-hour power profile**, which is then tiled twice across the 48-hour horizon.

```
profile[s] = mean over days of Patv at time-of-day slot s      s = 0..143
forecast[t, h] = profile[(h - 1) mod 144]
```

The average is NaN-aware, so missing readings are skipped rather than counted as zero.


## An important constraint this baseline reveals

The obvious version of climatology — **one mean per turbine, repeated 288 times** — is **rejected outright** by the official scorer. Every turbine would emit a constant series, tripping `is_identical_prediction`, and the whole submission is thrown out with a `MetricsError`.

Verified directly: a flat per-turbine mean returns `combined_score = -1e6`, not a score.

Averaging by *time of day* keeps the forecast varying across the horizon, so it is scoreable. This is worth knowing generally: the metric will not accept any method whose output is effectively constant, however statistically reasonable that output might be.

## Behaviour

| Property | Value |
|---|---|
| Parameters | none |
| Training | none |
| Inputs used | `Patv` only |
| History used | all 2016 steps (14 full days) |

Predictions are clipped at zero.

## Weaknesses

1. **Ignores current conditions entirely.** If the turbine is at full output right now, the forecast for ten minutes' time is still the historical average.
2. **Only 14 days of climatology.** A true seasonal average would want months.
3. **Assumes a meaningful daily cycle.** Where wind has little diurnal structure, the profile is close to a flat mean — and that flatness is what the guard nearly catches.
4. **Ignores wind speed.**

## Usage

```bash
python evaluation/evaluator.py baselines/classical/baseline_climatology.py
```


---

## Parameters added for the sweep

Four module-level constants were added so `analysis/sweep.py` can vary the model
without editing it. **All four default to the original behaviour**, verified by
comparing forecasts on a validation window: max absolute difference 0.0.

| Constant | Default | What it does |
|---|---|---|
| `N_DAYS` | `None` (all 14) | Average only the most recent *N* days of the window's history. The interpretable answer to "14 days or 5 days". |
| `AGG` | `"mean"` | `"mean"` or `"median"` across days. SDWPF power is heavily zero-inflated — curtailment, icing, maintenance — so a couple of parked days drag a mean profile down and leave a median untouched. |
| `HALFLIFE_DAYS` | `None` (uniform) | Exponentially weight the days instead of cutting them off: the most recent day gets weight 1, a day `HALFLIFE_DAYS` earlier gets 0.5. Only applies when `AGG` is `"mean"`. NaN-aware, so a gap does not bias its day toward zero. |
| `SMOOTH_STEPS` | `0` (off) | Width, in 10-minute steps, of a **circular** moving average over the finished 144-step profile. Circular because time of day wraps: 23:50 is adjacent to 00:00. |

**Why both `N_DAYS` and `HALFLIFE_DAYS`.** They answer the same question with
different sharpness. A hard cutoff trusts day 5 completely and discards day 6
entirely, which is not a belief anyone actually holds; the half-life says recent
days matter more and older ones fade. `N_DAYS` is the readable summary,
`HALFLIFE_DAYS` is the better-posed version. If they disagree about how much
recent history matters, trust the half-life curve.

**Why `SMOOTH_STEPS` is swept crossed with `N_DAYS`, not alone.** Averaging few
days leaves a visibly noisy profile, which the model then forecasts as though it
were structure. Smoothing trades that noise for a little blur. The two knobs only
make sense together, so the grid pairs them rather than varying each separately.
