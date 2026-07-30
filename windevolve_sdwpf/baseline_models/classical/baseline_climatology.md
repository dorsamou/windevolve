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
python evaluator.py baseline_climatology.py
```

