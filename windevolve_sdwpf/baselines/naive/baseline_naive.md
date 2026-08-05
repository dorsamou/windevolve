# Naive Baseline (Seasonal Naive)

Reference for `baseline_naive.py`.

## What it does

Copy a block of recent history forward, tiling it across the 48-hour horizon.

```
forecast(h) = Patv(T - LAG + ((h - 1) mod LAG) + 1)        h = 1 .. 288
```

`T` is the forecast origin, `LAG` is the block length in 10-minute steps. Nothing is fitted.


## The LAG parameter, and one real constraint

| `LAG` | Method | Meaning |
|---|---|---|
| 1 | true persistence | repeat the last value — **rejected by the scorer** |
| 144 | daily seasonal naive | tomorrow looks like today (default) |
| 288 | two-day block | the next 48h looks like the last 48h |

**Why `LAG = 1` fails.** Classical persistence repeats a single value for the whole horizon, so every turbine emits a constant series. The official scorer's `is_identical_prediction` guard rejects that outright — verified: it returns the `-1e6` failure penalty, not a score. Any `LAG >= 2` copies a varying block and is scoreable.

## Behaviour

| Property | Value |
|---|---|
| Parameters | none fitted (`LAG` is a configuration choice) |
| Training | none |
| Inputs used | `Patv` only; ignores wind speed and the other 9 variables |
| History used | last `LAG` of the 2016 available steps |

Missing readings are replaced by 0 before copying; short histories are edge-padded; predictions are clipped at zero.

## Weaknesses

These are exactly the openings a better method exploits:

1. **Decays with horizon.** Accurate in the first hour, progressively worse across 48 hours.
2. **Ignores wind speed.** `Wspd` drives power through the turbine power curve, and this never reads it.
3. **Copies noise and outages.** A turbine parked during the copied block is forecast as parked.
4. **Uses a fraction of the history.** With `LAG = 144`, thirteen of the fourteen supplied days are discarded.

## Usage

```bash
# Run from the package root (windevolve_sdwpf/)
python evaluator.py baseline_models/naive/baseline_naive.py

# Compare lags without editing the file
SDWPF_NAIVE_LAG=144 python evaluator.py baseline_models/naive/baseline_naive.py
SDWPF_NAIVE_LAG=288 python evaluator.py baseline_models/naive/baseline_naive.py
```

## Citation

Program contract and scoring follow the official Baidu KDD Cup 2022 SDWPF evaluator (PaddlePaddle/PaddleSpatial, `apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation`).
Dataset: Zhou et al., arXiv:2208.04360 (2022); *Scientific Data* 11:649 (2024), doi:10.1038/s41597-024-03427-5.
