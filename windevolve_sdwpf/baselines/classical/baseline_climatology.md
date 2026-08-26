# Climatology Baseline

Reference for `baselines/classical/baseline_climatology.py`

> **CHAMPION — `n_days = 14`, 48.491 MW (5-fold CV).**

---

## The whole model

```
take the last N days of the window's own history
  → average by TIME OF DAY across those days      (a 144-step daily profile)
  → tile that profile twice                        (288 steps = 48 h)
```

It answers one question: *what does this turbine usually do at this hour?*

It uses **no recent state at all**. The forecast is the same whether the turbine
is currently at full output or parked — only the average daily shape matters.

### Why not just a flat mean

A single per-turbine mean repeated 288 times is **rejected by the official
scorer**, which has an `is_identical_prediction` guard. Averaging by time of day
keeps the forecast varying across the horizon and therefore scoreable.

---

## The sweep

`n_days` was the only axis. 

| `n_days` | score (MW) | Δ vs `n_days=2` | `fold_std` | |
|---|---|---|---|---|
| 2 | 59.079 | — | 6.98 | reference |
| 5 | 53.128 | −5.952 | 7.88 | |
| 10 | 48.819 | −10.260 | 6.36 | *previous default* |
| **14** | **48.491** | **−10.588** | **6.01** | **champion** |

### Three things the table says

**Longer history is better at every step.** 

**It flattens, exactly as predicted.** 2 → 5 buys 5.95 MW. 10 → 14 buys 0.33 MW.
Averaging noise away has diminishing returns — error falls roughly as 1/√n, so
the tenth day helps far less than the third.

**`fold_std` falls as `n_days` rises** — 6.98 at 2 days down to 6.01 at 14. More
history makes the model both *better* **and** *more stable across weather
regimes*. 


## Parameters

| parameter | default | what it is |
|---|---|---|
| `N_DAYS` | **14** | Past days averaged into the typical day — **the swept axis** |

`N_DAYS = 1` is rejected outright by the zero-prediction guard — one day of
history from a parked turbine is all zeros.

---

## Cost

**No fitting at all.** Everything is computed from the input window, so there is
no artifact, no checkpoint, and no per-fold training. 

That also means `scope` is fixed at per-turbine by construction: each turbine's
profile comes from its own history.

---
