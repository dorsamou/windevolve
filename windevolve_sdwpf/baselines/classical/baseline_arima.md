# ARMA Baseline

Reference for `baselines/classical/baseline_arima.py`

> **CHAMPION — `order = (2, 0, 1)`, `fit_days = 30`, 44.300 MW (5-fold CV).**

---

## The whole model

134 independent models, one per turbine, each reading **only that turbine's own
power series** — no wind speed, no time of day, no neighbours.

```
offline:     30 days of training data  →  coefficients  (c, φ₁, φ₂, θ₁)
at forecast: replay the window's last day  →  state
             project 288 steps            →  the 48 h forecast
```

### Coefficients versus state

An ARMA forecast needs **two separate things**

- **Coefficients** say *how the series behaves* — how strongly it reverts, how
  much of the last shock carries forward. Learned once, offline.
- **State** says *where the series is right now*. The coefficients cannot supply
  this.

The fitted model's state is stuck at the end of the *training* data, months
before the window being forecast. So `apply()` replays the window's recent
history through the Kalman filter with coefficients frozen, dragging the state up
to the present. Recipe versus place in the story: fitting learns the recipe, the
warm-up reads you the last page.

---

## What the orders mean

```
ARMA(p, d, q)
       │  │  └── q = how many past SHOCKS feed the next value
       │  └───── d = differencing (0 here)
       └──────── p = how many past VALUES feed the next value
```

For `(2, 0, 1)`:

```
y(t) = c + φ₁·y(t−1) + φ₂·y(t−2) + θ₁·ε(t−1) + ε(t)
           └──────  AR(2)  ──────┘  └─ MA(1) ─┘
```

`ε(t)` is the **shock** — the part of `y(t)` the model could not predict, i.e.
actual minus predicted.

### Why `d = 0`

Differencing models `y(t) − y(t−1)` — the *change* — instead of `y(t)` itself.
You want it when a series is **unanchored**: no value it gets pulled back
toward, so it wanders wherever the last step left it. A turbine's output lives in **[0, 1500] kW** and cannot
leave. A genuine random walk is unbounded and would wander to 5,000 or −800 over
245 days.


## The sweep

| axis | value | score (MW) | Δ vs ref | `fold_std` |
|---|---|---|---|---|
| order | `(1, 0, 0)` | 44.606 | −1.096 | 4.55 |
| order | `(1, 0, 1)` — previous default | 44.512 | −1.190 | 4.43 |
| order | **`(2, 0, 1)`** | **44.300** | **−1.401** | 4.15 |
| fit_days | 10 — *reference* | 45.702 | — | 4.65 |
| fit_days | 90 | 44.513 | −1.189 | 4.50 |

`fit_days` was capped at 90 because **F1 has only 95 training days** 


## Parameters

| parameter | default | what it is |
|---|---|---|
| `DEFAULT_ORDER` | **(2, 0, 1)** | p past values, d differencing, q past shocks |
| `FIT_STEPS` | 30 days | History used to estimate the coefficients |
| `APPLY_STEPS` | 144 (1 day) | Warm-up steps replayed to set the state |
| scope | per turbine | 134 separate coefficient sets |


## Cost

CPU-only, and the most expensive classical model: **134 fits per fold**, so 670
fits for one grid point across five folds. The 5-point sweep took about 54
minutes of fitting.

Every fold needs its own artifact, fitted on that fold's own `fold_train.csv` —
reusing one across folds would mean four of the five scored coefficients
estimated from data inside their own validation period.