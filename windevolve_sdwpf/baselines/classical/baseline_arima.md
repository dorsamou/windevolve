# ARIMA Baseline

Reference for `baseline_arima.py`.

## What it is

A **per-turbine, non-seasonal ARIMA(p,d,q)** fitted once on the training series. Unlike the naive and climatology baselines, this one **estimates parameters from data**.

### ARIMA in one line each

- **AR (p)** — predict from a weighted sum of recent *values*. `power(t) = φ₁·power(t−1) + … + φₚ·power(t−p)`. Persistence is the trivial case φ₁=1.
- **I (d)** — difference the series `d` times to remove drift. Here **d = 0**: the ADF test (below) shows the series is already stationary, so no differencing is applied and the model works on the raw power. (When used, differencing is plain subtraction `y(t) − y(t−1)`, *not* a variance transform like box-cox; box-cox would need strictly positive values, which the zeros from parked turbines break.)
- **MA (q)** — predict from a weighted sum of recent *errors*.

Written ARIMA(p,d,q); here the default is **(1,0,1)** — an ARMA(1,1) model, since the ADF test shows no differencing is needed.

## How the training data is preprocessed

The training path is deliberately light — much lighter than the test-scoring path — because it only needs a clean 1-D power series to estimate coefficients from. For each turbine, `_load_turbine_series`:

1. **Reads only that turbine's rows and only the `Patv` column** (via `skiprows` and `usecols`), so the other 12 columns and other turbines are never loaded into memory.
2. **Replaces NaN with 0** (`np.nan_to_num`) — parked or missing readings become zeros.
3. **Keeps the trailing `FIT_STEPS`** (the last 245 days by default).

That is the whole preprocessing. Note what it does *not* do: no masking, no kW→MW conversion, no splitting into per-turbine blocks. Those belong to the *evaluation* path (`evaluator.py` / `metrics.py`), which must reproduce the official score.

## The three diagnostic tests

The three ARIMA orders (p, d, q) are each chosen by one diagnostic on the training series. Run in the ACF/PACF/ADF notebook.

### ACF — the AutoCorrelation Function (chooses q, and flags seasonality)

The ACF measures how correlated the series is with **its own past**, at each lag *k*: "how similar is power now to power *k* steps ago?" It captures *all* influence at that lag, including indirect paths (lag 20 relating to now *through* lag 10).

- **What it shows on this data:** a slow, smooth geometric decay from ~0.97, with **no sharp cutoff** at any low lag. A slow decay is the signature of a highly persistent (AR-dominated) process. The absence of a sharp cutoff means there is little pure moving-average structure, so **q is small (0 or 1)**.
- **The dashed line at lag 144:** a true daily cycle would show a distinct **bump** at lag 144 (power now resembling power 24 h ago). The decay passes smoothly through it with no bump → **no daily seasonality**, so no seasonal term. This independently confirms the naive finding that the LAG=144 win was recency, not a daily cycle.

### PACF — the Partial AutoCorrelation Function (chooses p)

The PACF measures the same lag-*k* correlation but **strips out the indirect paths** — the *direct* effect of lag *k* beyond what lags 1…k−1 already explain.

- **What it shows on this data:** a single sharp spike at lag 1, then everything collapses into the noise band and stays there. A PACF that **cuts off sharply after lag 1** is the textbook AR(1) signature → **p = 1**.

A slowly-decaying ACF beside a lag-1 PACF spike is the canonical AR(1) fingerprint, so p and q are well-justified by hand — no search required.

### ADF — the Augmented Dickey-Fuller test (chooses d)

The ADF test formally checks for **stationarity** — whether the series has a "unit root," a stochastic trend that makes it wander without returning to a stable mean. That wandering is exactly what differencing (the `d`) removes.

- **Null hypothesis:** the series *has* a unit root (non-stationary → needs differencing).
- **p < 0.05:** reject the null → the series is **stationary** → this level of differencing is enough.
- **p ≥ 0.05:** cannot reject → difference once more and re-test.

- **What it shows on this data:** the ADF statistic is around −14 to −17 with **p ≈ 1e-26** across turbines — an overwhelming rejection of the unit root. The series is **stationary → d = 0**.
- **Why this matters:** the slow ACF decay initially *looked* non-stationary and suggested d=1. The ADF test overrules that: the series only looks non-stationary because it is highly persistent, but it is bounded (wind power runs 0 to rated capacity and physically cannot wander off), so it is in fact stationary. Differencing a stationary series (over-differencing) introduces spurious MA structure and hurts the model, so **d = 0 is correct**, giving an ARMA(1,1) rather than ARIMA(1,1,1).

## The four design choices

**1. Per turbine.** ARIMA fits the coefficients of one series, so each of the 134 turbines gets its own model.

**2. Fit once.** Each model is fitted a single time on the training series and cached. At inference the fixed coefficients are *applied* to the window's own history (statsmodels `apply`), which re-filters the model on new data **without re-estimating parameters**.

**3. Order (1,0,1) = ARMA(1,1), non-seasonal — from the tests above.** p = 1 (PACF), d = 0 (ADF), q = 1 (ACF), no seasonal term (ACF has no lag-144 bump). A useful side effect of d = 0: forecasts decay toward the series *mean* rather than continuing a trend, so parked turbines revert toward their (low) mean instead of collapsing to a flat zero — fewer degenerate all-zero windows.

**4. No exogenous input.** SARIMAX's "X" would naturally be wind speed, but forecasting 48 h ahead would then require *future* wind speed, which the evaluation protocol forbids.

## Parameters

| Setting | Default | Meaning |
|---|---|---|
| `--order` | `1,0,1` | ARIMA (p,d,q), passed as `p,d,q` |
| `--fit-days` | 245 | trailing training days used per turbine |
| `--model` | `arima_models.joblib` | output/input path for the fitted models |
| `--turbines` | all | subset for quick tests, e.g. `0-9` |

**Two history windows, used at different times:**

- `FIT_STEPS` (`--fit-days` × 144) — history used **once, at fitting**, to estimate the coefficients. Defaults to the full **245 days**.
- `APPLY_STEPS` — history used **at inference, per window**, to seed the state before forecasting. Defaults to 3 days. Only the recent lags matter to a fitted ARMA(1,1), so this is a conservative warm-up, not information the forecast depends on in bulk.

## Usage

```bash
# Fit one ARMA per turbine (writes arima_models.joblib)
python baseline_models/classical/baseline_arima.py --turbines 0-133

# Score with the official metric
python evaluator.py baseline_models/classical/baseline_arima.py

# Pass a different order on the command line (format: p,d,q)
python baseline_models/classical/baseline_arima.py --order 2,0,1
```

### Saving and comparing multiple orders

`--model` sets where the fit is saved, so different orders can live in different files, and `SDWPF_ARIMA_MODEL` selects which one to score:

```bash
# fit three orders to three files
python baseline_models/classical/baseline_arima.py --order 1,0,1 --model arma_101.joblib
python baseline_models/classical/baseline_arima.py --order 2,0,1 --model arma_201.joblib
python baseline_models/classical/baseline_arima.py --order 1,1,1 --model arima_111.joblib

# score each by pointing the env var at its file
SDWPF_ARIMA_MODEL=arma_101.joblib python evaluator.py baseline_models/classical/baseline_arima.py
SDWPF_ARIMA_MODEL=arma_201.joblib python evaluator.py baseline_models/classical/baseline_arima.py
```

Without `--model`, refitting overwrites the default `arima_models.joblib`.

## Weaknesses

1. **Linear**, so it cannot represent the power curve — the fundamental ceiling.
2. **No wind speed**, because future weather is not available.
3. **No cross-turbine information**; each series is modelled in isolation.
4. **Fixed order** for every turbine. The plots look near-identical across turbines, so a shared order is reasonable here, but `pmdarima.auto_arima` could pick per-turbine orders if turbines prove heterogeneous.


---

## Measured results, and the resolution of the d conflict

The code previously defaulted to `DEFAULT_ORDER = (1, 1, 1)` while this document
argued for `d = 0`. **The document was right.** The defaults have been changed to
match it, and the sweep on the 197/48 validation split confirms the argument
empirically. See `docs/baseline_report.md` §4.4.

### Order — d = 0 confirmed, and emphatically

| order | score (MW) | note |
|---|---|---|
| (1,0,0) | 45.063 | |
| **(1,0,1)** | **44.976** | the order this document predicts |
| (2,0,1) | 45.537 | |
| (2,0,2) | 44.883 | 0.09 MW better, inside the noise floor |
| (1,1,0) | — | **rejected**: 92.54% of turbines near-identical |
| (1,1,1) | 56.631 | **+11.66 MW**, roughly 30 sigma |

Differencing costs 11.66 MW, and `(1,1,0)` does not produce a score at all: an
over-differenced stationary series behaves like a random walk, whose multi-step
forecast is a flat line, which the official identical-prediction guard rejects.
This is exactly the failure mode the ADF section predicts.

The side effect noted above — that `d = 0` makes forecasts decay toward the
series mean rather than collapsing to a flat value, giving fewer degenerate
windows — is visible in the same table: the `d = 1` orders are the only ones in
the entire ARMA sweep that were rejected or catastrophically bad.

Order beyond (1,0,1) buys nothing measurable. AR(1) captures essentially all the
usable autocorrelation.

### Fit length — shorter is better, contrary to the original assumption

| fit_days | 30 | 90 | 197 |
|---|---|---|---|
| score (MW) | **44.324** | 44.681 | 44.976 |

Monotonic. The original reasoning — stable AR(1) structure means more data gives
tighter coefficient estimates — does not survive measurement. The dynamics drift
with season and weather regime, so older data is actively misleading rather than
merely redundant. `FIT_STEPS` now defaults to 30 days.

This is the **opposite** of what climatology wants, where longer history wins by
15 MW. The two models estimate different quantities: climatology estimates the
diurnal *level* profile, which is stable; ARMA estimates the *dynamics*, which
are not.

### Warm-up length — no effect at all

| apply_steps | 144 | 288 | 432 | 1008 | 2016 |
|---|---|---|---|---|---|
| score (MW) | 44.9758 | 44.9758 | 44.9758 | 44.9758 | 44.9758 |

Identical to four decimal places across a 14x range; direct comparison of the
forecast arrays shows agreement to 6e-14, i.e. floating point only. With
coefficients frozen, a *stationary* ARMA's Kalman filter converges to steady
state within a few dozen steps and forgets its initial conditions, so earlier
history cannot matter. A non-stationary model would not forget — a further small
confirmation that `d = 0` is correct.

`APPLY_STEPS` now defaults to one day, the shortest measured-equivalent setting.
Measured effect on runtime is modest (roughly 25% faster per window than 2016
steps), and it is purely an inference-speed lever.

### Standing

At **44.324 MW** this is the best-scoring baseline in the suite, ahead of a tuned
GRU (44.537) and a tuned XGBoost (45.445) — while reading only past `Patv`,
where XGBoost reads nine additional sensor channels.