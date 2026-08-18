# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/arms/karpathy/ws/greedy-cv/forecaster.py

Description:
    Linear shrinkage forecaster for SDWPF 48h-ahead active power. Each turbine's
    forecast is its recent level pulled towards its own climatological mean at a
    rate fitted, per fold, from that fold's training data:

        pred_t(h) = mu_t + beta(h) * (x_now_t - mu_t)

    where mu_t is turbine t's mean power over the fold's training period, x_now_t
    is its mean power over the last hour of the infile, and beta(h) is the
    least-squares regression of power h steps ahead on the current level, pooled
    over all 134 turbines. beta = 1 would be pure persistence, beta = 0 pure
    climatology.

    This replaces the seeded per-turbine GRU baseline (134 independent networks,
    ~1000 lines, five training runs per configuration) after a sequence of
    experiments established that the GRU was not using its extra machinery:

      * Past ~24h its output is a flat per-turbine constant -- 1.4 kW of
        variation across windows against 358 kW in the first hour.
      * Its inner-validation RMSE is ~455 kW against ~525 kW for climatology,
        i.e. it beats predicting the mean over 48h by about 13%.
      * Giving it properly scaled inputs made no difference. The seed divided
        every column by one scalar, so wind speed reached it at 0.016 standard
        deviations and blade pitch at 0.0018; standardising each column
        separately left the fit identical to within 0.7% in kW on every fold.

    Fitted on the same training data, the shrinkage above reproduces that
    behaviour directly: beta falls from 0.94 at 10 minutes to 0.11 at 12 hours
    and 0.00 by 36 hours, with the regression's R^2 down to 0.01 by 12 hours.
    Essentially all 48h-horizon skill available from on-site history lives in
    the first few hours, and the rest of the horizon is climatology.

Citation:
    Task and metric from the Baidu KDD Cup 2022 SDWPF competition.
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.
"""
import numpy as np                            # array math
import pandas as pd                           # CSV reading

DAY_LEN = 144            # 10-minute observations per day
RECENT_STEPS = 6         # steps averaged into the "current level" (1 hour)

# Fitted coefficients, keyed by (fold, training csv) -- NEVER a bare global.
# All five folds are scored in ONE process, so a fold-blind cache would serve
# fold 1's coefficients to every later fold and the numbers would be silently
# wrong while still looking reasonable.
_FIT_CACHE = {}


def _turbine_matrix(patv, capacity):
    """
    Description:
        Reshape a turbine-major Patv column into one row per turbine, with
        missing readings filled as zero -- the same convention the official
        loader applies before modelling.
    Args:
        patv: 1-D array of Patv for all turbines, stacked turbine-major
        capacity: number of turbines (134)
    Returns:
        A (capacity, steps_per_turbine) float array.
    """
    per = len(patv) // capacity
    return np.nan_to_num(patv[:capacity * per].astype(np.float64),
                         nan=0.0).reshape(capacity, per)


def _rolling_level(dev, steps):
    """
    Description:
        Trailing mean of `steps` observations, as a sliding window over each
        turbine's deviation series. Column i is the mean of deviations at times
        i .. i+steps-1, so it represents the level known at time i+steps-1.

        Averaging an hour rather than reading a single instant trades a small
        lag for a large reduction in sensor noise; at 10-minute resolution a
        lone reading is a poor estimate of the state that persists.
    Args:
        dev: (capacity, steps_per_turbine) deviations from each turbine's mean
        steps: window length (RECENT_STEPS)
    Returns:
        A (capacity, steps_per_turbine - steps + 1) float array.
    """
    csum = np.cumsum(np.concatenate(
        [np.zeros((dev.shape[0], 1)), dev], axis=1), axis=1)
    return (csum[:, steps:] - csum[:, :-steps]) / float(steps)


def _fit(envs, capacity, output_len):
    """
    Description:
        Fit the shrinkage curve on this fold's training file: each turbine's
        climatological mean, and beta(h) for every horizon step.

        beta(h) is a single pooled coefficient rather than 134 separate ones.
        Each turbine contributes its own deviations, so per-turbine level is
        already removed; what is shared is the DECAY, which is a property of
        the weather rather than of an individual machine, and pooling it over
        134 turbines is what makes the far end of the curve stable.
    Args:
        envs: the settings dict passed to forecast()
        capacity: number of turbines (134)
        output_len: forecast horizon in steps (288)
    Returns:
        A (mu, beta) pair: per-turbine means (capacity,) and coefficients
        (capacity, output_len).
    """
    key = (envs.get("fold"), envs.get("path_to_train"))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]

    patv = pd.read_csv(envs["path_to_train"], usecols=["Patv"])["Patv"].values
    power = _turbine_matrix(patv, capacity)
    mu = power.mean(axis=1)
    dev = power - mu[:, None]

    level = _rolling_level(dev, RECENT_STEPS)        # known at t = i + steps - 1
    n_steps = dev.shape[1]
    # One coefficient per TURBINE per horizon, rather than one pooled curve.
    # Each turbine has 13k-31k observations behind every coefficient, so these
    # are well determined; the question is whether turbines genuinely differ in
    # how fast they forget their state, or whether 134 separate curves just add
    # variance to a quantity that is really a property of the weather.
    beta = np.zeros((dev.shape[0], output_len), dtype=np.float64)
    for h in range(1, output_len + 1):
        width = n_steps - h - RECENT_STEPS + 1       # usable (level, target) pairs
        if width <= 0:
            break                                    # leaves beta at 0 = climatology
        x = level[:, :width]
        y = dev[:, RECENT_STEPS - 1 + h:RECENT_STEPS - 1 + h + width]
        denom = np.sum(x * x, axis=1)
        beta[:, h - 1] = np.where(denom > 0, np.sum(x * y, axis=1) / np.maximum(denom, 1e-9), 0.0)

    mean_beta = beta.mean(axis=0)
    print("[fit] fold {}: {} turbines, per-turbine beta; mean 10min {:.3f} "
          "6h {:.3f} 24h {:.3f} (spread across turbines at 6h {:.3f})".format(
              envs.get("fold"), capacity, mean_beta[0],
              mean_beta[min(35, output_len - 1)],
              mean_beta[min(143, output_len - 1)],
              beta[:, min(35, output_len - 1)].std()), flush=True)
    _FIT_CACHE[key] = (mu, beta)
    return mu, beta


def forecast(envs):
    """
    Description:
        Produce a 48h active-power forecast for every turbine by shrinking each
        one's current level towards its climatological mean along the fitted
        beta curve. This is the program contract used by evaluator.py.
    Args:
        envs: settings dict; must contain "path_to_test_x" (the infile) and
              "path_to_train" (this fold's training csv). Optional "capacity"
              and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", 134)
    output_len = envs.get("output_len", 288)
    mu, beta = _fit(envs, capacity, output_len)

    patv = pd.read_csv(envs["path_to_test_x"], usecols=["Patv"])["Patv"].values
    per = len(patv) // capacity
    recent = patv[:capacity * per].astype(np.float64).reshape(
        capacity, per)[:, -RECENT_STEPS:]

    # A turbine whose last hour is entirely missing has no usable current
    # level, so it falls back to climatology rather than to nan_to_num's zero --
    # which would otherwise read as "becalmed" and drag the forecast down.
    with np.errstate(invalid="ignore"):
        level = np.nanmean(recent, axis=1)
    level = np.where(np.isnan(level), mu, level)

    predictions = mu[:, None] + (level - mu)[:, None] * beta
    return predictions[:, :, None]
