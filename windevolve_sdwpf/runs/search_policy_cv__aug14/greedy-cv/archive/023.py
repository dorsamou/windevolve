# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/arms/karpathy/ws/greedy-cv/forecaster.py

Description:
    Nonparametric shrinkage forecaster for SDWPF 48h-ahead active power. Each
    turbine's forecast is its climatological mean plus the average future
    deviation historically observed from a similar current level:

        pred_t(h) = mu_t + g_h(x_now_t - mu_t)

    where g_h is estimated by binning the current-level deviation into quantile
    bins and averaging the realised deviation h steps later within each bin,
    pooled over all 134 turbines and interpolated between bin centres.

    This is the linear shrinkage model with its straight line replaced by a free
    curve. The relationship is close to linear but not exactly: at the 6h
    horizon the observed conditional mean sits ~27 kW below a straight fit at
    the bottom of the level range and ~19 kW below it at the top, against
    predictions of 100-235 kW, so the true curve is compressed at both
    extremes. That is physically expected -- power is bounded below by zero and
    above by the rated 1500 kW, so a turbine already near either limit cannot
    move as far as a linear rule extrapolates.

    Each bin holds ~70k observations at every horizon, so the curve is well
    determined; the risk here is not sampling noise but the same
    non-stationarity that has punished added flexibility elsewhere in this run.

Citation:
    Task and metric from the Baidu KDD Cup 2022 SDWPF competition.
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.
"""
import numpy as np                            # array math
import pandas as pd                           # CSV reading

DAY_LEN = 144            # 10-minute observations per day
RECENT_STEPS = 6         # steps averaged into the "current level" (1 hour)
N_BINS = 24              # quantile bins of current level (~70k samples each)

# Fitted curves, keyed by (fold, training csv) -- NEVER a bare global. All five
# folds are scored in ONE process, so a fold-blind cache would serve fold 1's
# curve to every later fold and the numbers would be silently wrong.
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
        Trailing mean of `steps` observations as a sliding window over each
        turbine's deviation series. Column i is the mean over times
        i .. i+steps-1, i.e. the level known at time i+steps-1.
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
        Fit the response curves on this fold's training file: each turbine's
        climatological mean, the bin centres, and g_h for every horizon step.

        Bins are quantiles of the pooled level deviation, so every bin carries a
        similar number of observations rather than a similar width -- the level
        distribution is heavily concentrated near zero, and equal-width bins
        would put almost everything in one cell and leave the informative tails
        estimated from a handful of points.
    Args:
        envs: the settings dict passed to forecast()
        capacity: number of turbines (134)
        output_len: forecast horizon in steps (288)
    Returns:
        A (mu, centres, curves) triple: per-turbine means (capacity,), bin
        centres (n_bins,), and the response curves (n_bins, output_len).
    """
    key = (envs.get("fold"), envs.get("path_to_train"))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]

    patv = pd.read_csv(envs["path_to_train"], usecols=["Patv"])["Patv"].values
    power = _turbine_matrix(patv, capacity)
    mu = power.mean(axis=1)
    dev = power - mu[:, None]

    level = _rolling_level(dev, RECENT_STEPS)
    offset = RECENT_STEPS - 1
    width = dev.shape[1] - offset - output_len       # positions with a full future
    if width <= 0:
        raise ValueError("training file too short for a {}-step horizon".format(output_len))

    x = level[:, :width].ravel()
    edges = np.quantile(x, np.linspace(0.0, 1.0, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N_BINS - 1)
    counts = np.maximum(np.bincount(index, minlength=N_BINS), 1)
    centres = np.bincount(index, weights=x, minlength=N_BINS) / counts

    curves = np.zeros((N_BINS, output_len), dtype=np.float64)
    for h in range(1, output_len + 1):
        target = dev[:, offset + h:offset + h + width].ravel()
        curves[:, h - 1] = np.bincount(index, weights=target, minlength=N_BINS) / counts

    order = np.argsort(centres)                      # np.interp needs sorted x
    centres, curves = centres[order], curves[order]
    print("[fit] fold {}: {} bins over {} positions; response at 6h for the "
          "lowest/median/highest bin {:.0f} / {:.0f} / {:.0f} kW".format(
              envs.get("fold"), N_BINS, width * capacity,
              curves[0, min(35, output_len - 1)],
              curves[N_BINS // 2, min(35, output_len - 1)],
              curves[-1, min(35, output_len - 1)]), flush=True)
    _FIT_CACHE[key] = (mu, centres, curves)
    return mu, centres, curves


def forecast(envs):
    """
    Description:
        Produce a 48h active-power forecast for every turbine by looking its
        current level up on the fitted response curve for each horizon step.
        This is the program contract used by evaluator.py.
    Args:
        envs: settings dict; must contain "path_to_test_x" (the infile) and
              "path_to_train" (this fold's training csv). Optional "capacity"
              and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", 134)
    output_len = envs.get("output_len", 288)
    mu, centres, curves = _fit(envs, capacity, output_len)

    patv = pd.read_csv(envs["path_to_test_x"], usecols=["Patv"])["Patv"].values
    per = len(patv) // capacity
    recent = patv[:capacity * per].astype(np.float64).reshape(
        capacity, per)[:, -RECENT_STEPS:]

    # A turbine whose last hour is entirely missing has no usable current level,
    # so it falls back to climatology rather than to a zero that would read as
    # "becalmed" and drag the forecast down.
    with np.errstate(invalid="ignore"):
        level = np.nanmean(recent, axis=1)
    level = np.where(np.isnan(level), mu, level) - mu

    # np.interp clamps outside the fitted range, so a level beyond anything seen
    # in training gets the end bin's response rather than a linear extrapolation.
    predictions = np.empty((capacity, output_len), dtype=np.float64)
    for h in range(output_len):
        predictions[:, h] = np.interp(level, centres, curves[:, h])
    predictions += mu[:, None]
    return predictions[:, :, None]
