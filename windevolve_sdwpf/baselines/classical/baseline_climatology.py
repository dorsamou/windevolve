# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/baseline_models/classical/baseline_climatology.py

Description:
    Tier-1 statistical baseline: diurnal climatology. For each turbine it
    averages the 14 days of history by TIME OF DAY, producing a mean 24h power
    profile, then repeats that profile twice to cover the 48h horizon. It uses
    no recent-state information at all -- only the average daily shape -- so it
    answers "what does this turbine usually do at this hour".

    Note: a FLAT per-turbine mean (one number repeated 288 times) is REJECTED by
    the official scorer's is_identical_prediction guard, because every turbine
    would emit a constant series. Averaging by time of day keeps the forecast
    varying across the horizon and therefore scoreable.

Citation:
    Program contract and scoring follow the official Baidu KDD Cup 2022 SDWPF
    evaluator (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    python evaluator.py baseline_models/classical/baseline_climatology.py
"""
import numpy as np                # array reshaping and nan-aware means
import pandas as pd               # read the infile CSV

CAPACITY = 134                    # turbines (fallback if absent from envs)
OUTPUT_LEN = 288                  # forecast horizon in 10-min steps
DAY_LEN = 144                     # steps per day (24h at 10-min resolution)

# How many of the most recent history days enter the profile. None = all of
# them, which is the original behaviour and stays the default. Fewer days makes
# the profile more responsive to the current weather regime and noisier; more
# days makes it a stabler long-run average. This is the model's only real
# hyperparameter, so it is the axis analysis/sweep.py varies.
N_DAYS = None

# How the days are combined at each time of day: "mean" (original) or "median".
# SDWPF power is heavily zero-inflated -- curtailment, icing, maintenance -- so
# a handful of parked days pull a mean profile down while leaving a median
# profile untouched. Which is preferable is an empirical question, not an
# obvious one, which is why it is swept rather than chosen here.
AGG = "mean"

# Half-life, in days, of an exponential weighting over the history days: the
# day before the origin gets weight 1, a day HALFLIFE_DAYS earlier gets 0.5, and
# so on. None = uniform weighting, the original behaviour.
#
# This is the better-posed version of the "14 days or 5 days" question. A hard
# N_DAYS cutoff throws away day 6 entirely while trusting day 5 completely,
# which is not a belief anyone holds; a half-life says recent days matter more
# and older ones fade. N_DAYS is kept because it is the interpretable version
# and answers the question as asked. Only applies when AGG is "mean".
HALFLIFE_DAYS = None

# Width, in 10-minute steps, of a circular moving average applied to the finished
# 144-step profile. 0 = off. Averaging few days leaves a noisy profile; smoothing
# trades that noise for a little blur, and the two knobs interact -- which is why
# the sweep crosses them rather than varying each alone.
SMOOTH_STEPS = 0


def forecast(envs):
    """
    Description:
        Forecast each turbine's next 48h as its average daily power profile,
        computed by time of day over the available history and tiled twice.
    Args:
        envs: settings dict; must contain "path_to_test_x". Optional "capacity"
              and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", CAPACITY)
    output_len = envs.get("output_len", OUTPUT_LEN)

    df = pd.read_csv(envs["path_to_test_x"])
    total = len(df) // capacity                       # history steps per turbine
    patv = df["Patv"].to_numpy(dtype=float)           # NaNs kept for nan-aware mean

    preds = np.zeros((capacity, output_len, 1), dtype=float)
    n_days = max(total // DAY_LEN, 1)                 # whole days of history
    # Keep only the most recent N_DAYS of that history, if asked.
    keep = n_days if N_DAYS is None else max(1, min(int(N_DAYS), n_days))
    use_median = str(AGG).lower() == "median"

    # Exponential day weights, newest last. Uniform when HALFLIFE_DAYS is None.
    if HALFLIFE_DAYS is None or use_median:
        weights = None
    else:
        age = np.arange(keep - 1, -1, -1, dtype=float)   # 0 = most recent day
        weights = 0.5 ** (age / float(HALFLIFE_DAYS))

    reps = int(np.ceil(output_len / DAY_LEN))
    for t in range(capacity):
        block = patv[t * total:(t + 1) * total][:n_days * DAY_LEN]
        grid = block.reshape(n_days, DAY_LEN)         # rows = days, cols = time of day
        grid = grid[-keep:]                           # most recent `keep` days only
        with np.errstate(invalid="ignore"):
            if use_median:
                profile = np.nanmedian(grid, axis=0)
            elif weights is None:
                profile = np.nanmean(grid, axis=0)
            else:
                # NaN-aware weighted mean: missing readings drop out of both the
                # numerator and the weight total, so a gap does not bias the day
                # it falls in toward zero.
                ok = ~np.isnan(grid)
                wgt = np.where(ok, weights[:, None], 0.0)
                denom = wgt.sum(axis=0)
                profile = np.where(denom > 0,
                                   np.nansum(np.where(ok, grid, 0.0) * wgt, axis=0)
                                   / np.maximum(denom, 1e-12),
                                   np.nan)
        profile = np.nan_to_num(profile, nan=0.0)
        if SMOOTH_STEPS and int(SMOOTH_STEPS) > 1:
            # Circular, because time of day wraps: 23:50 is adjacent to 00:00.
            k = int(SMOOTH_STEPS)
            kern = np.ones(k) / k
            profile = np.convolve(np.tile(profile, 3), kern, mode="same")[
                DAY_LEN:2 * DAY_LEN]
        preds[t, :, 0] = np.tile(profile, reps)[:output_len]
    return np.clip(preds, 0.0, None)
