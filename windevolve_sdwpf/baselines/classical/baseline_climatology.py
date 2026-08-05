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
    reps = int(np.ceil(output_len / DAY_LEN))
    for t in range(capacity):
        block = patv[t * total:(t + 1) * total][:n_days * DAY_LEN]
        grid = block.reshape(n_days, DAY_LEN)         # rows = days, cols = time of day
        with np.errstate(invalid="ignore"):
            profile = np.nanmean(grid, axis=0)        # mean power at each time of day
        profile = np.nan_to_num(profile, nan=0.0)
        preds[t, :, 0] = np.tile(profile, reps)[:output_len]
    return np.clip(preds, 0.0, None)
