# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/baselines/classical/baseline_climatology.py

Description:
    Diurnal climatology. Averages the last N_DAYS of the window's own history by
    TIME OF DAY into a 24h profile, then tiles it twice to cover the 48h horizon.
    Uses no recent state -- it answers "what does this turbine usually do at this
    hour". No fitting and no artifact: everything comes from the input window.


    Sweep results and the reasoning behind every default are in
    baseline_climatology.md.

Citation:
    Contract and scoring follow the official Baidu KDD Cup 2022 SDWPF evaluator
    (PaddlePaddle/PaddleSpatial, apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022).

Example usage:
    python evaluation/evaluator.py baselines/classical/baseline_climatology.py
"""
import numpy as np
import pandas as pd

CAPACITY = 134                    # turbines (fallback if absent from envs)
OUTPUT_LEN = 288                  # forecast horizon in 10-min steps
DAY_LEN = 144                     # steps per day (24h at 10-min resolution)

# Past days averaged into the profile. None = all 14. THE SWEPT AXIS. See the .md.
N_DAYS = 14


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

    reps = int(np.ceil(output_len / DAY_LEN))
    for t in range(capacity):
        block = patv[t * total:(t + 1) * total][:n_days * DAY_LEN]
        grid = block.reshape(n_days, DAY_LEN)         # rows = days, cols = time of day
        grid = grid[-keep:]                           # most recent `keep` days only
        with np.errstate(invalid="ignore"):
            # NaN-aware: a missing reading drops out of its time-of-day column
            # rather than biasing that slot toward zero.
            profile = np.nanmean(grid, axis=0)
        profile = np.nan_to_num(profile, nan=0.0)
        preds[t, :, 0] = np.tile(profile, reps)[:output_len]
    return np.clip(preds, 0.0, None)