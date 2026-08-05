# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/baseline_models/naive/baseline_naive.py

Description:
    Tier-0 naive benchmark: the seasonal naive forecaster. For each turbine it
    copies a block of recent history forward, tiling it to cover the 48h horizon:

        forecast(h) = Patv(T - LAG + ((h - 1) mod LAG) + 1)

    where T is the forecast origin and LAG is the block length in 10-minute
    steps. There are no parameters and no training data.

    LAG selects a member of the family:
        LAG = 144  daily seasonal naive  -- tomorrow looks like today (default)
        LAG = 288  two-day block         -- the next 48h looks like the last 48h
        LAG = 1    true persistence      -- REJECTED, see below

    Note on true persistence: the classical naive forecast repeats the single
    most recent value for the whole horizon. That produces a constant series for
    every turbine, which the official scorer rejects via is_identical_prediction
    (verified: it returns the failure penalty, not a score). Any LAG >= 2 copies
    a varying block and is therefore scoreable. This is a real constraint of the
    competition metric, not a choice made here.

Example usage:
    # Run from the package root (windevolve_sdwpf/):
    python evaluator.py baseline_models/naive/baseline_naive.py
    # Compare lags without editing the file:
    SDWPF_NAIVE_LAG=288 python evaluator.py baseline_models/naive/baseline_naive.py
"""
import os                         # read the optional LAG override
import numpy as np                # tiling, padding, NaN handling
import pandas as pd               # read the infile CSV

CAPACITY = 134                    # turbines (fallback if absent from envs)
OUTPUT_LEN = 288                  # forecast horizon in 10-min steps (48h)
DAY_LEN = 144                     # steps per day (24h at 10-min resolution)

# Block length copied forward, in 10-minute steps. 144 = daily seasonal naive.
# Override with the SDWPF_NAIVE_LAG environment variable to compare lags.
LAG = int(os.environ.get("SDWPF_NAIVE_LAG", DAY_LEN))


def forecast(envs):
    """
    Description:
        Forecast each turbine's next 48h by copying its most recent LAG steps
        forward, repeating that block as many times as the horizon requires.
    Args:
        envs: settings dict; must contain "path_to_test_x" (the infile path).
              Optional "capacity" and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", CAPACITY)
    output_len = envs.get("output_len", OUTPUT_LEN)
    lag = max(int(LAG), 1)                             # guard against LAG <= 0

    df = pd.read_csv(envs["path_to_test_x"])           # the infile (all turbines)
    total = len(df) // capacity                        # rows per turbine
    patv = np.nan_to_num(df["Patv"].to_numpy(dtype=float), nan=0.0) # convert NaN to 0.0 for the Patv column of the infile

    reps = int(np.ceil(output_len / lag))                 # how many times to repeat the block to cover the horizon, ex. if output_len=288 and lag=144, then reps=2
    preds = np.zeros((capacity, output_len, 1), dtype=float) # allocate the output array ex. (capacity, output_len, 1) = (134, 288, 1)
    for t in range(capacity): # 
        block = patv[t * total:(t + 1) * total][-lag:]  # the most recent LAG steps
        if block.size < lag:                            # pad unusually short history
            block = np.pad(block, (lag - block.size, 0), mode="edge")
        preds[t, :, 0] = np.tile(block, reps)[:output_len]
    return np.clip(preds, 0.0, None)                    # power cannot be negative
