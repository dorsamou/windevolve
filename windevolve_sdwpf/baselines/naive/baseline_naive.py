# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/baselines/naive/baseline_naive.py

Description:
    Tier-0 naive benchmark: the seasonal naive forecaster. For each turbine it
    copies a block of recent history forward, tiling it to cover the 48h horizon:

        forecast(h) = Patv(T - LAG + ((h - 1) mod LAG) + 1)

    where T is the forecast origin and LAG is the block length in 10-minute
    steps. There is no training and no fitted state.

    WHAT LAG ACTUALLY CONTROLS. Two regimes, and they are not the same idea:

      LAG < OUTPUT_LEN   the block is shorter than the horizon and is REPEATED.
                         LAG = 144 copies yesterday twice.
      LAG >= OUTPUT_LEN  the block is longer than the horizon, reps collapses to
                         1, and only the block's FIRST OUTPUT_LEN steps are used.
                         This is a TIME SHIFT, not a longer memory:
                           LAG = 288   the most recent 48h
                           LAG = 432   the 48h ending one day ago
                           LAG = 2016  the oldest 48h in the 14-day input
                         Anything above 288 therefore forecasts with older data,
                         which is a different model, not a bigger one.

    DEFAULT: LAG = 432, the best-scoring setting measured on the 197/48
    validation split (59.42 MW, against 68.65 for LAG = 288 and outright
    rejection for LAG = 144). See docs/baseline_report.md section 4.2.

    WHY NOT LAG = 144. Daily seasonal naive is UNSCOREABLE on SDWPF. Copying the
    previous 24h forward emits an all-zero forecast for any turbine that was
    parked that day, and the official is_zero_prediction guard rejects the whole
    window once too many turbines are all-zero. Measured on the 197/48 validation
    split, LAG = 144 was rejected with 66.42% of turbines all-zero; 53% of
    turbine-windows in that split start parked, so this reflects the farm's
    operating profile rather than an unlucky window. Six of ten swept lags fail
    this way. See docs/baseline_report.md section 4.2.

    LAG = 1 is likewise rejected, by a different guard: repeating a single value
    gives a constant series per turbine, which is_identical_prediction refuses.
    Both are constraints of the competition metric, not choices made here.

    CAVEAT ON THE DEFAULT. Two effects are confounded in the LAG = 432 result and
    this model cannot separate them: an older block sits differently against the
    diurnal cycle, AND it is more likely to contain non-zero readings, which is
    part of why it survives the zero guard. The margin is also specific to which
    turbines were parked during these particular 48 days. Report it as the
    best-on-validation setting rather than as evidence about wind persistence.

    The same property makes it the more robust choice: LAG = 288 copies the most
    recent 48h and so emits zeros for recently-parked turbines, leaving it
    exposed to the guard that rejected LAG = 144. LAG = 432 draws on older
    history and is correspondingly less likely to fail to produce a score at
    all.

Example usage:
    # Run from the package root (windevolve_sdwpf/):
    python evaluation/evaluator.py baselines/naive/baseline_naive.py
    # Compare lags without editing the file:
    SDWPF_NAIVE_LAG=432 python evaluation/evaluator.py baselines/naive/baseline_naive.py
"""
import os                         # read the optional LAG override
import numpy as np                # tiling, padding, NaN handling
import pandas as pd               # read the infile CSV

CAPACITY = 134                    # turbines (fallback if absent from envs)
OUTPUT_LEN = 288                  # forecast horizon in 10-min steps (48h)
DAY_LEN = 144                     # steps per day (24h at 10-min resolution)

# Block length copied forward, in 10-minute steps. 432 = the 48h window ending
# one day ago -- best on the validation split, and NOT 144 (daily seasonal
# naive), which the official zero-prediction guard rejects outright on this
# dataset. See the module docstring. Override with SDWPF_NAIVE_LAG.
LAG = int(os.environ.get("SDWPF_NAIVE_LAG", 3 * DAY_LEN))


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

    # reps = 1 whenever lag >= output_len, so the tile is a no-op and the slice
    # below keeps the block's FIRST output_len steps -- i.e. the oldest part of
    # it. That is what makes large lags a time shift rather than a longer memory.
    reps = int(np.ceil(output_len / lag))
    preds = np.zeros((capacity, output_len, 1), dtype=float)
    for t in range(capacity):
        block = patv[t * total:(t + 1) * total][-lag:]  # the most recent LAG steps
        if block.size < lag:                            # pad unusually short history
            block = np.pad(block, (lag - block.size, 0), mode="edge")
        preds[t, :, 0] = np.tile(block, reps)[:output_len]
    return np.clip(preds, 0.0, None)                    # power cannot be negative