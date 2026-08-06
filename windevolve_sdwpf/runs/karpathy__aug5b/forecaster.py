# -*- coding: utf-8 -*-
"""
path: arms/karpathy/workspace/forecaster.py

Description:
    The forecaster. THIS IS THE FILE YOU EDIT. It implements the required
    forecast(envs) contract. Strategy: persistence has real skill only for the
    first few hours, after which the best no-NWP forecast collapses to the
    turbine's own climatological level. So each turbine's prediction decays
    exponentially from a recent-history anchor toward its 14-day mean.

Scoring:
    python ../../../evaluation/evaluator.py forecaster.py > run.log 2>&1
    Lower competition_score is better. See ../program.md for the full loop.

Citation:
    Program contract and scoring follow the official Baidu KDD Cup 2022 SDWPF
    evaluator (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    >>> import forecaster
    >>> preds = forecaster.forecast({"path_to_test_x":
    ...     "../../../evaluation/splits/val_windows/infile/0001in.csv",
    ...     "capacity": 134, "output_len": 288})
    >>> preds.shape                      # (134, 288, 1)
"""

import numpy as np                # array math
import pandas as pd               # read the infile CSV

CAPACITY = 134                    # number of turbines (fallback if not in envs)
OUTPUT_LEN = 288                  # forecast horizon in 10-min steps (fallback)

ANCHOR_STEPS = 1                  # recent window used as the persistence anchor
TAU_STEPS = 36                    # e-folding decay toward climatology (6h)
SHRINK = 0.85                     # the metric is half MAE, which prefers a level
                                  # below the conditional mean on this skewed target
LONG_WEIGHT = 0.5                 # weight on the long-run level vs the 14-day one

_CACHE = {}                       # long-run levels, computed once per process


def forecast(envs):
    """
    Description:
        Forecast each turbine as an exponential decay from its recent power
        level toward its own climatological mean over the supplied history.
    Args:
        envs: settings dict; must contain "path_to_test_x" (the infile path).
              Optional "capacity" and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", CAPACITY)          # turbines to predict
    output_len = envs.get("output_len", OUTPUT_LEN)    # horizon length

    df = pd.read_csv(envs["path_to_test_x"])           # the infile (all turbines)
    total = len(df) // capacity                        # history steps per turbine
    # (capacity, total) view of the turbine-major stack; NaN = missing reading.
    patv = df["Patv"].to_numpy(dtype=float)[:capacity * total].reshape(capacity, total)

    # Climatology and anchor ignore missing readings rather than counting them
    # as zero, which would bias both levels downward.
    clim = _nanmean_rows(patv)                         # per-turbine 14-day level
    anchor = _nanmean_rows(patv[:, -ANCHOR_STEPS:])    # per-turbine recent level

    # Fourteen days estimates a turbine's own level noisily. Its multi-month
    # level from the training split is a much steadier estimate of the same
    # quantity, so blend the two; the pair beats either alone.
    long_level = _long_run_level(envs.get("path_to_train"), capacity)
    if long_level is not None:
        clim = (1.0 - LONG_WEIGHT) * clim + LONG_WEIGHT * long_level

    # Weight on the anchor decays with lead time; climatology takes over.
    w = np.exp(-np.arange(output_len) / TAU_STEPS)
    preds = anchor[:, None] * w[None, :] + clim[:, None] * (1.0 - w[None, :])
    return np.clip(preds * SHRINK, 0.0, None)[:, :, None]   # power is non-negative


def _long_run_level(path, capacity):
    """
    Per-turbine mean Patv over the training split, computed once and cached.
    Returns None if no training path was supplied or it cannot be read, so the
    forecast falls back to the 14-day level alone.
    """
    if not path:
        return None
    key = ("long", path, capacity)
    if key not in _CACHE:
        try:
            col = pd.read_csv(path, usecols=["Patv"])["Patv"].to_numpy(dtype=float)
            rows = (len(col) // capacity) * capacity
            _CACHE[key] = _nanmean_rows(col[:rows].reshape(capacity, -1))
        except Exception:
            _CACHE[key] = None
    return _CACHE[key]


def _nanmean_rows(block):
    """Row-wise mean ignoring NaN; rows that are entirely NaN become 0."""
    valid = np.isfinite(block)
    counts = valid.sum(axis=1)
    totals = np.where(valid, block, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), 0.0)
