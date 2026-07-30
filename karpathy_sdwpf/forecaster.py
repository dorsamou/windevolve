# -*- coding: utf-8 -*-
"""
path: karpathy_sdwpf/forecaster.py

Description:
    THE ONLY FILE THE AGENT EDITS.

    Seeded as a byte-for-byte behavioural copy of
    ../windevolve_sdwpf/initial_program.py so that both harness arms start from
    an identical program. Implements the required forecast(envs) contract with a
    last-2-days persistence strategy: for each turbine it repeats the most recent
    288 recorded Patv values as the 48h forecast. Persistence is valid (it varies
    over the horizon, so it passes the zero/identical guards) and leaves clear
    headroom for the loop to improve.

    Everything here is fair game: features, model, fitting, blending, per-turbine
    handling. The contract and the constraints in program.md are not.

    PERFORMANCE CONTRACT: forecast() is called once per window (40 times per
    evaluation on the validation split). Any expensive work -- reading
    path_to_train, fitting a model, loading checkpoints -- MUST be done once and
    cached at module level, not on every call.

Citation:
    Program contract and scoring follow the official Baidu KDD Cup 2022 SDWPF
    evaluator (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    >>> import forecaster
    >>> preds = forecaster.forecast({"path_to_test_x":
    ...     "val_windows/infile/0001in.csv", "capacity": 134, "output_len": 288})
    >>> preds.shape                      # (134, 288, 1)
"""
import numpy as np                # array construction / padding / NaN handling
import pandas as pd               # read the infile CSV

CAPACITY = 134                    # number of turbines (fallback if not in envs)
OUTPUT_LEN = 288                  # forecast horizon in 10-min steps (fallback)

# ---------------------------------------------------------------------------
# Module-level cache. Anything fitted on envs["path_to_train"] belongs here so
# it happens once per evaluation rather than once per window.
#
#   _MODEL = None
#
#   def _get_model(envs):
#       global _MODEL
#       if _MODEL is None:
#           _MODEL = _fit(envs["path_to_train"])
#       return _MODEL
# ---------------------------------------------------------------------------


def forecast(envs):
    """
    Description:
        Produce a 48h active-power forecast for every turbine by persistence:
        repeat each turbine's last output_len recorded Patv values.
    Args:
        envs: settings dict; must contain "path_to_test_x" (the infile path).
              Also carries "path_to_train" (training CSV, already truncated to
              exclude the validation period) and "path_to_locations" (turbine
              x/y). Optional "capacity" and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", CAPACITY)          # turbines to predict
    output_len = envs.get("output_len", OUTPUT_LEN)    # horizon length

    df = pd.read_csv(envs["path_to_test_x"])           # the infile (all turbines)
    total = len(df) // capacity                        # history steps per turbine
    # Pull Patv for all turbines and replace missing readings with 0.
    patv = np.nan_to_num(df["Patv"].to_numpy(dtype=float), nan=0.0)

    preds = np.zeros((capacity, output_len, 1), dtype=float)
    for t in range(capacity):
        block = patv[t * total:(t + 1) * total]        # this turbine's history
        tail = block[-output_len:]                     # its most recent horizon
        if tail.size < output_len:                     # pad short histories
            tail = np.pad(tail, (output_len - tail.size, 0), mode="edge")
        preds[t, :, 0] = tail
    return preds
