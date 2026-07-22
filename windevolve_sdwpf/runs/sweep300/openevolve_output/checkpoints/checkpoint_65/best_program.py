# EVOLVE-BLOCK-START
"""
Seed forecaster for SDWPF, KDD Cup 2022 protocol (per-turbine, applied to all 134).

Task: at each origin, predict the next output_len=288 values of active power
(Patv, kW) -- 48 hours ahead -- using ONLY the input_len=144-step history window.
No future weather. The SAME function is applied to every turbine; the farm score
sums the per-turbine scores, so this function must generalize across turbines.

Inputs:
  train_series    : DataFrame of ONE turbine's training rows (columns=feature_cols).
  history_windows : np.ndarray (n_origins, input_len, n_features). Patv is at
                    feature_cols.index(target_col). Gap-filled (finite).
  feature_cols, target_col, input_len, output_len.
Return:
  np.ndarray (n_origins, output_len), Patv forecasts in kW.

Beat the persistence seed: use wind speed, recent-power trend/mean, diurnal
position, or fit a model on train_series. Fit on train_series only; no future data.
Return shape exactly (n_origins, output_len); power >= 0.
"""
import numpy as np


def forecast(train_series, history_windows, feature_cols, target_col, input_len, output_len):
    t_idx = feature_cols.index(target_col)
    w_idx = feature_cols.index("Wspd")
    # Use 1h mean as a responsive starting state
    start = np.mean(history_windows[:, -12:, t_idx], axis=1)
    # Use 6h mean as a more stable trend anchor
    trend = np.mean(history_windows[:, -72:, t_idx], axis=1)
    
    # Revert to median climatology
    long_term = train_series[target_col].median()
    
    # Decay toward median, using a power law that stays near current state longer
    t = np.linspace(0, 1, output_len)
    # Combine start and trend for initial state, decay toward long_term
    init_state = 0.7 * start + 0.3 * trend
    preds = init_state[:, None] * (1 - t**0.4) + long_term * (t**0.4)
    
    return np.clip(preds, 0.0, train_series[target_col].quantile(0.99))
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import pandas as pd
    cols = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
    hw = np.random.RandomState(0).rand(8, 144, len(cols)) * 100
    tr = pd.DataFrame(np.random.rand(500, len(cols)) * 100, columns=cols)
    out = forecast(tr, hw, cols, "Patv", 144, 288)
    # checks the shape of the output 
    print("shape:", out.shape, "| expected:", (8, 288))
