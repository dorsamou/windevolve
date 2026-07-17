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
    t = feature_cols.index(target_col)
    # Use training data to estimate a local power-wind speed relationship
    w_idx = feature_cols.index("Wspd")
    w_train = train_series["Wspd"].values
    p_train = train_series[target_col].values
    # Simple linear proxy: expected power at current wind
    alpha = np.sum(w_train * p_train) / (np.sum(w_train**2) + 1e-3)
    
    # Anchor forecast: decay toward wind-implied power, not just diurnal mean
    curr_w = history_windows[:, -1, w_idx][:, None]
    target_p = curr_w * alpha
    
    last_p = history_windows[:, -1, t][:, None]
    steps = np.arange(1, output_len + 1)
    decay = np.exp(-steps / 96.0)
    
    # Capture diurnal profile: average power per 15-min step (96 steps/day)
    df_vals = train_series[target_col].values
    cycle = np.array([np.mean(df_vals[i::96]) for i in range(96)])
    diurnal = np.tile(cycle, 3)[:output_len]
    
    # Forecast: Blend persistence, wind-implied target, and diurnal profile
    preds = (last_p * decay) + (target_p * 0.4 * (1 - decay)) + (diurnal * 0.6 * (1 - decay))
    
    # Add momentum: damp trend more aggressively
    recent_trend = (history_windows[:, -1, t] - history_windows[:, -5, t]) / 4
    preds += recent_trend[:, None] * (1 - decay) * 1.5
    return np.clip(preds, 0.0, None)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import pandas as pd
    cols = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
    hw = np.random.RandomState(0).rand(8, 144, len(cols)) * 100
    tr = pd.DataFrame(np.random.rand(500, len(cols)) * 100, columns=cols)
    out = forecast(tr, hw, cols, "Patv", 144, 288)
    # checks the shape of the output 
    print("shape:", out.shape, "| expected:", (8, 288))
