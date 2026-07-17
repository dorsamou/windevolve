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
    # Capture diurnal profile: average power per 15-min step (96 steps/day)
    df_vals = train_series[target_col].values
    cycle = np.array([np.mean(df_vals[i::96]) for i in range(96)])
    diurnal = np.tile(cycle, 3)[:output_len]
    
    # Use last power as a starting point
    last_p = history_windows[:, -1, t][:, None]
    steps = np.arange(1, output_len + 1)
    decay = np.exp(-steps / 72.0)
    
    # Use wind speed relationship as a long-term anchor
    ws_idx = feature_cols.index("Wspd")
    w_train, p_train = train_series["Wspd"].values, train_series[target_col].values
    # Robust scaling for power-wind relationship
    scale = np.nanmedian(p_train / (w_train + 0.5))
    target_p = (history_windows[:, -1, ws_idx][:, None] * scale)
    
    # Blend: persistence, diurnal cycle, and wind-implied target
    preds = (last_p * decay) + (diurnal * 0.3 * (1 - decay)) + (target_p * 0.7 * (1 - decay))
    
    # Add trend adjustment
    recent_trend = (history_windows[:, -1, t] - history_windows[:, -13, t]) / 12
    preds += recent_trend[:, None] * (1 - decay) * 2
    
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
