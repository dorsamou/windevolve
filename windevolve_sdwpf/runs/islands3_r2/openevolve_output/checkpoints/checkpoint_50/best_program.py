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
    # Power curve lookup table (25 bins)
    w_idx = feature_cols.index("Wspd")
    bins = np.linspace(0, 25, 26)
    idx = np.digitize(train_series["Wspd"], bins)
    lookup = train_series.groupby(idx)[target_col].mean().reindex(range(27), fill_value=0).values
    
    # Diurnal profile
    train_series = train_series.copy()
    train_series['hour'] = (train_series.index // 6) % 24
    daily = train_series.groupby('hour')[target_col].mean().reindex(range(24), fill_value=0).values
    
    # Forecast components
    curr_w = history_windows[:, -1, w_idx]
    base = lookup[np.clip(np.digitize(curr_w, bins), 0, 26)]
    
    steps = np.arange(output_len)
    h_idx = (steps // 6) % 24
    
    # Blend: Decay momentum, increase reliance on curve + diurnal
    decay = np.exp(-steps / 36.0)
    last_p = history_windows[:, -1, t]
    recent_trend = (history_windows[:, -1, t] - history_windows[:, -12, t]) / 12
    
    trend = recent_trend[:, None] * np.exp(-steps / 12.0)
    preds = (last_p[:, None] * decay) + ((1 - decay) * (0.6 * base[:, None] + 0.4 * daily[h_idx])) + trend
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
