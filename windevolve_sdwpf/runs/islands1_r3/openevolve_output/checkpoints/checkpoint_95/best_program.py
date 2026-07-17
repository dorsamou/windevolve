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
    t, w = feature_cols.index(target_col), feature_cols.index("Wspd")
    # Estimate expected power via wind speed binning (median for robustness)
    avg_wspd = np.mean(history_windows[:, -36:, w], axis=1)
    bins = np.linspace(0, 25, 25)
    curve = train_series.groupby(np.digitize(train_series["Wspd"], bins))[target_col].median()
    target_val = np.array([curve.get(i, train_series[target_col].median()) for i in np.digitize(avg_wspd, bins)])
    
    # Incorporate diurnal cycle
    diurnal = train_series.groupby(np.arange(len(train_series)) % 144)[target_col].median().values
    diurnal_seq = np.tile(diurnal, 3)[:output_len]
    
    # Target is a blend of wind-based expectation and diurnal signal
    final_target = 0.4 * target_val[:, None] + 0.6 * diurnal_seq
    
    # Adaptive decay: steeper decay for higher volatility
    w_vol = np.std(history_windows[:, -36:, w], axis=1) + 0.1
    decay_rate = 0.015 + 0.01 * (w_vol / (w_vol + 2.0))
    decay = np.exp(-decay_rate[:, None] * np.arange(output_len))
    
    # Add trend momentum based on the last 12 steps
    vol = np.std(history_windows[:, -24:, t], axis=1) + 1.0
    trend = (history_windows[:, -1, t] - np.mean(history_windows[:, -12:, t], axis=1)) * (50 / (50 + vol))
    trend_adj = trend[:, None] * np.exp(-0.06 * np.arange(output_len))
    
    preds = history_windows[:, -1, t][:, None] * decay + final_target * (1 - decay) + trend_adj
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
