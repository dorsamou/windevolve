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
    # Extract diurnal profile (144 steps = 24h)
    cycle = train_series[target_col].values.reshape(-1, 144).mean(axis=0)
    cycle_preds = np.tile(cycle, output_len // 144 + 1)[:output_len]
    
    # Estimate power based on recent wind speed
    w = feature_cols.index("Wspd")
    bins = np.linspace(0, 20, 10)
    bin_idx = np.digitize(train_series["Wspd"], bins)
    means = [train_series[target_col][bin_idx == i].mean() for i in range(len(bins) + 1)]
    steady = np.interp(history_windows[:, -1, w], bins, np.nan_to_num(means, nan=0)[:len(bins)])
    
    # Momentum: difference between last 2h and previous 2h
    momentum = (history_windows[:, -12:, t].mean(axis=1) - history_windows[:, -24:-12, t].mean(axis=1)) * 0.2
    
    # Adaptive decay based on recent volatility
    vol = history_windows[:, -48:, t].std(axis=1) / (train_series[target_col].std() + 1e-6)
    decay_rate = np.clip(0.015 + 0.04 * vol, 0.01, 0.08)
    decay = np.exp(-decay_rate[:, None] * np.arange(output_len))
    
    # Target is a blend of cycle, steady-state, and momentum
    target = (0.4 * cycle_preds[None, :] + 0.6 * steady[:, None]) + momentum[:, None]
    
    # Blend persistence with target
    last_p = history_windows[:, -1, t]
    preds = last_p[:, None] * decay + target * (1 - decay)
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
