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
    t_idx = feature_cols.index(target_col)
    w_idx = feature_cols.index("Wspd")
    # Physical estimate based on recent wind speed
    w_curr = np.mean(history_windows[:, -24:, w_idx], axis=1)
    w_max = train_series["Wspd"].quantile(0.95) + 1e-6
    est_p = train_series[target_col].quantile(0.99) * np.power(np.clip(w_curr / w_max, 0, 1), 2.2)
    
    m1, m6 = np.mean(history_windows[:, -12:, t_idx], axis=1), np.mean(history_windows[:, -72:, t_idx], axis=1)
    t = np.linspace(0, 1, output_len)
    vol = np.std(history_windows[:, -144:, t_idx], axis=1)
    # Adaptive decay: faster reversion when recent volatility is high
    decay_rate = 2.5 + 1.5 * (vol / (train_series[target_col].std() + 1e-3))
    decay = np.exp(-t[None, :] * decay_rate[:, None])
    
    # Blend momentum-based start with physical estimate
    start = 0.55 * m1 + 0.45 * est_p
    # Diurnal scale based on local volatility for better amplitude control
    diurnal = 0.08 * np.sin(t * 4 * np.pi + 0.4) * vol[:, None]
    
    # Dampen trend projection slightly to prevent overshoot
    trend = (m1[:, None] - m6[:, None]) * t[None, :] * 0.3
    preds = (start[:, None] + trend + diurnal) * decay + train_series[target_col].median() * (1 - decay)
    
    return np.clip(preds, 0.0, train_series[target_col].quantile(0.995))
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import pandas as pd
    cols = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
    hw = np.random.RandomState(0).rand(8, 144, len(cols)) * 100
    tr = pd.DataFrame(np.random.rand(500, len(cols)) * 100, columns=cols)
    out = forecast(tr, hw, cols, "Patv", 144, 288)
    # checks the shape of the output 
    print("shape:", out.shape, "| expected:", (8, 288))
