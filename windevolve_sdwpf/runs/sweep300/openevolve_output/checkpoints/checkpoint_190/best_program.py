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
    # Use ratio of recent power to recent wind speed to estimate current efficiency
    w_idx = feature_cols.index("Wspd")
    recent_p = np.mean(history_windows[:, -72:, t_idx], axis=1)
    recent_w = np.mean(history_windows[:, -72:, w_idx], axis=1)
    avg_w = train_series["Wspd"].mean() + 1e-6
    # Use 1h mean as current state and 6h mean for trend; capture momentum
    m1 = np.mean(history_windows[:, -12:, t_idx], axis=1)
    m6 = np.mean(history_windows[:, -72:, t_idx], axis=1)
    
    # Dynamic scaling based on recent efficiency (recent power / recent wind)
    # This captures the turbine's current operating state better than a static mean
    eff = np.clip(recent_p / (recent_w + 0.1), 0, train_series[target_col].max())
    
    t = np.linspace(0, 1, output_len)
    diurnal = 0.06 * np.sin(t * 4 * np.pi) * train_series[target_col].std()
    
    # Use median for robustness and refine decay
    long_term = train_series[target_col].median()
    decay = np.exp(-t * 2.8)
    
    # Local diurnal scaling based on history volatility, phase shifted
    v_hist = np.std(history_windows[:, -144:, t_idx], axis=1)
    diurnal = 0.08 * np.sin(t * 4 * np.pi + 0.5) * v_hist[:, None]
    
    # Combined baseline with adaptive momentum damping
    w_std = np.std(history_windows[:, -72:, w_idx], axis=1)
    mom_damp = np.exp(-w_std[:, None] / (train_series["Wspd"].std() + 1e-3))
    base = 0.7 * m1[:, None] + 0.3 * (eff[:, None] * (recent_w[:, None] / (train_series["Wspd"].mean() + 1e-1)))
    
    # Use square root of time for momentum to allow longer retention
    mom_term = (m1[:, None] - m6[:, None]) * (t[None, :]**0.5) * mom_damp
    preds = (base + mom_term) * decay + long_term * (1 - decay) + diurnal
    
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
