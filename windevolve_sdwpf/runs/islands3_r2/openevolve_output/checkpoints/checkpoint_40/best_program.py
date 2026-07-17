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
    t, ws = feature_cols.index(target_col), feature_cols.index("Wspd")
    # Map wind speed to mean power using training data
    bins = np.linspace(0, 25, 26)
    idx = np.digitize(train_series["Wspd"], bins)
    lookup = train_series.groupby(idx)[target_col].mean().reindex(range(27), fill_value=0).values
    # Base: lookup current wind speed, Blend: recent power level
    curr_ws = np.mean(history_windows[:, :, ws], axis=1)
    base = lookup[np.clip(np.digitize(curr_ws, bins), 0, 26)]
    recent = np.mean(history_windows[:, -24:, t], axis=1)
    # Add diurnal component: mean power per hour of day
    train_series = train_series.copy()
    train_series['hour'] = (train_series.index // 6) % 24  # Assuming 10-min data, 6 steps/hr
    daily_profile = train_series.groupby('hour')[target_col].mean().values
    
    # Horizon-dependent blending:
    # Recent power dominates early, decays over time. 
    # Climatology (base) and diurnal cycle dominate later.
    h = np.arange(output_len)
    alpha = np.exp(-h / 48.0)  # Decay factor for recent power
    
    # Align diurnal profile to current time if possible (simplified as % 24)
    diurnal = daily_profile[h % 24]
    
    preds = (alpha * recent[:, None]) + ((1 - alpha) * (0.6 * base[:, None] + 0.4 * diurnal))
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
