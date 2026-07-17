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
    # Capture local power state: map recent wind speed to an expected power level
    w = feature_cols.index("Wspd")
    last_wspd = history_windows[:, -1, w]
    # Binning: use training data to find mean power for similar wind speeds
    bins = np.linspace(0, 15, 15)
    bin_idx = np.digitize(last_wspd, bins)
    
    # Calculate conditional mean power for each bin from training data
    train_series['bin'] = np.digitize(train_series["Wspd"], bins)
    bin_means = train_series.groupby('bin')[target_col].mean().reindex(range(len(bins)+1), fill_value=0)
    
    # Estimate based on recent observed power and the wind-speed-derived baseline
    recent = np.mean(history_windows[:, -12:, t], axis=1)
    curve_est = bin_means[bin_idx].values
    
    # Mean reversion: forecast transitions from recent state to the long-term median
    long_term_avg = train_series[target_col].median()
    decay = np.linspace(1.0, 0.0, output_len)
    
    # Blend current state/curve with long-term climate over time
    base = 0.7 * recent + 0.3 * curve_est
    preds = np.outer(base, decay) + np.outer(np.full(len(recent), long_term_avg), 1.0 - decay)
    
    return np.clip(preds, 0.0, 5000.0)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import pandas as pd
    cols = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
    hw = np.random.RandomState(0).rand(8, 144, len(cols)) * 100
    tr = pd.DataFrame(np.random.rand(500, len(cols)) * 100, columns=cols)
    out = forecast(tr, hw, cols, "Patv", 144, 288)
    # checks the shape of the output 
    print("shape:", out.shape, "| expected:", (8, 288))
