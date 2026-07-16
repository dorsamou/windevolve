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
    # Estimate power based on recent wind speed using a simple lookup/binning approach
    w_idx = feature_cols.index("Wspd")
    p_idx = feature_cols.index(target_col)
    
    # Binning Wspd to estimate expected Patv from train_series
    bins = np.linspace(0, 25, 26)
    bin_indices = np.digitize(train_series["Wspd"], bins)
    mean_p_by_w = [train_series[bin_indices == i][target_col].mean() for i in range(len(bins) + 1)]
    mean_p_by_w = np.nan_to_num(mean_p_by_w)
    
    # Apply to the latest wind speed in the history window
    latest_wspd = np.clip(history_windows[:, -1, w_idx], 0, 25)
    w_bins = np.digitize(latest_wspd, bins)
    preds = np.array([mean_p_by_w[i] for i in w_bins])
    
    return np.repeat(preds[:, None], output_len, axis=1)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import pandas as pd
    cols = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
    hw = np.random.RandomState(0).rand(8, 144, len(cols)) * 100
    tr = pd.DataFrame(np.random.rand(500, len(cols)) * 100, columns=cols)
    out = forecast(tr, hw, cols, "Patv", 144, 288)
    print("shape:", out.shape, "| expected:", (8, 288))
