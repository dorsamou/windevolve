# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/baseline_models/tree/baseline_gbdt.py

Description:
    Tier-2 machine-learning baseline: gradient boosted decision trees (XGBoost).

    ONE GLOBAL model shared by all 134 turbines, which makes it the closest
    structural analogue to an evolved single-function forecaster and the fair
    comparator for the "does it generalize across turbines" question.

    Formulation: DIRECT multi-horizon regression. Each training row is one
    (turbine, forecast origin, horizon step) triple. The features summarize the
    state at the origin plus the horizon distance; the target is the actual
    power at origin + h. At prediction time the same features are built for
    h = 1..288 and the model is called once per turbine, so the full 48h
    trajectory is produced directly (no recursion, so no error compounding).

    This is the first baseline that reads WIND SPEED, which drives power through
    the nonlinear power curve. Trees represent that curve naturally by splitting
    the wind-speed axis into regions -- something a linear model (ARIMA) cannot
    do -- which is the main reason a tree model can beat the naive/ARIMA floor.

    Features (all derived from history at or before the origin -- never future):
      h                  horizon step, 1..288
      hour_of_day        time of day at the predicted step (captures diurnal shape)
      patv_last          power at the origin
      patv_mean_1h/6h/24h  recent mean power over 6/36/144 steps
      patv_std_24h       recent variability
      wspd_last          wind speed at the origin
      wspd_mean_6h/24h   recent mean wind speed
      patv_yesterday     power one day before the predicted step
      patv_turbine_mean  the turbine's own mean power (a capacity proxy)

Citation:
    Program contract and scoring follow the official Baidu KDD Cup 2022 SDWPF
    evaluator (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # Run from the package root (windevolve_sdwpf/):
    # 1. Train the global model on the 245-day CSV (writes gbdt_model.joblib):
    python baseline_models/tree/baseline_gbdt.py --origins 150
    # 2. Score it with the official metric:
    python evaluator.py baseline_models/tree/baseline_gbdt.py
"""
import os                                  # paths and env vars
import time                                # training timers
import argparse                            # training CLI

import numpy as np                         # feature construction
import pandas as pd                        # CSV reading
import joblib                              # model persistence
import xgboost as xgb                      # the gradient boosting library

CAPACITY = 134                # turbines
OUTPUT_LEN = 288              # forecast horizon in 10-min steps (48h)
DAY_LEN = 144                 # steps per day
START_COL = 3                 # first modeled column (Wspd); Patv is the target

_HERE = os.path.dirname(os.path.abspath(__file__))

# Where the fitted model is written / read from (next to this file).
MODEL_PATH = os.environ.get("SDWPF_GBDT_MODEL", os.path.join(_HERE, "gbdt_model.joblib"))

# The 245-day training CSV.
TRAIN_CSV = os.environ.get(
    "SDWPF_TRAIN_FILE",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/sdwpf_245days_v1.csv",
)

# The feature names, in the fixed column order used everywhere below.
FEATURES = ["h", "hour_of_day", "patv_last", "patv_mean_1h", "patv_mean_6h",
            "patv_mean_24h", "patv_std_24h", "wspd_last", "wspd_mean_6h",
            "wspd_mean_24h", "patv_yesterday", "patv_turbine_mean"]

_CACHE = {"model": None}      # module-level cache; forecast() runs 142 times


def _build_features(patv, wspd, origin, horizons, turbine_mean):
    """
    Description:
        Build the feature matrix for one turbine at one forecast origin, for the
        requested horizon steps. Uses only data at or before the origin.
    Args:
        patv: 1-D array of this turbine's power history (NaNs already zeroed)
        wspd: 1-D array of this turbine's wind speed history
        origin: index of the forecast origin within those arrays
        horizons: 1-D integer array of horizon steps (1-based)
        turbine_mean: this turbine's mean power over the available history
    Returns:
        A float array of shape (len(horizons), len(FEATURES)).
    """
    hist_p = patv[:origin + 1]                    # power up to and including the origin
    hist_w = wspd[:origin + 1]                    # wind speed up to the origin
    patv_last = hist_p[-1]                         # power right at the origin
    patv_mean_1h = hist_p[-6:].mean()             # last hour (6 steps)
    patv_mean_6h = hist_p[-36:].mean()            # last 6 hours
    patv_mean_24h = hist_p[-DAY_LEN:].mean()      # last day
    patv_std_24h = hist_p[-DAY_LEN:].std()        # last day's variability
    wspd_last = hist_w[-1]                         # wind speed at the origin
    wspd_mean_6h = hist_w[-36:].mean()
    wspd_mean_24h = hist_w[-DAY_LEN:].mean()

    n = len(horizons)
    # Power one day before each predicted step; where that index is before the
    # start of history, fall back to the recent daily mean.
    yesterday_idx = origin + horizons - DAY_LEN
    patv_yesterday = np.where(yesterday_idx >= 0,
                              patv[np.clip(yesterday_idx, 0, len(patv) - 1)],
                              patv_mean_24h)
    hour_of_day = ((origin + horizons) % DAY_LEN) / 6.0    # 0..24 in hours

    X = np.empty((n, len(FEATURES)), dtype=np.float32)
    X[:, 0] = horizons                 # h
    X[:, 1] = hour_of_day
    X[:, 2] = patv_last
    X[:, 3] = patv_mean_1h
    X[:, 4] = patv_mean_6h
    X[:, 5] = patv_mean_24h
    X[:, 6] = patv_std_24h
    X[:, 7] = wspd_last
    X[:, 8] = wspd_mean_6h
    X[:, 9] = wspd_mean_24h
    X[:, 10] = patv_yesterday
    X[:, 11] = turbine_mean
    return X


def build_training_set(csv_path, n_origins=150, horizon_stride=12, seed=42):
    """
    Description:
        Sample forecast origins from the training CSV and build the direct
        multi-horizon training matrix. Sampling keeps the problem tractable:
        every origin could contribute 288 rows, so horizons are subsampled.
    Args:
        csv_path: path to sdwpf_245days_v1.csv
        n_origins: forecast origins sampled per turbine
        horizon_stride: keep every Nth horizon step (12 -> 24 of the 288)
        seed: RNG seed for origin sampling
    Returns:
        A tuple (X, y) of float arrays.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)
    total = len(df) // CAPACITY
    patv_all = np.nan_to_num(df["Patv"].to_numpy(dtype=np.float32), nan=0.0)
    wspd_all = np.nan_to_num(df["Wspd"].to_numpy(dtype=np.float32), nan=0.0)
    horizons = np.arange(1, OUTPUT_LEN + 1, horizon_stride)

    Xs, ys = [], []
    lo, hi = DAY_LEN, total - OUTPUT_LEN - 1        # need history behind, truth ahead
    for t in range(CAPACITY):
        p = patv_all[t * total:(t + 1) * total]
        w = wspd_all[t * total:(t + 1) * total]
        tmean = float(p.mean())
        origins = rng.integers(lo, hi, size=n_origins)
        for o in origins:
            Xs.append(_build_features(p, w, int(o), horizons, tmean))
            ys.append(p[o + horizons])
    return np.vstack(Xs), np.concatenate(ys)


def train(csv_path=None, n_origins=150, horizon_stride=12, seed=42,
          model_path=None, n_estimators=400, max_depth=6, learning_rate=0.05):
    """
    Description:
        Fit the global XGBoost model and write it to disk.
    Args:
        csv_path: training CSV (defaults to TRAIN_CSV)
        n_origins: forecast origins sampled per turbine
        horizon_stride: horizon subsampling stride
        seed: RNG seed for sampling and for the model
        model_path: output path (defaults to MODEL_PATH)
        n_estimators: number of boosting rounds (trees)
        max_depth: maximum depth of each tree
        learning_rate: shrinkage applied to each tree's contribution
    Returns:
        The fitted model.
    """
    csv_path = csv_path or TRAIN_CSV
    model_path = model_path or MODEL_PATH
    print("building training set from", csv_path)
    t0 = time.perf_counter()
    X, y = build_training_set(csv_path, n_origins, horizon_stride, seed)
    print("  {:,} rows x {} features in {:.1f}s".format(X.shape[0], X.shape[1],
                                                        time.perf_counter() - t0))

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,      # number of trees, added one after another
        max_depth=max_depth,            # how many yes/no questions deep each tree goes
        learning_rate=learning_rate,    # shrink each tree's correction (smaller = safer)
        subsample=0.8,                  # each tree sees 80% of rows (reduces overfitting)
        colsample_bytree=0.8,           # each tree sees 80% of features
        objective="reg:squarederror",   # minimize squared error (matches RMSE)
        random_state=seed,
        n_jobs=-1,                      # use all CPU cores
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    print("  fitted {} trees in {:.1f}s".format(n_estimators, time.perf_counter() - t0))

    joblib.dump({"model": model, "features": FEATURES}, model_path)
    print("saved to", model_path)
    return model


def _load_model():
    """
    Description:
        Load the fitted model once and cache it for the remaining windows.
    Args:
        None
    Returns:
        The fitted XGBoost regressor.
    """
    if _CACHE["model"] is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model at {!r}. Train it first: "
                "python baseline_models/tree/baseline_gbdt.py".format(MODEL_PATH))
        _CACHE["model"] = joblib.load(MODEL_PATH)["model"]
    return _CACHE["model"]


def forecast(envs):
    """
    Description:
        Predict each turbine's next 48h with the shared XGBoost model, building
        features from that turbine's history in the test window.
    Args:
        envs: settings dict; must contain "path_to_test_x". Optional "capacity"
              and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", CAPACITY)
    output_len = envs.get("output_len", OUTPUT_LEN)
    model = _load_model()

    df = pd.read_csv(envs["path_to_test_x"])
    total = len(df) // capacity
    patv_all = np.nan_to_num(df["Patv"].to_numpy(dtype=np.float32), nan=0.0)
    wspd_all = np.nan_to_num(df["Wspd"].to_numpy(dtype=np.float32), nan=0.0)
    horizons = np.arange(1, output_len + 1)

    # Build features for every turbine, predict in one batched call for speed.
    blocks = []
    for t in range(capacity):
        p = patv_all[t * total:(t + 1) * total]
        w = wspd_all[t * total:(t + 1) * total]
        blocks.append(_build_features(p, w, len(p) - 1, horizons, float(p.mean())))
    X = np.vstack(blocks)                              # (capacity*output_len, n_features)
    yhat = model.predict(X).reshape(capacity, output_len)

    preds = np.clip(yhat, 0.0, None)[:, :, None].astype(float)
    return preds


def main():
    """
    Description:
        Parse arguments and train the global XGBoost model.
    Args:
        None
    Returns:
        None
    """
    ap = argparse.ArgumentParser(description="Train the global XGBoost SDWPF baseline")
    ap.add_argument("--data", type=str, default=TRAIN_CSV, help="Path to the 245-day CSV")
    ap.add_argument("--model", type=str, default=MODEL_PATH, help="Output model path")
    ap.add_argument("--origins", type=int, default=150, help="Origins sampled per turbine")
    ap.add_argument("--stride", type=int, default=12, help="Horizon subsampling stride")
    ap.add_argument("--trees", type=int, default=400, help="Number of boosting rounds")
    ap.add_argument("--depth", type=int, default=6, help="Max tree depth")
    ap.add_argument("--lr", type=float, default=0.05, help="Learning rate (shrinkage)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    a = ap.parse_args()
    train(a.data, a.origins, a.stride, a.seed, a.model,
          n_estimators=a.trees, max_depth=a.depth, learning_rate=a.lr)


if __name__ == "__main__":
    main()
