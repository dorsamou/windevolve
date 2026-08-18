# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/arms/karpathy/ws/greedy-cv/forecaster.py

Description:
    Gradient-boosted forecaster for SDWPF 48h-ahead active power. One model per
    fold predicts a turbine's power deviation from its climatological mean at
    horizon h, from the state of its recent history plus h itself:

        pred_t(h) = mu_t + gbm(lvl_1h, lvl_24h, wspd_1h, farm_1h, mu_t, h)

    Fitted per fold on that fold's own training file and cached under the fold
    name. Horizon enters as a feature rather than being given its own model, so
    one fit covers all 288 steps and the decay of predictability is learned
    rather than imposed.

    This is a third model class, tested to see whether a nonlinear learner can
    find structure the linear shrinkage model and the GRU both miss. The prior
    is not strong: extra predictors have repeatedly failed here, and the reason
    is non-stationarity rather than sampling noise -- with 1.8M training rows,
    adding a 24h power level still lost 0.22 MW, and beta fitted on the two
    halves of a training period differs by factors of 1.5 to 1.8. A flexible
    learner is more exposed to that, not less. It is worth one measurement
    because trees can express the one thing the linear model cannot: that
    persistence differs between calm and windy states, and that a turbine
    reading zero against a generating farm is faulted rather than becalmed.

Citation:
    Task and metric from the Baidu KDD Cup 2022 SDWPF competition.
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.
"""
import numpy as np                            # array math
import pandas as pd                           # CSV reading
import xgboost as xgb                         # the booster

DAY_LEN = 144            # 10-minute observations per day
SHORT_STEPS = 6          # steps in the "current level" window (1 hour)
LONG_STEPS = DAY_LEN     # steps in the slower background level (1 day)

# Origins sampled per turbine. Every horizon 1..288 is used at each origin, so
# rows are highly correlated within an origin and the effective sample size is
# the ORIGIN count, not the row count: 134 turbines x 70 origins is ~9400
# independent states behind the fit. Sampling origins rather than thinning
# horizons keeps every h represented exactly, so the model never has to
# interpolate across a horizon it was not shown.
N_ORIGINS = 70

GBM_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 50,      # rows are correlated within an origin
    "tree_method": "hist",
    "objective": "reg:squarederror",
}
N_ROUNDS = 300

# Fitted boosters, keyed by (fold, training csv) -- NEVER a bare global. All
# five folds are scored in ONE process, so a fold-blind cache would serve fold
# 1's model to every later fold and the numbers would be silently wrong.
_FIT_CACHE = {}


def _columns(path, capacity, names):
    """
    Description:
        Read the named columns of a turbine-major SDWPF csv and reshape each to
        one row per turbine, filling missing readings with zero -- the same
        convention the official loader applies before modelling.
    Args:
        path: csv to read (a fold_train.csv or a window infile)
        capacity: number of turbines (134)
        names: column names to return
    Returns:
        A dict mapping name -> (capacity, steps_per_turbine) float array.
    """
    frame = pd.read_csv(path, usecols=list(names))
    per = len(frame) // capacity
    out = {}
    for name in names:
        values = frame[name].values[:capacity * per].astype(np.float64)
        out[name] = np.nan_to_num(values, nan=0.0).reshape(capacity, per)
    return out


def _trailing_mean(series, steps):
    """
    Description:
        Trailing mean of `steps` observations as a sliding window. Column i is
        the mean over times i .. i+steps-1, i.e. the level known at i+steps-1.
    Args:
        series: (capacity, steps_per_turbine) array
        steps: window length
    Returns:
        A (capacity, steps_per_turbine - steps + 1) float array.
    """
    csum = np.cumsum(np.concatenate(
        [np.zeros((series.shape[0], 1)), series], axis=1), axis=1)
    return (csum[:, steps:] - csum[:, :-steps]) / float(steps)


def _state(data, mu, wspd_mu):
    """
    Description:
        The four history features, aligned so that column k is the state known
        at time LONG_STEPS - 1 + k. The long window sets the alignment because
        it is the last to become available.
    Args:
        data: dict of (capacity, steps) arrays from _columns()
        mu: per-turbine mean power (capacity,)
        wspd_mu: per-turbine mean wind speed (capacity,)
    Returns:
        A list of four (capacity, width) arrays: short and long power levels,
        wind speed level, and the farm-wide short level.
    """
    lag = LONG_STEPS - SHORT_STEPS
    short = _trailing_mean(data["Patv"], SHORT_STEPS)[:, lag:] - mu[:, None]
    wind = _trailing_mean(data["Wspd"], SHORT_STEPS)[:, lag:] - wspd_mu[:, None]
    long = _trailing_mean(data["Patv"], LONG_STEPS) - mu[:, None]
    width = min(short.shape[1], long.shape[1])
    short, wind, long = short[:, :width], wind[:, :width], long[:, :width]
    farm = np.broadcast_to(short.mean(axis=0), short.shape)
    return [short, long, wind, farm]


def _fit(envs, capacity, output_len):
    """
    Description:
        Fit this fold's booster on its own training file.
    Args:
        envs: the settings dict passed to forecast()
        capacity: number of turbines (134)
        output_len: forecast horizon in steps (288)
    Returns:
        A (mu, wspd_mu, booster) triple.
    """
    key = (envs.get("fold"), envs.get("path_to_train"))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]

    data = _columns(envs["path_to_train"], capacity, ("Patv", "Wspd"))
    power = data["Patv"]
    mu = power.mean(axis=1)
    wspd_mu = data["Wspd"].mean(axis=1)
    dev = power - mu[:, None]

    feats = _state(data, mu, wspd_mu)
    offset = LONG_STEPS - 1                      # time of feature column 0
    usable = feats[0].shape[1] - output_len      # origins with a full future
    if usable <= 0:
        raise ValueError("training file too short for a {}-step horizon".format(output_len))

    # Evenly spaced origins across the whole training period.
    origins = np.linspace(0, usable - 1, min(N_ORIGINS, usable)).astype(int)
    horizons = np.arange(1, output_len + 1)

    n_rows = capacity * len(origins) * output_len
    X = np.empty((n_rows, 6), dtype=np.float32)
    y = np.empty(n_rows, dtype=np.float32)
    row = 0
    for o in origins:
        block = capacity * output_len
        for j, f in enumerate(feats):
            X[row:row + block, j] = np.repeat(f[:, o], output_len)
        X[row:row + block, 4] = np.repeat(mu, output_len)
        X[row:row + block, 5] = np.tile(horizons, capacity)
        y[row:row + block] = dev[:, offset + o + 1:offset + o + 1 + output_len].ravel()
        row += block

    booster = xgb.train(GBM_PARAMS, xgb.DMatrix(X[:row], label=y[:row]),
                        num_boost_round=N_ROUNDS)
    print("[gbm] fold {}: {} origins x {} turbines x {} horizons = {} rows".format(
        envs.get("fold"), len(origins), capacity, output_len, row), flush=True)
    _FIT_CACHE[key] = (mu, wspd_mu, booster)
    return mu, wspd_mu, booster


def forecast(envs):
    """
    Description:
        Produce a 48h active-power forecast for every turbine from this fold's
        booster. This is the program contract used by evaluator.py.
    Args:
        envs: settings dict; must contain "path_to_test_x" and "path_to_train".
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", 134)
    output_len = envs.get("output_len", 288)
    mu, wspd_mu, booster = _fit(envs, capacity, output_len)

    data = _columns(envs["path_to_test_x"], capacity, ("Patv", "Wspd"))
    short = data["Patv"][:, -SHORT_STEPS:].mean(axis=1) - mu
    long = data["Patv"][:, -LONG_STEPS:].mean(axis=1) - mu
    wind = data["Wspd"][:, -SHORT_STEPS:].mean(axis=1) - wspd_mu
    farm = np.full(capacity, short.mean())

    X = np.empty((capacity * output_len, 6), dtype=np.float32)
    for j, f in enumerate((short, long, wind, farm)):
        X[:, j] = np.repeat(f, output_len)
    X[:, 4] = np.repeat(mu, output_len)
    X[:, 5] = np.tile(np.arange(1, output_len + 1), capacity)

    pred = booster.predict(xgb.DMatrix(X)).reshape(capacity, output_len)
    return (pred + mu[:, None])[:, :, None]
