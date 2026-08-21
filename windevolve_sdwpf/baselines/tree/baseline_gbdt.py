# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/baselines/tree/baseline_gbdt.py

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

    FEATURE SETS. The feature list is not fixed. FEATURE_SET selects one of six
    NESTED tiers, each a strict superset of the one before, so the difference
    between two adjacent tiers isolates exactly what the added information is
    worth:

        power       own past power and time only -- NO wind speed
        base        + wind speed  (the original 12 features; the reference)
        turbine     + the turbine's other channels: direction, temperatures,
                      pitch angles, reactive power, a parked-fraction flag
        farm        + farm-wide aggregates at the origin (all 134 turbines)
        neighbour   + the k nearest turbines' power and wind speed
        xy          + the turbine's raw x/y coordinates

    `power` vs `base` measures what wind speed buys. `farm` and `neighbour`
    are the SPATIAL tiers, and they answer whether turbine layout carries
    anything the competition's spatial entrants were after.

    A WARNING ABOUT THE `xy` TIER. Each turbine has unique coordinates, so x/y
    is a turbine IDENTIFIER in disguise. A tree given x/y can memorise
    per-turbine behaviour, which quietly converts the global model into 134
    per-turbine models and destroys the one property that makes this baseline
    the right comparator for the generalisation question. It is included
    because the comparison is informative, and it must be reported separately
    rather than folded into "the tuned XGBoost". `farm` and `neighbour` do not
    have this problem: both are aggregates over OTHER turbines and carry no
    identity.

Citation:
    Program contract and scoring follow the official Baidu KDD Cup 2022 SDWPF
    evaluator (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # Run from the package root (windevolve_sdwpf/):
    # 1. Train the global model (writes gbdt_model.joblib):
    python baselines/tree/baseline_gbdt.py --origins 150
    python baselines/tree/baseline_gbdt.py --features neighbour --origins 150
    # 2. Score it with the official metric:
    python evaluation/evaluator.py baselines/tree/baseline_gbdt.py
    # 3. Or sweep every tier at once:
    python analysis/sweep.py gbdt --only features
"""
import os                                  # paths and env vars
import time                                # training timers
import argparse                            # training CLI
import sys

import numpy as np                         # feature construction
import pandas as pd                        # CSV reading
import joblib                              # model persistence
import xgboost as xgb                      # the gradient boosting library

CAPACITY = 134                # turbines
OUTPUT_LEN = 288              # forecast horizon in 10-min steps (48h)
DAY_LEN = 144                 # steps per day
START_COL = 3                 # first modeled column (Wspd); Patv is the target

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths import VAL_TRAIN_FILE, LOCATION_FILE

# Where the fitted model is written / read from (next to this file).
MODEL_PATH = os.environ.get("SDWPF_GBDT_MODEL", os.path.join(_HERE, "gbdt_model.joblib"))

# Truncated training csv -- see note in paths.py. Fitting on the full 245-day
# file would leak the validation period into the model.
TRAIN_CSV = VAL_TRAIN_FILE

# Which nested feature tier to use.
#
# MEASURED, against a noise floor of +/-0.396 MW:
#
#   power      9 feats  45.834      base minus wind speed
#   base      12 feats  46.139  <-- DEFAULT
#   turbine   22 feats  48.151      +2.01 MW WORSE, roughly 5 sigma
#   farm      26 feats  47.582
#   neighbour 29 feats  47.789
#   xy        31 feats  47.609
#
# More information makes this model WORSE, not better. Three readings:
#
#   power -> base       wind speed at the origin buys nothing measurable. The
#                       target is 48h out and nearly independent of present
#                       wind; without a weather forecast there is little to use.
#   base -> turbine     the largest single effect in the sweep, and negative.
#                       These are origin-time snapshots held constant across all
#                       288 horizons, so they let the model fit the origin state
#                       when the distant target does not depend on it -- and
#                       colsample_bytree dilutes the few useful columns among 22.
#                       Consistent with every capacity axis saying "overfitting".
#   neighbour -> xy     turbine coordinates uniquely identify a turbine, so this
#                       is the strongest available invitation to memorise
#                       per-turbine behaviour. It changes nothing, which is
#                       evidence there is little turbine-specific structure to
#                       exploit. Relevant to the project's generalisation
#                       question; report it separately, never as "tuned XGBoost".
#
# The turbine tier may deserve a retest at the tuned capacity below, since the
# penalty may reflect over-capacity rather than the features themselves.
FEATURE_SET = os.environ.get("SDWPF_GBDT_FEATURES", "base")

# ---------------------------------------------------------------------------
# Model capacity. All three defaults come from the sweep on the 197/48
# validation split; see docs/baseline_report.md section 4.5.
#
#   trees   200  45.445   400  46.139   800  46.683
#   depth     4  45.512     6  46.139     9  47.180
#   lr     0.03  45.759  0.05  46.139   0.1  47.231
#
# Every axis is monotonic toward a SMALLER model, each at 3-4 sigma of the
# measured noise floor (+/-0.396 MW): this model overfits at the original
# settings. CAVEAT: each optimum sits at the EDGE of the swept range, so the
# true optimum is below these values and 45.445 is a lower bound rather than
# this model's best. Extending the grid downward (trees 50/100, depth 2/3,
# lr 0.01/0.02) is an open item.
# ---------------------------------------------------------------------------
N_ESTIMATORS = 200            # boosting rounds
MAX_DEPTH = 4                 # maximum tree depth
LEARNING_RATE = 0.03          # shrinkage applied to each tree's contribution

# How many nearest turbines the "neighbour" tier aggregates over.
K_NEIGHBOURS = int(os.environ.get("SDWPF_GBDT_K", "5"))

# ---------------------------------------------------------------------------
# Feature tiers. Nested by SET, and `base` keeps the ORIGINAL column order so a
# model retrained at the reference configuration is identical to the one that
# produced the previously reported number -- a reordering would perturb which
# columns colsample_bytree draws and move the score for no reason.
# ---------------------------------------------------------------------------
_BASE = ["h", "hour_of_day", "patv_last", "patv_mean_1h", "patv_mean_6h",
         "patv_mean_24h", "patv_std_24h", "wspd_last", "wspd_mean_6h",
         "wspd_mean_24h", "patv_yesterday", "patv_turbine_mean"]

_INC_TURBINE = ["wdir_sin", "wdir_cos", "etmp_last", "itmp_last",
                "pab_max_last", "prtv_last", "zero_frac_24h", "wspd_std_24h",
                "patv_mean_48h", "wspd_yesterday"]
_INC_FARM = ["farm_patv_now", "farm_wspd_now", "farm_patv_24h", "turbine_share"]
_INC_NEIGHBOUR = ["nbr_patv_last", "nbr_wspd_last", "nbr_wspd_mean_6h"]
_INC_XY = ["turb_x", "turb_y"]

_TIER_FEATURES = {
    # `power` is `base` with the three wind-speed columns removed, so the
    # difference between the two is exactly what wind speed is worth.
    "power": [f for f in _BASE if not f.startswith("wspd")],
    "base": list(_BASE),
    "turbine": _BASE + _INC_TURBINE,
    "farm": _BASE + _INC_TURBINE + _INC_FARM,
    "neighbour": _BASE + _INC_TURBINE + _INC_FARM + _INC_NEIGHBOUR,
    "xy": _BASE + _INC_TURBINE + _INC_FARM + _INC_NEIGHBOUR + _INC_XY,
    # Non-nested: base plus ONLY the neighbour columns. Every other tier is
    # cumulative, so this is the only one that isolates what the 3 neighbour
    # features are worth by themselves.
    "base+nbr": _BASE + _INC_NEIGHBOUR,
}
TIERS = ["power", "base", "turbine", "farm", "neighbour", "xy"]


def feature_names(feature_set=None):
    """
    Description:
        The ordered feature list for one tier.
    Args:
        feature_set: tier name (defaults to the module-level FEATURE_SET)
    Returns:
        A list of feature names, in the fixed column order used everywhere.
    """
    want = feature_set or FEATURE_SET
    if want not in _TIER_FEATURES:
        raise ValueError("unknown feature set {!r}; choose from {}".format(want, TIERS))
    return list(_TIER_FEATURES[want])


# The original 12, kept as a name so older callers and the docs still resolve.
FEATURES = feature_names("base")

_CACHE = {"model": None, "features": None, "feature_set": None, "k": K_NEIGHBOURS}

def _resolve_path(envs, template):
    """
    Description:
        Resolve a model/checkpoint path, substituting "{fold}" when the
        evaluator supplies a fold name in envs. This is what lets ONE process
        score every fold of a cross-validation layout: without it, the path is
        fixed at import time and all five folds would silently load the same
        artefact -- which is a leak, not a score.
    Args:
        envs: the settings dict passed to forecast() (may be None)
        template: the configured path, optionally containing "{fold}"
    Returns:
        The resolved filesystem path.
    """
    if "{fold}" not in template:
        return template
    fold = (envs or {}).get("fold")
    if not fold:
        raise ValueError(
            "path {!r} contains '{{fold}}' but the evaluator supplied no fold. "
            "Use --folds-root, or set a path without the placeholder."
            .format(template))
    return template.format(fold=fold)



# ---------------------------------------------------------------------------
# Farm context: everything a feature might need, loaded once per CSV
# ---------------------------------------------------------------------------
def _read_locations(location_file=None):
    """Read the turbine layout, sorted by TurbID. Returns None if unavailable."""
    path = location_file or LOCATION_FILE
    if not path or not os.path.exists(path):
        return None
    loc = pd.read_csv(path)
    cols = {c.lower(): c for c in loc.columns}
    xc, yc = cols.get("x"), cols.get("y")
    if xc is None or yc is None:
        return None
    key = cols.get("turbid", loc.columns[0])
    loc = loc.sort_values(key).reset_index(drop=True)
    return loc[[xc, yc]].to_numpy(dtype=np.float32)


def _neighbour_index(capacity, k, location_file=None):
    """
    Description:
        For each turbine, the indices of its k nearest turbines by Euclidean
        distance in the farm layout. A turbine is never its own neighbour.
    Args:
        capacity: number of turbines
        k: neighbours per turbine
        location_file: path to the turbine location CSV
    Returns:
        An integer array of shape (capacity, k), or None if no layout is found.
    """
    xy = _read_locations(location_file)
    if xy is None:
        return None
    xy = xy[:capacity]
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)                      # never your own neighbour
    return np.argsort(d, axis=1)[:, :k]


def build_context(df, capacity, feature_set=None, k=None, location_file=None):
    """
    Description:
        Reshape one CSV into per-turbine channel arrays and precompute
        everything the requested tier needs. Channels beyond power and wind
        speed are only read when the tier asks for them, so the cheap tiers stay
        cheap.
    Args:
        df: the loaded CSV (turbine-major, 13 official columns)
        capacity: number of turbines
        feature_set: tier name
        k: neighbours per turbine
        location_file: path to the turbine location CSV
    Returns:
        A context dict consumed by _build_features().
    """
    want = feature_set or FEATURE_SET
    k = K_NEIGHBOURS if k is None else k
    names = set(feature_names(want))
    total = len(df) // capacity

    def chan(col):
        a = np.nan_to_num(df[col].to_numpy(dtype=np.float32), nan=0.0)
        return a.reshape(capacity, total)

    ctx = {"total": total, "capacity": capacity, "feature_set": want,
           "patv": chan("Patv"), "wspd": chan("Wspd")}
    ctx["turbine_mean"] = ctx["patv"].mean(axis=1)

    if "wdir_sin" in names:
        ctx["wdir"] = chan("Wdir")
        ctx["etmp"] = chan("Etmp")
        ctx["itmp"] = chan("Itmp")
        ctx["prtv"] = chan("Prtv")
        ctx["pab"] = np.maximum(np.maximum(chan("Pab1"), chan("Pab2")), chan("Pab3"))
    if "farm_patv_now" in names:
        ctx["farm_patv"] = ctx["patv"].mean(axis=0)          # (total,)
        ctx["farm_wspd"] = ctx["wspd"].mean(axis=0)
        ctx["farm_mean"] = float(ctx["patv"].mean())
    if "nbr_patv_last" in names:
        ctx["nbr"] = _neighbour_index(capacity, k, location_file)
        if ctx["nbr"] is None:
            raise FileNotFoundError(
                "the {!r} feature tier needs the turbine location file; looked "
                "at {!r}".format(want, location_file or LOCATION_FILE))
    if "turb_x" in names:
        xy = _read_locations(location_file)
        if xy is None:
            raise FileNotFoundError(
                "the {!r} feature tier needs the turbine location file".format(want))
        ctx["xy"] = xy[:capacity]
    return ctx


def _build_features(ctx, t, origin, horizons):
    """
    Description:
        Build the feature matrix for one turbine at one forecast origin, for the
        requested horizon steps. Uses only data at or before the origin -- for
        this turbine and, in the spatial tiers, for the others too.
    Args:
        ctx: context from build_context()
        t: turbine index
        origin: index of the forecast origin within the series
        horizons: 1-D integer array of horizon steps (1-based)
    Returns:
        A float32 array of shape (len(horizons), n_features).
    """
    names = feature_names(ctx["feature_set"])
    patv, wspd = ctx["patv"], ctx["wspd"]
    p, w = patv[t], wspd[t]
    hist_p, hist_w = p[:origin + 1], w[:origin + 1]

    patv_mean_24h = hist_p[-DAY_LEN:].mean()
    n = len(horizons)
    ones = np.ones(n, dtype=np.float32)

    # Power one day before each predicted step; where that index is before the
    # start of history, fall back to the recent daily mean.
    yidx = origin + horizons - DAY_LEN
    inb = yidx >= 0
    clipped = np.clip(yidx, 0, len(p) - 1)

    f = {}
    f["h"] = horizons.astype(np.float32)
    f["hour_of_day"] = ((origin + horizons) % DAY_LEN) / 6.0
    f["patv_last"] = ones * hist_p[-1]
    f["patv_mean_1h"] = ones * hist_p[-6:].mean()
    f["patv_mean_6h"] = ones * hist_p[-36:].mean()
    f["patv_mean_24h"] = ones * patv_mean_24h
    f["patv_std_24h"] = ones * hist_p[-DAY_LEN:].std()
    f["patv_yesterday"] = np.where(inb, p[clipped], patv_mean_24h)
    f["patv_turbine_mean"] = ones * ctx["turbine_mean"][t]

    if "wspd_last" in names:
        f["wspd_last"] = ones * hist_w[-1]
        f["wspd_mean_6h"] = ones * hist_w[-36:].mean()
        f["wspd_mean_24h"] = ones * hist_w[-DAY_LEN:].mean()

    if "wdir_sin" in names:
        # Direction is circular: 359 deg and 1 deg are one degree apart, but as
        # a raw number they are 358 apart and a tree would have to spend splits
        # rediscovering that. sin/cos removes the discontinuity.
        wd = np.deg2rad(float(ctx["wdir"][t][origin]))
        f["wdir_sin"] = ones * np.sin(wd)
        f["wdir_cos"] = ones * np.cos(wd)
        f["etmp_last"] = ones * ctx["etmp"][t][origin]
        f["itmp_last"] = ones * ctx["itmp"][t][origin]
        # Pitch angle: blades feathered means the turbine is deliberately
        # spilling wind. The strongest available signal for "this turbine is
        # currently not producing and will probably stay that way".
        f["pab_max_last"] = ones * ctx["pab"][t][origin]
        f["prtv_last"] = ones * ctx["prtv"][t][origin]
        f["zero_frac_24h"] = ones * float((hist_p[-DAY_LEN:] <= 0).mean())
        f["wspd_std_24h"] = ones * hist_w[-DAY_LEN:].std()
        f["patv_mean_48h"] = ones * hist_p[-2 * DAY_LEN:].mean()
        f["wspd_yesterday"] = np.where(inb, w[clipped], hist_w[-DAY_LEN:].mean())

    if "farm_patv_now" in names:
        lo24 = max(0, origin - DAY_LEN + 1)
        f["farm_patv_now"] = ones * ctx["farm_patv"][origin]
        f["farm_wspd_now"] = ones * ctx["farm_wspd"][origin]
        f["farm_patv_24h"] = ones * ctx["farm_patv"][lo24:origin + 1].mean()
        denom = ctx["farm_mean"] if ctx["farm_mean"] else 1.0
        f["turbine_share"] = ones * (ctx["turbine_mean"][t] / denom)

    if "nbr_patv_last" in names:
        nb = ctx["nbr"][t]
        lo6 = max(0, origin - 36 + 1)
        f["nbr_patv_last"] = ones * float(patv[nb, origin].mean())
        f["nbr_wspd_last"] = ones * float(wspd[nb, origin].mean())
        f["nbr_wspd_mean_6h"] = ones * float(wspd[nb, lo6:origin + 1].mean())

    if "turb_x" in names:
        f["turb_x"] = ones * ctx["xy"][t, 0]
        f["turb_y"] = ones * ctx["xy"][t, 1]

    X = np.empty((n, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = f[name]
    return X


def build_training_set(csv_path, n_origins=150, horizon_stride=12, seed=42,
                       feature_set=None, k=None, turbines=None):
    """
    Description:
        Sample forecast origins from the training CSV and build the direct
        multi-horizon training matrix. Sampling keeps the problem tractable:
        every origin could contribute 288 rows, so horizons are subsampled.
    Args:
        csv_path: path to the truncated training CSV
        n_origins: forecast origins sampled per turbine
        horizon_stride: keep every Nth horizon step (12 -> 24 of the 288)
        seed: RNG seed for origin sampling
        feature_set: tier name
        k: neighbours per turbine
        turbines: iterable of turbine indices to sample from, or None for all.
                  Used by scope="per_turbine" to build one turbine's matrix.
                  Note the CONTEXT is still built from the whole farm, so the
                  farm and neighbour feature tiers keep working -- only the rows
                  are restricted.
    Returns:
        A tuple (X, y) of float arrays.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)
    ctx = build_context(df, CAPACITY, feature_set, k)
    total = ctx["total"]
    horizons = np.arange(1, OUTPUT_LEN + 1, horizon_stride)

    Xs, ys = [], []
    lo, hi = DAY_LEN, total - OUTPUT_LEN - 1        # need history behind, truth ahead
    for t in (range(CAPACITY) if turbines is None else turbines):
        origins = rng.integers(lo, hi, size=n_origins)
        for o in origins:
            Xs.append(_build_features(ctx, t, int(o), horizons))
            ys.append(ctx["patv"][t][o + horizons])
    return np.vstack(Xs), np.concatenate(ys)


def train(csv_path=None, n_origins=150, horizon_stride=12, seed=42,
          model_path=None, n_estimators=None, max_depth=None, learning_rate=None,
          feature_set=None, k=None, subsample=0.8, colsample_bytree=0.8,
          min_child_weight=1, objective="reg:squarederror", scope="global"):
    """
    Description:
        Fit the global XGBoost model and write it to disk, recording which
        feature tier produced it so inference cannot silently mismatch.
    Args:
        csv_path: training CSV (defaults to TRAIN_CSV)
        n_origins: forecast origins sampled per turbine
        horizon_stride: horizon subsampling stride
        seed: RNG seed for sampling and for the model
        model_path: output path (defaults to MODEL_PATH)
        n_estimators: number of boosting rounds (trees)
        max_depth: maximum depth of each tree
        learning_rate: shrinkage applied to each tree's contribution
        feature_set: tier name
        k: neighbours per turbine for the spatial tiers
        subsample / colsample_bytree / min_child_weight: XGBoost regularisation
        scope: "global" fits ONE model on all 134 turbines pooled, which is the
               published baseline. "per_turbine" fits 134 independent models,
               each seeing only its own turbine's rows. The global model gets
               134x the data for one parameter set; a per-turbine model gets a
               specialist fit on roughly 200 days. Which wins is the scope axis.
        objective: XGBoost loss. "reg:squarederror" matches RMSE;
                   "reg:absoluteerror" matches MAE. The competition score is
                   half of each, so neither is the right loss and both are worth
                   measuring.
    Returns:
        The fitted model.
    """
    csv_path = csv_path or TRAIN_CSV
    model_path = model_path or MODEL_PATH
    fs = feature_set or FEATURE_SET
    n_estimators = N_ESTIMATORS if n_estimators is None else n_estimators
    max_depth = MAX_DEPTH if max_depth is None else max_depth
    learning_rate = LEARNING_RATE if learning_rate is None else learning_rate
    names = feature_names(fs)
    print("building training set from", csv_path)
    print("  feature set: {} ({} features)".format(fs, len(names)))
    t0 = time.perf_counter()

    def _make():
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            objective=objective,
            random_state=seed,
            n_jobs=-1,
        )

    if scope == "per_turbine":
        # One model per turbine. The context is still built farm-wide inside
        # build_training_set, so neighbour and farm features remain available;
        # only the training ROWS are restricted to that turbine.
        models = []
        for t in range(CAPACITY):
            Xi, yi = build_training_set(csv_path, n_origins, horizon_stride,
                                        seed, fs, k, turbines=[t])
            mi = _make()
            mi.fit(Xi, yi)
            models.append(mi)
            if t % 10 == 0 or t == CAPACITY - 1:
                print("  turbine {:>3}/{}  ({:,} rows)  {:.1f}s elapsed"
                      .format(t, CAPACITY - 1, Xi.shape[0],
                              time.perf_counter() - t0))
        joblib.dump({"models": models, "features": names, "feature_set": fs,
                     "scope": "per_turbine",
                     "k": K_NEIGHBOURS if k is None else k}, model_path)
        print("saved {} per-turbine models to {}".format(len(models), model_path))
        return models

    X, y = build_training_set(csv_path, n_origins, horizon_stride, seed, fs, k)
    print("  {:,} rows x {} features in {:.1f}s".format(X.shape[0], X.shape[1],
                                                        time.perf_counter() - t0))

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,      # number of trees, added one after another
        max_depth=max_depth,            # how many yes/no questions deep each tree goes
        learning_rate=learning_rate,    # shrink each tree's correction (smaller = safer)
        subsample=subsample,            # fraction of rows each tree sees
        colsample_bytree=colsample_bytree,   # fraction of features each tree sees
        min_child_weight=min_child_weight,   # minimum evidence to justify a split
        objective=objective,            # the loss being minimised
        random_state=seed,
        n_jobs=-1,                      # use all CPU cores
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    print("  fitted {} trees in {:.1f}s".format(n_estimators, time.perf_counter() - t0))

    joblib.dump({"model": model, "features": names, "feature_set": fs,
                 "scope": "global",
                 "k": K_NEIGHBOURS if k is None else k}, model_path)
    print("saved to", model_path)
    return model


def _load_model(envs=None):
    """
    Description:
        Load the fitted model once and cache it, together with the feature tier
        it was trained on -- so a checkpoint trained on one tier can never be
        scored with another tier's features.
    Args:
        None
    Returns:
        The fitted XGBoost regressor.
    """
    path = _resolve_path(envs, MODEL_PATH)
    if _CACHE["model"] is None or _CACHE.get("path") != path:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "No trained model at {!r}. Train it first: "
                "python baselines/tree/baseline_gbdt.py".format(path))
        print("[gbdt] loading model from: {}".format(path), flush=True)
        blob = joblib.load(path)
        _CACHE["scope"] = blob.get("scope", "global")
        # per-turbine checkpoints hold a list under "models"; global ones hold a
        # single estimator under "model". Older checkpoints predate the field
        # and are global by construction.
        _CACHE["models"] = blob.get("models")
        _CACHE["model"] = blob.get("model")
        _CACHE["path"] = path
        _CACHE["features"] = blob.get("features", FEATURES)
        _CACHE["feature_set"] = blob.get("feature_set", "base")
        _CACHE["k"] = blob.get("k", K_NEIGHBOURS)
    return _CACHE["model"]


def forecast(envs):
    """
    Description:
        Predict each turbine's next 48h with the shared XGBoost model, building
        features from the test window's own history. The feature tier is taken
        from the checkpoint, not from the module constant, so a mismatch is
        impossible.
    Args:
        envs: settings dict; must contain "path_to_test_x". Optional "capacity",
              "output_len" and "path_to_locations" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", CAPACITY)
    output_len = envs.get("output_len", OUTPUT_LEN)
    model = _load_model(envs)

    df = pd.read_csv(envs["path_to_test_x"])
    ctx = build_context(df, capacity, _CACHE["feature_set"], _CACHE["k"],
                        envs.get("path_to_locations"))
    horizons = np.arange(1, output_len + 1)
    origin = ctx["total"] - 1

    # Build features for every turbine, predict in one batched call for speed.
    blocks = [_build_features(ctx, t, origin, horizons) for t in range(capacity)]

    if _CACHE.get("scope") == "per_turbine":
        # Each turbine is predicted by its own model, so the blocks cannot be
        # stacked into one call.
        models = _CACHE["models"]
        yhat = np.vstack([models[t].predict(blocks[t])[None, :]
                          for t in range(capacity)])
    else:
        X = np.vstack(blocks)                          # (capacity*output_len, n_features)
        yhat = model.predict(X).reshape(capacity, output_len)

    return np.clip(yhat, 0.0, None)[:, :, None].astype(float)


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
    ap.add_argument("--data", type=str, default=TRAIN_CSV, help="Path to the training CSV")
    ap.add_argument("--model", type=str, default=MODEL_PATH, help="Output model path")
    ap.add_argument("--features", type=str, default=FEATURE_SET, choices=TIERS,
                    help="Feature tier (default: {})".format(FEATURE_SET))
    ap.add_argument("--k", type=int, default=K_NEIGHBOURS,
                    help="Nearest turbines aggregated by the spatial tiers")
    ap.add_argument("--origins", type=int, default=150, help="Origins sampled per turbine")
    ap.add_argument("--stride", type=int, default=12, help="Horizon subsampling stride")
    ap.add_argument("--trees", type=int, default=N_ESTIMATORS,
                    help="Number of boosting rounds (default: {})".format(N_ESTIMATORS))
    ap.add_argument("--depth", type=int, default=MAX_DEPTH,
                    help="Max tree depth (default: {})".format(MAX_DEPTH))
    ap.add_argument("--lr", type=float, default=LEARNING_RATE,
                    help="Learning rate / shrinkage (default: {})".format(LEARNING_RATE))
    ap.add_argument("--objective", type=str, default="reg:squarederror",
                    help="XGBoost loss (reg:squarederror | reg:absoluteerror)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    a = ap.parse_args()
    train(a.data, a.origins, a.stride, a.seed, a.model,
          n_estimators=a.trees, max_depth=a.depth, learning_rate=a.lr,
          feature_set=a.features, k=a.k, objective=a.objective)


if __name__ == "__main__":
    main()