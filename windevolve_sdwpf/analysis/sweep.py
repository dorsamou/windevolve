# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/analysis/sweep.py

Description:
    One sweep harness for every baseline. Scores a baseline repeatedly under
    different parameter settings on the VALIDATION split, through the same
    official grader the evolve loop uses, and appends one row per configuration
    to a TSV as soon as that configuration finishes.

    Why one harness instead of five scripts: a sweep is only interpretable if
    every point in it was measured with the identical ruler. Five scripts drift.
    This file owns the ruler once; each model contributes only a GRID -- a list
    of parameter dicts -- and, if it needs fitting, a PREPARE function.

    Monitoring is EVENT-TRIGGERED, not polled. After each configuration the
    harness (a) appends its row to the TSV and fsyncs, (b) prints a one-line
    event with the score, the delta against the reference configuration, and an
    ETA, and (c) rewrites a small JSON status file. So `tail -f` on either file
    shows progress the instant a configuration lands, and a killed sweep loses
    at most the configuration that was in flight.

    Every baseline is scored WITHOUT modification: the harness imports the
    baseline module by path and overrides module-level constants on the loaded
    object. Nothing is written back to the baseline's source.

    THE SCALE AXIS. Every grid supports "scale", a uniform multiplier applied to
    the prediction. This is not a hyperparameter of any model -- it is a CONTROL.
    Shrinking an uninformative forecast toward the conditional mean improves
    0.5*(RMSE+MAE) substantially (see the repo README), so an evolutionary search
    will find shrinkage early and it will look like model discovery. Measuring
    the shrinkage-optimal version of each baseline puts that credit where it
    belongs before the search runs.

Citation:
    Scoring is delegated to evaluation/evaluator.py, which delegates to the
    official Baidu KDD Cup 2022 SDWPF grader (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation), imported verbatim.
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # from the package root (windevolve_sdwpf/)
    python analysis/sweep.py --list                     # show every grid
    python analysis/sweep.py naive --dry-run            # print the plan, run nothing
    python analysis/sweep.py naive                      # ~5 min,   no fitting
    python analysis/sweep.py climatology                # ~10 min,  no fitting
    python analysis/sweep.py gbdt                       # ~1-2 h,   retrains per config
    python analysis/sweep.py arma                       # ~1-2 h,   refits per order
    python analysis/sweep.py gru                        # ~30 min,  inference-only

    # shorten anything for a smoke test
    python analysis/sweep.py gbdt --windows 3 --only trees

    # watch it from another shell
    tail -f runs/sweeps/sweep_gbdt.tsv
"""
import os                                  # paths, env, fsync
import sys                                 # sys.path, exit codes
import json                                # config serialisation, status file
import time                                # timing and ETA
import argparse                            # CLI
import datetime                            # timestamps
import traceback                           # failure reporting
import importlib.util                      # load baselines by path

import numpy as np                         # arrays

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluator as ev                     # the ruler; also gives us VAL_DIR
from paths import VAL_DIR, VAL_TRAIN_FILE, RUNS_DIR, BASELINES_DIR

SWEEP_DIR = os.path.join(RUNS_DIR, "sweeps")

# Column order of the results TSV. Fixed, because plots.py reads it by name and
# a reordering would silently mislabel a figure.
COLUMNS = ["ts", "model", "axis", "label", "config", "score", "rmse", "mae",
           "latest_rmse", "n_windows", "eval_seconds", "fit_seconds",
           "valid", "reason"]


# ---------------------------------------------------------------------------
# Module loading and parameter injection
# ---------------------------------------------------------------------------
def load_baseline(rel_path):
    """
    Description:
        Import a baseline by path into a FRESH module object, so that its
        module-level constants and its module-level cache start clean. Every
        configuration gets its own module; nothing leaks between them.
    Args:
        rel_path: path to the baseline .py, relative to baselines/
    Returns:
        The loaded module object.
    """
    path = os.path.join(BASELINES_DIR, rel_path)
    if not os.path.isfile(path):
        raise FileNotFoundError("no baseline at {}".format(path))
    spec = importlib.util.spec_from_file_location(
        "sweep_candidate_{}".format(abs(hash(path + str(time.time())))), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_overrides(module, overrides):
    """
    Description:
        Set module-level constants on a loaded baseline. Refuses to invent a
        new name: if the attribute does not already exist, the grid is wrong
        about the baseline's interface and should fail loudly rather than set a
        constant nothing reads.
    Args:
        module: the loaded baseline
        overrides: dict of {ATTRIBUTE_NAME: value}
    Returns:
        None
    """
    for name, value in overrides.items():
        if not hasattr(module, name):
            raise AttributeError(
                "{} has no module-level {!r}; the grid and the baseline "
                "disagree".format(module.__name__, name))
        setattr(module, name, value)


def wrap_scale(module, scale):
    """
    Description:
        Wrap forecast() so its output is multiplied by a constant. Applied last,
        after every other override, so it composes with any configuration.
    Args:
        module: the loaded baseline
        scale: multiplier (1.0 = untouched)
    Returns:
        None
    """
    if scale == 1.0:
        return
    inner = module.forecast

    def scaled(envs, _inner=inner, _s=float(scale)):
        return np.asarray(_inner(envs), dtype=float) * _s

    module.forecast = scaled


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(module, val_dir, n_windows):
    """
    Description:
        Score a loaded baseline on the validation split through the official
        grader. Mirrors evaluator.evaluate()'s failure handling exactly -- it
        never raises -- so one broken configuration cannot end a sweep.
    Args:
        module: loaded baseline exposing forecast(envs)
        val_dir: validation window directory
        n_windows: cap on windows scored (None = all)
    Returns:
        A metrics dict, always containing "valid".
    """
    try:
        return ev._score_forecaster(module, val_dir, n_windows=n_windows)
    except ev.EvalBudgetExceeded as e:
        return {"valid": False, "reason": "eval budget exceeded: {}".format(e)}
    except Exception as e:
        # Includes om.MetricsError -- a degenerate prediction rejected by the
        # official guards, which at extreme shrinkage is a real outcome, not a
        # bug. Recorded as such so the sweep curve shows where the floor is.
        return {"valid": False, "reason": "{}: {}".format(type(e).__name__, e)}


# ---------------------------------------------------------------------------
# Grids. Each entry is a dict:
#   axis      which parameter this point varies (groups points into one curve)
#   label     short human-readable name for the point
#   overrides {MODULE_CONSTANT: value} applied to the loaded baseline
#   scale     uniform prediction multiplier (the control axis)
#   fit       optional dict passed to the model's prepare() before scoring
# ---------------------------------------------------------------------------
def _point(axis, label, overrides=None, scale=1.0, fit=None):
    """Build one grid point with defaults filled in."""
    return {"axis": axis, "label": label, "overrides": overrides or {},
            "scale": scale, "fit": fit}


SCALES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def grid_naive():
    """
    Description:
        Seasonal naive. One parameter: LAG, the block of recent history copied
        forward. LAG=1 is unscoreable (constant per turbine -> rejected by the
        official identical-prediction guard), so the sweep starts at 2.
    Args:
        None
    Returns:
        A list of grid points.
    """
    pts = []
    for lag in [2, 6, 18, 36, 72, 144, 288, 432, 1008, 2016]:
        pts.append(_point("lag", "lag={}".format(lag), {"LAG": lag}))
    for s in SCALES:
        pts.append(_point("scale", "lag=144 scale={}".format(s),
                          {"LAG": 144}, scale=s))
    return pts


def grid_climatology():
    """
    Description:
        Diurnal climatology. The question this model exists to answer is how
        much of the 48h forecast is just "what this turbine usually does at this
        hour", so the axes are about how that typical day is estimated.

          n_days     hard cutoff: average the last N days only. The
                     interpretable version of "14 days or 5 days".
          halflife   soft cutoff: exponentially weight the days, half-life in
                     days. Better posed than n_days -- a hard cutoff trusts day
                     5 completely and discards day 6, which nobody believes --
                     but less interpretable, so both are measured.
          agg        mean or median. SDWPF power is heavily zero-inflated, so a
                     couple of parked days drag a mean profile down and leave a
                     median untouched.
          smooth     circular moving average over the finished profile. Crossed
                     with n_days rather than varied alone, because a short
                     window makes a noisy profile and smoothing is the cure --
                     the two knobs only make sense together.
    Args:
        None
    Returns:
        A list of grid points.
    """
    pts = []
    for n in [1, 2, 3, 5, 7, 10, 14]:
        pts.append(_point("n_days", "n_days={}".format(n), {"N_DAYS": n}))
    for hl in [0.5, 1, 2, 3, 5, 10]:
        pts.append(_point("halflife", "halflife={}".format(hl),
                          {"N_DAYS": None, "HALFLIFE_DAYS": hl}))
    for n in [3, 7, 14]:
        pts.append(_point("agg", "median n_days={}".format(n),
                          {"N_DAYS": n, "AGG": "median"}))
    for n, sm in [(3, 6), (3, 18), (3, 36), (14, 6), (14, 18), (14, 36)]:
        pts.append(_point("smooth", "n_days={} smooth={}".format(n, sm),
                          {"N_DAYS": n, "SMOOTH_STEPS": sm}))
    for s_ in SCALES:
        pts.append(_point("scale", "n_days=14 scale={}".format(s_),
                          {"N_DAYS": 14}, scale=s_))
    return pts


def grid_arma():
    """
    Description:
        Per-turbine ARIMA(p,d,q). Three parameters, in decreasing cost order:

          order      (p,d,q) -- REFITS all 134 turbines. The d axis is the
                     interesting one: an ADF test on SDWPF power rejects a unit
                     root decisively, so d=1 should over-difference and lose.
                     ARIMA(1,1,1) sits in this sweep as the demonstration.
          fit_days   trailing training history -- also refits.
          APPLY_STEPS how much of a window's 14-day history is re-filtered to
                     seed the forecast. NO refit; pure inference cost.
    Args:
        None
    Returns:
        A list of grid points.
    """
    pts = []
    for order in [(1, 0, 0), (1, 0, 1), (2, 0, 1), (2, 0, 2), (1, 1, 0), (1, 1, 1)]:
        pts.append(_point("order", "order={}".format(",".join(map(str, order))),
                          fit={"order": order}))
    for d in [30, 90, 197]:
        pts.append(_point("fit_days", "order=1,0,1 fit_days={}".format(d),
                          fit={"order": (1, 0, 1), "fit_days": d}))
    for a in [144, 288, 432, 1008, 2016]:
        pts.append(_point("apply_steps", "order=1,0,1 apply={}".format(a),
                          {"APPLY_STEPS": a}, fit={"order": (1, 0, 1)}))
    for s in SCALES:
        pts.append(_point("scale", "order=1,0,1 scale={}".format(s),
                          scale=s, fit={"order": (1, 0, 1)}))
    return pts


def grid_gbdt():
    """
    Description:
        Global XGBoost, coordinate sweep around the default configuration
        (base features, trees=400, depth=6, lr=0.05, origins=150, stride=12,
        seed=42). A full factorial over these axes would be thousands of fits;
        a coordinate sweep is ~40 and answers the question the report asks,
        which is which knob moves the score at all.

        THE FEATURES AXIS IS THE INTERESTING ONE and it is an ablation rather
        than tuning. The six tiers are nested, so each adjacent pair isolates
        one block of information:

          power -> base        what WIND SPEED is worth
          base -> turbine      what the turbine's OTHER CHANNELS are worth --
                               direction, temperatures, pitch angle (which says
                               "this turbine is feathered and will stay off"),
                               reactive power, parked fraction
          turbine -> farm      what FARM-WIDE STATE is worth
          farm -> neighbour    what the NEAREST TURBINES are worth -- the real
                               spatial signal, since upwind turbines see weather
                               before downwind ones do
          neighbour -> xy      what TURBINE IDENTITY is worth

        The last step is the one to be careful with. x/y is unique per turbine,
        so it is an identifier: a tree given it can memorise per-turbine
        behaviour, which turns the global model into 134 private models and
        destroys the property that makes this baseline the right comparator for
        the generalisation question. Report it separately; do not fold it into
        "the tuned XGBoost".

        The objective axis is also worth having: the competition score is
        0.5*(RMSE+MAE), so squared error over-weights the tail relative to what
        is being scored and absolute error under-weights it. Neither is right,
        which makes the comparison informative.

        The seed axis is not a hyperparameter -- it is the NOISE FLOOR. No other
        difference in this sweep is meaningful unless it exceeds that spread.
    Args:
        None
    Returns:
        A list of grid points.
    """
    base = {"n_origins": 150, "horizon_stride": 12, "seed": 42,
            "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
            "feature_set": "base"}

    def with_(**kw):
        d = dict(base)
        d.update(kw)
        return d

    pts = []
    for s_ in [42, 43, 44]:
        pts.append(_point("seed", "seed={}".format(s_), fit=with_(seed=s_)))
    for fs in ["power", "base", "turbine", "farm", "neighbour", "xy"]:
        pts.append(_point("features", "features={}".format(fs),
                          fit=with_(feature_set=fs)))
    for k in [3, 10, 20]:
        pts.append(_point("k_neighbours", "neighbour k={}".format(k),
                          fit=with_(feature_set="neighbour", k=k)))
    for t in [200, 400, 800]:
        pts.append(_point("trees", "trees={}".format(t), fit=with_(n_estimators=t)))
    for d in [4, 6, 9]:
        pts.append(_point("depth", "depth={}".format(d), fit=with_(max_depth=d)))
    for lr in [0.03, 0.05, 0.1]:
        pts.append(_point("lr", "lr={}".format(lr), fit=with_(learning_rate=lr)))
    for o in [25, 50, 150, 400]:
        pts.append(_point("origins", "origins={}".format(o), fit=with_(n_origins=o)))
    for st in [24, 12, 4]:
        pts.append(_point("stride", "stride={}".format(st), fit=with_(horizon_stride=st)))
    for obj in ["reg:squarederror", "reg:absoluteerror"]:
        pts.append(_point("objective", "objective={}".format(obj.split(":")[1]),
                          fit=with_(objective=obj)))
    # The best-guess combination, so the sweep ends with one point that is not a
    # single-coordinate move. Confirm against the coordinate results before
    # believing it -- a coordinate sweep does not license this extrapolation.
    pts.append(_point("combined", "tuned combination",
                      fit=with_(feature_set="neighbour", n_origins=400,
                                horizon_stride=4, n_estimators=800, max_depth=8,
                                learning_rate=0.05)))
    for s_ in SCALES:
        pts.append(_point("scale", "default scale={}".format(s_), scale=s_, fit=base))
    return pts


def grid_gru():
    """
    Description:
        Per-turbine GRU -- a REAL training sweep. Every axis below except
        `provenance` and `scale` retrains all 134 models, because that is what
        a hyperparameter is: something the model learns differently under.

        The corrected training loop is now the default (gru_baseline.py), so
        nothing here is fighting a broken optimizer. The goal is a GRU that
        performs as well as this model family reasonably can, not one that
        reproduces a published number.

        COST. One point = 134 small models. Sequentially that is ~40 min; the
        workload is launch-bound rather than compute-bound, so sharding with
        --turbines across 4-6 concurrent processes brings it to ~7-10 min.
        Per-turbine seeding (default) is what makes that sharding valid.

        Axes, in the order worth running them:

          capacity   hidden 48 -> 96 -> 128, and 1 vs 2 layers. The official 48
                     hidden units is 22,801 parameters per turbine, which is
                     tiny, and nothing is shared between the 134 models.
          input_len  144 -> 288 -> 432. THE most suspicious default: the model
                     reads 24h of history and predicts 48h. Expect this to
                     matter more than capacity.
          optim      lr and its decay. The official 1e-4 halving every epoch is
                     ~1/500th of its start by epoch 10, so the model may simply
                     stop learning before it converges. Test lr and decay
                     together -- a higher lr with fast decay is not the same
                     experiment as a higher lr held flat.
          budget     epochs and patience. With the decay fixed, does more
                     training help, or is early stopping firing for a reason?
          batch      16 / 32 / 64 / 128. Interacts with lr, so read it against
                     the optim axis rather than alone.
          dropout    0.0 / 0.05 / 0.2. 134 independent models on one turbine's
                     data each is a regime where regularisation might matter.
          provenance the official loop, defects and all. One point, kept so the
                     report can connect to the published 47.850. NOT a tuning
                     point -- never fold it into "best tuned".
          scale      the shrinkage control. No retrain.
    Args:
        None
    Returns:
        A list of grid points.
    """
    def T(**kw):
        """One retraining point: settings overrides for train_one_turbine."""
        return {"settings": kw}

    pts = [_point("capacity", "hidden=48 layers=2 (default)", fit=T())]

    for h in [96, 128]:
        pts.append(_point("capacity", "hidden={}".format(h), fit=T(hidden=h)))
    pts.append(_point("capacity", "layers=1", fit=T(lstm_layer=1)))
    pts.append(_point("capacity", "hidden=96 layers=1", fit=T(hidden=96, lstm_layer=1)))

    for il in [288, 432]:
        pts.append(_point("input_len", "input_len={}".format(il), fit=T(input_len=il)))

    for lr, dec, name in [(3e-4, 0.5, "lr=3e-4"), (1e-3, 0.5, "lr=1e-3"),
                          (1e-4, 1.0, "lr=1e-4 flat"), (1e-3, 0.9, "lr=1e-3 decay=0.9")]:
        pts.append(_point("optim", name, fit=T(lr=lr, lr_decay=dec)))

    for ep, pat, name in [(20, 5, "20 epochs pat=5"), (40, 8, "40 epochs pat=8")]:
        pts.append(_point("budget", name, fit=T(train_epochs=ep, patience=pat,
                                                lr_decay=0.9)))

    for bs in [16, 64, 128]:
        pts.append(_point("batch", "batch={}".format(bs), fit=T(batch_size=bs)))

    for dp in [0.0, 0.2]:
        pts.append(_point("dropout", "dropout={}".format(dp), fit=T(dropout=dp)))

    pts.append(_point("provenance", "official loop (defects)",
                      {"INFER_DROPOUT": True},
                      fit=T(zero_grad=False, double_step=True, global_clip=False,
                            eval_val=False, shuffle_val=True,
                            per_turbine_seed=False)))

    # One combined point, from the best guess on each axis. A coordinate sweep
    # does not license this -- confirm it against the coordinate results.
    pts.append(_point("combined", "tuned combination",
                      fit=T(hidden=96, input_len=288, lr=1e-3, lr_decay=0.9,
                            train_epochs=20, patience=5)))

    for sc in SCALES:
        pts.append(_point("scale", "default scale={}".format(sc), scale=sc, fit=T()))
    return pts


# ---------------------------------------------------------------------------
# Prepare hooks: the fitting a configuration needs before it can be scored.
# Each returns (overrides, fit_seconds). Artifacts are cached on a key derived
# from the fit parameters, so re-running a sweep does not refit what it already
# fitted -- and so the apply_steps and scale axes reuse one fit.
# ---------------------------------------------------------------------------
def prepare_arma(fit, module, cache_dir):
    """
    Description:
        Fit 134 per-turbine ARIMA models for one (order, fit_days), cached.
    Args:
        fit: the point's fit dict
        module: the loaded baseline
        cache_dir: where artifacts are cached
    Returns:
        ({"MODEL_PATH": path}, fit_seconds)
    """
    order = tuple(fit["order"])
    fit_days = int(fit.get("fit_days", 197))
    key = "arma_p{}d{}q{}_f{}.joblib".format(order[0], order[1], order[2], fit_days)
    path = os.path.join(cache_dir, key)
    t0 = time.perf_counter()
    if not os.path.exists(path):
        module.MODEL_PATH = path
        module.fit_all(csv_path=VAL_TRAIN_FILE, turbines=None, order=order,
                       fit_steps=fit_days * module.DAY_LEN, model_path=path)
    return {"MODEL_PATH": path}, time.perf_counter() - t0


def prepare_gbdt(fit, module, cache_dir):
    """
    Description:
        Train one global XGBoost model for a configuration, cached. The cache
        key is built from EVERY fit parameter, so two feature tiers or two
        objectives can never collide on one artifact -- a collision would
        silently score the wrong model and look like a null result.
    Args:
        fit: the point's fit dict
        module: the loaded baseline
        cache_dir: where artifacts are cached
    Returns:
        ({"MODEL_PATH": path}, fit_seconds)
    """
    tag = "_".join("{}{}".format(k, str(v).replace(":", "-"))
                   for k, v in sorted(fit.items()))
    path = os.path.join(cache_dir, "gbdt_{}.joblib".format(tag))
    t0 = time.perf_counter()
    if not os.path.exists(path):
        module.train(csv_path=VAL_TRAIN_FILE, model_path=path, **fit)
    return {"MODEL_PATH": path}, time.perf_counter() - t0


def _train_file_days():
    """Whole days in the truncated training CSV the sweep fits on."""
    rows = sum(1 for _ in open(VAL_TRAIN_FILE)) - 1
    return int(round(rows / 134.0 / 144.0))


def _checkpoints_match(ckpt_dir, capacity, want_days):
    """
    Description:
        True only if ckpt_dir holds a complete set of checkpoints AND they were
        trained on a file with `want_days` days. Anything else is refused, so a
        245-day checkpoint can never be silently scored against a validation
        split carved out of those same 245 days.
    Args:
        ckpt_dir: candidate checkpoint folder
        capacity: number of turbines expected
        want_days: day count the sweep's training file actually has
    Returns:
        bool
    """
    import torch as _torch
    paths = [os.path.join(ckpt_dir, "model_{}.pt".format(i)) for i in range(capacity)]
    if not all(os.path.exists(p) for p in paths):
        return False
    try:
        blob = _torch.load(paths[0], map_location="cpu", weights_only=False)
        got = int(blob.get("settings", {}).get("total_size", -1))
    except Exception:
        return False
    if got != want_days:
        print("  refusing to reuse {}: trained on a {}-day file, this sweep "
              "fits on {} days".format(ckpt_dir, got, want_days))
        return False
    return True


# Settings that map onto gru_baseline.py CLI flags, for sharded training.
_GRU_FLAGS = {"hidden": "--hidden", "lstm_layer": "--layers",
              "input_len": "--input-len", "dropout": "--dropout",
              "lr": "--lr", "lr_decay": "--lr-decay",
              "train_epochs": "--epochs", "batch_size": "--batch-size",
              "patience": "--patience"}

# Concurrent turbine shards. The training loop is launch-bound rather than
# compute-bound -- 22,801 parameters at batch 32 barely occupies the GPU -- so
# this is close to a linear speedup. Only valid because seeding is per-turbine
# by default; the provenance point, which restores official single-stream
# seeding, is forced to run sequentially.
GRU_WORKERS = int(os.environ.get("SDWPF_GRU_WORKERS", "4"))


def _gru_cli_flags(script):
    """
    Description:
        The long option strings gru_baseline.py's parser accepts, read
        statically from its source. Static parsing avoids importing torch just
        to ask a question about argparse.
    Args:
        script: path to gru_baseline.py
    Returns:
        A set of flag strings, e.g. {"--hidden", "--lr-decay"}.
    """
    import ast as _ast
    flags = set()
    try:
        tree = _ast.parse(open(script).read())
    except Exception:
        return flags
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, _ast.Constant) and str(arg.value).startswith("--"):
                    flags.add(arg.value)
    return flags


def prepare_gru(fit, module, cache_dir):
    """
    Description:
        Train all 134 per-turbine GRUs for one configuration, cached.

        Sharded across GRU_WORKERS concurrent processes by default, each taking
        a --turbines range and writing only its own model_<tid>.pt into the
        shared folder. Valid because each turbine seeds itself from
        OFFICIAL_FIX_SEED + tid, so a turbine's weights no longer depend on
        which turbines ran before it.

        The provenance point sets per_turbine_seed=False to reproduce the
        official single RNG stream, and is therefore trained SEQUENTIALLY in
        one pass -- sharding it would not reproduce the stream it exists to
        reproduce.

    Args:
        fit: the point's fit dict; fit["settings"] overrides default_settings()
        module: the loaded GRU baseline
        cache_dir: where checkpoint folders are cached
    Returns:
        ({"CHECKPOINTS": dir}, fit_seconds)
    """
    import random as _random
    import subprocess
    import torch as _torch

    overrides = fit.get("settings", {})
    capacity = 134
    t0 = time.perf_counter()
    tag = "_".join("{}{}".format(k, v) for k, v in sorted(overrides.items())) or "default"
    ckpt = os.path.join(cache_dir, "ckpt_{}".format(tag))

    # PROVENANCE. Checkpoints are only reused when they were trained on the same
    # file this sweep scores against. A checkpoint trained on the full 245-day
    # CSV has seen the validation period, so scoring it here would be leakage --
    # and nothing in the .pt filename says which file produced it. The saved
    # settings do: resolve_day_sizes() writes the training file's real day count
    # into settings["total_size"]. This is why the harness never falls back to
    # the module's default CHECKPOINTS folder.
    want_days = _train_file_days()
    if _checkpoints_match(ckpt, capacity, want_days):
        return {"CHECKPOINTS": ckpt}, 0.0

    os.makedirs(ckpt, exist_ok=True)
    official = not overrides.get("per_turbine_seed", True)
    print("  training 134 turbines [{}]: {}".format(
        "sequential, official RNG stream" if official
        else "{} shards".format(GRU_WORKERS),
        ", ".join("{}={}".format(k, v) for k, v in sorted(overrides.items()))
        or "defaults"), flush=True)

    if not official and GRU_WORKERS > 1 and all(k in _GRU_FLAGS for k in overrides):
        script = os.path.join(BASELINES_DIR, MODELS["gru"]["path"])
        # Preflight. Launching eight processes only to have each die on an
        # unknown argument wastes a minute and hides the cause, so check the
        # flags against the script's own parser first.
        have = _gru_cli_flags(script)
        want = {_GRU_FLAGS[k] for k in overrides}
        missing = sorted(want - have)
        if missing:
            raise RuntimeError(
                "{} does not accept {}. The copy on disk is older than this "
                "sweep expects -- re-import baselines/gru/gru_baseline.py.".format(
                    os.path.basename(script), ", ".join(missing)))
        edges = [round(i * capacity / GRU_WORKERS) for i in range(GRU_WORKERS + 1)]
        log_dir = os.path.join(ckpt, "_logs")
        os.makedirs(log_dir, exist_ok=True)
        procs = []
        for lo, hi in zip(edges, edges[1:]):
            if lo >= hi:
                continue
            cmd = [sys.executable, script, "--data", VAL_TRAIN_FILE,
                   "--checkpoints", ckpt, "--turbines", "{}-{}".format(lo, hi - 1)]
            for k, v in overrides.items():
                cmd += [_GRU_FLAGS[k], str(v)]
            log = os.path.join(log_dir, "shard_{}-{}.log".format(lo, hi - 1))
            fh = open(log, "w")
            fh.write("$ {}\n\n".format(" ".join(cmd)))
            fh.flush()
            procs.append((subprocess.Popen(cmd, stdout=fh,
                                           stderr=subprocess.STDOUT), fh, log, cmd))
        failed = []
        for p, fh, log, cmd in procs:
            rc = p.wait()
            fh.close()
            if rc != 0:
                failed.append((rc, log, cmd))
        if failed:
            # Surface the real error rather than the fact that something failed.
            rc, log, cmd = failed[0]
            print("\n  SHARD FAILED (exit {}). Command was:\n    {}\n"
                  "  Last lines of {}:".format(rc, " ".join(cmd), log), flush=True)
            try:
                with open(log) as fh:
                    tail = fh.read().strip().splitlines()[-25:]
                for line in tail:
                    print("    | {}".format(line), flush=True)
            except OSError:
                print("    (log unreadable)", flush=True)
            raise RuntimeError(
                "{} of {} GRU training shards failed; see {}".format(
                    len(failed), len(procs), log))
        missing = [i for i in range(capacity)
                   if not os.path.exists(os.path.join(ckpt, "model_{}.pt".format(i)))]
        if missing:
            raise RuntimeError("shards finished but {} checkpoints are missing "
                               "(e.g. turbine {})".format(len(missing), missing[0]))
        return {"CHECKPOINTS": ckpt}, time.perf_counter() - t0

    # Sequential path: the provenance point, or anything not expressible as CLI
    # flags.
    settings = module.default_settings()
    settings.update(overrides)
    settings["_full_run"] = True
    device = module.pick_device()
    data = module.load_modeled_columns(VAL_TRAIN_FILE, settings["start_col"])
    module.resolve_day_sizes(data, settings)
    if official:
        seed = module.OFFICIAL_FIX_SEED
        _random.seed(seed)
        np.random.seed(seed)
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    for tid in range(capacity):
        module.train_one_turbine(data, tid, settings, device, ckpt)
    return {"CHECKPOINTS": ckpt}, time.perf_counter() - t0


_SCALE = ("Multiply every prediction by a constant. Not a hyperparameter -- a control. "
          "The metric rewards shrinking an uninformative forecast toward the conditional "
          "mean, and the evolutionary search will find that within a few generations. "
          "Measuring it here means the search gets credit for discovery rather than for "
          "calibration. Expect it to help the weak forecasters a lot and the strong ones "
          "barely at all: a well-calibrated forecast has nothing to shrink toward.")

# ---------------------------------------------------------------------------
# What each axis is FOR. Read with `python analysis/sweep.py <model> --explain`.
#
#   kind    control  = not a hyperparameter; a check on the metric itself
#           noise    = repeats of one configuration; the resolution limit
#           ablation = answers a question about the DATA or the METHOD; the
#                      result is reportable whichever way it comes out
#           tuning   = looks for a better number and nothing else
#   core    True  = runs by default
#           False = only with --extended or an explicit --only
# ---------------------------------------------------------------------------
AXIS_INFO = {
    ("naive", "lag"): ("ablation", True,
        "How long a block of recent history is copied forward. lag=144 is 'tomorrow "
        "looks like today', lag=288 is 'the next 48h looks like the last 48h'. lag=1 is "
        "textbook persistence and is UNSCOREABLE -- it emits a constant per turbine and "
        "the official identical-prediction guard rejects it -- so the sweep starts at 2."),
    ("naive", "scale"): ("control", True, _SCALE),

    ("climatology", "n_days"): ("ablation", True,
        "How many trailing days are averaged into the daily profile. Answers 'is the "
        "recent weather regime worth more than the long-run daily shape'."),
    ("climatology", "halflife"): ("ablation", True,
        "The same question, better posed: weight days exponentially instead of cutting "
        "them off. A hard cutoff trusts day 5 completely and discards day 6, which is "
        "not a belief anyone holds."),
    ("climatology", "agg"): ("ablation", True,
        "Mean or median across days. SDWPF power is heavily zero-inflated, so a couple "
        "of parked days drag a mean profile down and leave a median untouched."),
    ("climatology", "smooth"): ("tuning", False,
        "Circular moving average over the finished profile, crossed with n_days. Few "
        "days makes a noisy profile that the model then forecasts as if it were "
        "structure; smoothing trades that noise for blur."),
    ("climatology", "scale"): ("control", True, _SCALE),

    ("arma", "order"): ("ablation", True,
        "The (p,d,q) order. The d axis is the point: an ADF test on SDWPF power rejects "
        "a unit root decisively, so d=1 should over-difference and lose. ARIMA(1,1,1) "
        "lives here rather than as a separate baseline. REFITS 134 turbines per point."),
    ("arma", "fit_days"): ("ablation", True,
        "Trailing training history per turbine. If 30 days matches 197, the coefficients "
        "are stable and the fit is cheap -- itself a useful finding. REFITS."),
    ("arma", "apply_steps"): ("tuning", True,
        "How much of a window's 14-day history is re-filtered to seed the forecast. No "
        "refit; the model's main inference-speed lever."),
    ("arma", "scale"): ("control", True, _SCALE),

    ("gbdt", "seed"): ("noise", True,
        "Three fits of the IDENTICAL configuration, differing only in the RNG that "
        "samples forecast origins and builds trees. This is the NOISE FLOOR: no other "
        "difference in this sweep means anything unless it exceeds this spread."),
    ("gbdt", "features"): ("ablation", True,
        "Six NESTED feature tiers, so each adjacent pair isolates one block of "
        "information. power->base is what wind speed is worth. base->turbine adds the "
        "channels the model currently ignores (pitch angle says a turbine is feathered "
        "and will stay off). turbine->farm adds farm-wide state. farm->neighbour adds "
        "the k nearest turbines -- the real spatial signal. neighbour->xy adds raw "
        "coordinates, which are a turbine IDENTIFIER: that gap estimates how much of the "
        "score is turbine-specific rather than transferable. Report xy separately."),
    ("gbdt", "objective"): ("ablation", True,
        "Squared vs absolute error. The score is 0.5*(RMSE+MAE), so squared error "
        "over-weights the tail and absolute under-weights it. Neither is right, which is "
        "what makes the comparison worth running."),
    ("gbdt", "trees"): ("tuning", True, "Boosting rounds. Capacity."),
    ("gbdt", "depth"): ("tuning", True, "Tree depth. Capacity."),
    ("gbdt", "lr"): ("tuning", True, "Shrinkage per tree. Trades rounds against step size."),
    ("gbdt", "origins"): ("tuning", False,
        "Forecast origins sampled per turbine -- how much TRAINING DATA, as opposed to "
        "how much model. Worth knowing which of the two binds."),
    ("gbdt", "stride"): ("tuning", False,
        "Horizon subsampling. The model trains on 24 of the 288 horizons and is applied "
        "to all 288, interpolating via the h feature. Tests whether that costs anything."),
    ("gbdt", "k_neighbours"): ("tuning", False,
        "Neighbourhood size for the spatial tiers. Only meaningful if `features` shows "
        "neighbour helping at all."),
    ("gbdt", "combined"): ("tuning", False,
        "One point combining the best guess from every axis. A coordinate sweep does not "
        "license this extrapolation -- confirm it against the coordinate results."),
    ("gbdt", "scale"): ("control", True, _SCALE),

    ("gru", "capacity"): ("tuning", True,
        "Hidden size and layer count. The official 48 hidden units is 22,801 parameters "
        "per turbine -- tiny -- and nothing is shared between the 134 models. RETRAINS."),
    ("gru", "input_len"): ("tuning", True,
        "How many history steps the model reads. The official 144 (24h) is used to "
        "predict 288 (48h): the model forecasts two days from one. This is the most "
        "suspicious default in the file and the axis most likely to move the score. "
        "RETRAINS."),
    ("gru", "optim"): ("tuning", True,
        "Learning rate and its per-epoch decay, varied TOGETHER. The official 1e-4 "
        "halving each epoch is ~1/500th of its starting value by epoch 10, so the model "
        "may stop learning before it converges. A higher lr with fast decay is a "
        "different experiment from a higher lr held flat. RETRAINS."),
    ("gru", "budget"): ("tuning", True,
        "Epochs and early-stopping patience, with the decay slowed so the extra epochs "
        "can do something. Answers whether the model ever finished training. RETRAINS."),
    ("gru", "batch"): ("tuning", False,
        "Minibatch size. Interacts with the learning rate, so read it against the optim "
        "axis rather than alone. RETRAINS."),
    ("gru", "dropout"): ("tuning", False,
        "Regularisation strength. 134 independent models, each fitted to one turbine's "
        "data, is a regime where this might matter. RETRAINS."),
    ("gru", "provenance"): ("ablation", True,
        "The official Baidu loop with all four defects restored: gradients never zeroed, "
        "two optimizer steps per batch, per-parameter clipping, dropout live at "
        "validation AND inference, single-stream seeding. One point, kept so the report "
        "can connect to the published 47.850. NOT a tuning point. RETRAINS, sequentially."),
    ("gru", "combined"): ("tuning", False,
        "Best guess from every axis at once. A coordinate sweep does not license this "
        "extrapolation -- confirm it against the coordinate results. RETRAINS."),
    ("gru", "scale"): ("control", True, _SCALE),
}


def axis_info(model, axis):
    """Look up an axis's (kind, core, description), with a safe default."""
    return AXIS_INFO.get((model, axis), ("tuning", True, "(no description)"))


MODELS = {
    "naive": {
        "path": "naive/baseline_naive.py",
        "grid": grid_naive,
        "prepare": None,
        "reference": "lag=144",
        "note": "no fitting; ~1 min per point",
    },
    "climatology": {
        "path": "classical/baseline_climatology.py",
        "grid": grid_climatology,
        "prepare": None,
        "reference": "n_days=14",
        "note": "no fitting; ~1 min per point",
    },
    "arma": {
        "path": "classical/baseline_arima.py",
        "grid": grid_arma,
        "prepare": prepare_arma,
        "reference": "order=1,0,1",
        "note": "refits 134 turbines per (order, fit_days); ~5-10 min per fit",
    },
    "gbdt": {
        "path": "tree/baseline_gbdt.py",
        "grid": grid_gbdt,
        "prepare": prepare_gbdt,
        "reference": "seed=42",
        "note": "retrains per point; ~3-5 min per point",
    },
    "gru": {
        "path": "gru/gru_baseline.py",
        "grid": grid_gru,
        "prepare": prepare_gru,
        "reference": "hidden=48 layers=2 (default)",
        "note": "every axis but scale RETRAINS 134 models: ~40 min sequential, ~7-10 min sharded across 4-6 --turbines processes",
    },
}


def post_load_gru(module, overrides):
    """
    Description:
        The GRU's one knob that is not a module constant. _load_all() calls
        model.train() on every checkpoint, faithfully reproducing the official
        predict.py's live dropout. To score with dropout OFF we must load first,
        then flip. Handled here rather than by editing gru_baseline.py, because
        the baseline's job is to stay faithful.
    Args:
        module: the loaded GRU baseline
        overrides: this point's override dict
    Returns:
        None
    """
    if not overrides.pop("_SWEEP_EVAL_MODE", False):
        return
    module._load_all(134)
    for m in module._CACHE["models"]:
        m.eval()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _row(model, pt, res, fit_seconds):
    """Assemble one TSV row from a scored grid point."""
    cfg = dict(pt["overrides"])
    cfg["scale"] = pt["scale"]
    if pt["fit"]:
        cfg["fit"] = {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in pt["fit"].items()}
    return {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "axis": pt["axis"],
        "label": pt["label"],
        "config": json.dumps(cfg, sort_keys=True),
        "score": round(res["competition_score"], 4) if res.get("valid") else "",
        "rmse": round(res["rmse"], 4) if res.get("valid") else "",
        "mae": round(res["mae"], 4) if res.get("valid") else "",
        "latest_rmse": round(res["latest_rmse"], 4) if res.get("valid") else "",
        "n_windows": res.get("n_windows", ""),
        "eval_seconds": res.get("eval_seconds", ""),
        "fit_seconds": round(fit_seconds, 1),
        "valid": int(bool(res.get("valid"))),
        "reason": res.get("reason", ""),
    }


def _append(tsv_path, row, write_header):
    """Append one row and force it to disk, so `tail -f` sees it immediately."""
    with open(tsv_path, "a") as fh:
        if write_header:
            fh.write("\t".join(COLUMNS) + "\n")
        fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _status(path, payload):
    """Rewrite the status file atomically."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run(model, args):
    """
    Description:
        Execute one model's sweep, emitting a TSV row and a console event as
        each configuration finishes.
    Args:
        model: key into MODELS
        args: parsed CLI namespace
    Returns:
        0 on success, 1 if every configuration failed.
    """
    spec = MODELS[model]
    points = spec["grid"]()
    if args.only:
        keep = set(args.only.split(","))
        points = [p for p in points if p["axis"] in keep]
        if not points:
            sys.stderr.write("no grid points on axis/axes {}\n".format(sorted(keep)))
            return 1
    elif not args.extended:
        dropped = sorted({p["axis"] for p in points
                          if not axis_info(model, p["axis"])[1]})
        points = [p for p in points if axis_info(model, p["axis"])[1]]
        if dropped:
            print("skipping (use --extended or --only): {}".format(", ".join(dropped)))

    # Measure the reference configuration FIRST, so every delta printed below is
    # anchored to the same number rather than to whatever happened to run first.
    points.sort(key=lambda p: p["label"] != spec["reference"])

    os.makedirs(SWEEP_DIR, exist_ok=True)
    cache_dir = os.path.join(SWEEP_DIR, "artifacts", model)
    os.makedirs(cache_dir, exist_ok=True)
    tsv = args.out or os.path.join(SWEEP_DIR, "sweep_{}.tsv".format(model))
    status_path = os.path.splitext(tsv)[0] + ".status.json"

    print("model      : {}   ({})".format(model, spec["note"]))
    print("baseline   : {}".format(os.path.join(BASELINES_DIR, spec["path"])))
    print("val_dir    : {}".format(VAL_DIR))
    print("windows    : {}".format(args.windows or "all"))
    print("points     : {}".format(len(points)))
    print("results    : {}".format(tsv))
    print("axes       : {}".format(", ".join(sorted({p["axis"] for p in points}))))
    if points and points[0]["label"] != spec["reference"]:
        print("NOTE       : the reference point {!r} is not in this subset, so "
              "deltas are\n             anchored to {!r} instead. Absolute "
              "scores are unaffected.".format(spec["reference"], points[0]["label"]))
    if args.dry_run:
        print("\n--dry-run, nothing executed:\n")
        for i, p in enumerate(points, 1):
            print("  {:>3}. [{:<11}] {:<28} overrides={} fit={}".format(
                i, p["axis"], p["label"], p["overrides"] or "-", p["fit"] or "-"))
        return 0

    write_header = not os.path.exists(tsv) or os.path.getsize(tsv) == 0
    ref_score, done, t_start = None, 0, time.perf_counter()

    for i, pt in enumerate(points, 1):
        t_pt = time.perf_counter()
        fit_seconds = 0.0
        try:
            mod = load_baseline(spec["path"])
            overrides = dict(pt["overrides"])
            if pt["fit"] is not None and spec["prepare"] is not None:
                fit_over, fit_seconds = spec["prepare"](pt["fit"], mod, cache_dir)
                overrides.update(fit_over)
            if model == "gru":
                post_load_gru(mod, overrides)
            apply_overrides(mod, overrides)
            wrap_scale(mod, pt["scale"])
            res = score(mod, VAL_DIR, args.windows)
        except Exception as e:
            traceback.print_exc()
            res = {"valid": False, "reason": "{}: {}".format(type(e).__name__, e)}

        row = _row(model, pt, res, fit_seconds)
        _append(tsv, row, write_header)
        write_header = False
        done += 1

        # --- the event line ---------------------------------------------------
        if res.get("valid"):
            if ref_score is None:
                ref_score = res["competition_score"]
            delta = res["competition_score"] - ref_score
            verdict = "{:8.3f} MW  ({:+.3f} vs ref)".format(
                res["competition_score"], delta)
        else:
            verdict = "INVALID   {}".format(res.get("reason", "")[:70])

        rate = (time.perf_counter() - t_start) / done
        eta = rate * (len(points) - done)
        print("[{:>3}/{}] {:<11} {:<30} {}   {:.0f}s   ETA {:.0f}m".format(
            i, len(points), pt["axis"], pt["label"], verdict,
            time.perf_counter() - t_pt, eta / 60.0), flush=True)

        _status(status_path, {
            "model": model, "done": done, "total": len(points),
            "last_label": pt["label"], "last_valid": bool(res.get("valid")),
            "last_score": res.get("competition_score"),
            "elapsed_seconds": round(time.perf_counter() - t_start, 1),
            "eta_seconds": round(eta, 1),
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    print("\ndone: {} points -> {}".format(done, tsv))
    return 0


def explain(model, args):
    """
    Description:
        Print, in plain language, exactly what a sweep will do before it is run:
        every axis, what kind of thing it is, whether it retrains anything, how
        many points it contributes, what settings those points take, and what
        question the axis answers. Nothing is executed.
    Args:
        model: key into MODELS
        args: parsed CLI namespace
    Returns:
        0
    """
    spec = MODELS[model]
    points = spec["grid"]()
    order = {"noise": 0, "ablation": 1, "control": 2, "tuning": 3}

    axes = {}
    for p in points:
        axes.setdefault(p["axis"], []).append(p)

    print("=" * 74)
    print("{}   --   {} grid points across {} axes".format(
        model.upper(), len(points), len(axes)))
    print("baseline : {}".format(os.path.join(BASELINES_DIR, spec["path"])))
    print("full doc : {}".format(
        os.path.join(BASELINES_DIR, spec["path"]).replace(".py", ".md")))
    print("reference: {!r}  (every delta is measured against this)".format(
        spec["reference"]))
    print("=" * 74)

    for axis in sorted(axes, key=lambda a: (order.get(axis_info(model, a)[0], 9), a)):
        kind, core, text = axis_info(model, axis)
        pts = axes[axis]
        refits = any(p["fit"] is not None for p in pts) and spec["prepare"] is not None
        print("")
        print("  {}  [{}]{}{}".format(
            axis, kind,
            "" if core else "   -- NOT run by default; needs --extended or --only",
            "   -- REFITS" if refits else ""))
        print("  {} point(s): {}".format(
            len(pts), ", ".join(p["label"] for p in pts)))
        for line in _wrap(text, 70):
            print("      {}".format(line))

    core_pts = [p for p in points if axis_info(model, p["axis"])[1]]
    print("")
    print("-" * 74)
    print("default run (`python analysis/sweep.py {}`) : {} points".format(
        model, len(core_pts)))
    print("everything  (`... --extended`)              : {} points".format(len(points)))
    print("cost note   : {}".format(spec["note"]))
    print("-" * 74)
    return 0


def _wrap(text, width):
    """Wrap a paragraph to a width, without pulling in textwrap's extras."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = w if not cur else cur + " " + w
    if cur:
        lines.append(cur)
    return lines


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Parameter sweep for one SDWPF baseline, scored on the "
                    "validation split with the official metric.")
    ap.add_argument("model", nargs="?", choices=sorted(MODELS),
                    help="which baseline to sweep")
    ap.add_argument("--list", action="store_true", help="list models and their grids")
    ap.add_argument("--explain", action="store_true",
                    help="describe every axis in plain language, run nothing")
    ap.add_argument("--extended", action="store_true",
                    help="also run the axes excluded from the default core set")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every configuration, run nothing")
    ap.add_argument("--windows", type=int, default=None,
                    help="cap validation windows (default: all; use 2-3 to smoke test)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated axes to run, e.g. 'seed,scale'")
    ap.add_argument("--out", type=str, default=None, help="results TSV path")
    args = ap.parse_args()

    if args.list:
        for name, spec in sorted(MODELS.items()):
            pts = spec["grid"]()
            axes = {}
            for p in pts:
                axes[p["axis"]] = axes.get(p["axis"], 0) + 1
            print("{:<12} {:>3} points   {}".format(name, len(pts), spec["note"]))
            print("             axes: {}".format(
                ", ".join("{}({})".format(k, v) for k, v in sorted(axes.items()))))
        return 0
    if not args.model:
        ap.error("give a model, or --list")
    if args.explain:
        return explain(args.model, args)
    return run(args.model, args)


if __name__ == "__main__":
    sys.exit(main())
