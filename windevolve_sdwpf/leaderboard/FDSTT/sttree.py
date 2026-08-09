# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/FDSTT/sttree.py

Description:
    Module 3 of 4 of FDSTT: the Spatio-partitioned Time-phased Tree.

    Despite the name, NO turbine coordinates are used. "Spatio-partitioned"
    means the 134 turbines are k-means clustered on WIND SPEED CORRELATION
    (paper Eq. 14), one LightGBM per cluster. "Time-phased" means the horizon
    index is handed to the model as a feature, so one tree per cluster covers
    all delta steps instead of needing one tree per step.

    Covers only the first delta = 30 steps (5 hours). Beyond that a tree has
    nothing to work with: its strongest features are recent readings, and it
    cannot extrapolate past values seen in training.

    Paper Section 4.1.2: 3 partitions, 1 temporal phase, delta = 30,
    learning rate 0.1, depth 15, 20 leaves.

    ANTI-LEAKAGE: the cluster assignment is fitted from the training period
    only and stored in fit/. It is never recomputed from a scoring window.

Feature row (77 columns), from Section 4.1.2:
    [ 0:30]  Wspd  at t-29 .. t
    [30:60]  Patv  at t-29 .. t
    [60:68]  Etmp Itmp Wdir Ndir Pab1 Prtv TurbID hour   at t
    [68:76]  x^sta: max/min/mean/median of Wspd and of Patv over 14 days
    [   76]  horizon ID, 1..30
    target:  Patv at t + ID

Usage:
    python leaderboard/FDSTT/sttree.py fit --fit-dir fit_val
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_HERE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from data import load_panel, load_fit, RAW_COLS, CAPACITY, STEPS_PER_DAY  # noqa: E402

DELTA = 30            # paper: horizon covered by the tree, 5 hours
LAG = 30              # paper: "wind speed and power at t-30+1 to t"
STA_DAYS = 14         # paper: statistics over the last 14 days
N_CLUSTERS = 3        # paper: "the number of spatial partitions is 3"

LGB_PARAMS = {"objective": "regression_l1", "learning_rate": 0.1,
              "max_depth": 15, "num_leaves": 20, "verbose": -1,
              "n_jobs": -1, "min_data_in_leaf": 50}


def _c(panel, name):
    return panel["values"][:, :, RAW_COLS[3:].index(name)]


# --------------------------------------------------------------------------
# spatio-partitioning -- Eq. 14, fitted on the training period only
# --------------------------------------------------------------------------
def fit_clusters(panel, n_clusters=N_CLUSTERS, seed=0):
    """
    Description:
        Pearson-correlate every pair of turbines' wind speed, then k-means the
        resulting correlation profiles. Each turbine's feature vector is its
        row of the correlation matrix -- i.e. "which turbines am I windy
        with" -- which is the sensible reading of the paper's one-line
        description.
    Returns:
        (134,) int cluster labels.
    """
    from sklearn.cluster import KMeans
    w = np.nan_to_num(_c(panel, "Wspd"), nan=0.0)          # (134, T)
    corr = np.corrcoef(w)
    corr = np.nan_to_num(corr, nan=0.0)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(corr).astype(np.int64)


# --------------------------------------------------------------------------
# tabular feature construction
# --------------------------------------------------------------------------
def _rows_for_origin(panel, o, turbines):
    """
    Description:
        Build the 76 non-horizon feature columns for one time origin, for the
        given turbines. Origin o means "t = o-1 is the last observed step".
    Returns:
        (len(turbines), 76) float32.
    """
    v = panel["values"]
    idx = {n: RAW_COLS[3:].index(n) for n in RAW_COLS[3:]}
    wspd = v[turbines, o - LAG:o, idx["Wspd"]]              # (n, 30)
    patv = v[turbines, o - LAG:o, idx["Patv"]]

    point = np.stack([v[turbines, o - 1, idx[n]] for n in
                      ("Etmp", "Itmp", "Wdir", "Ndir", "Pab1", "Prtv")], 1)
    turb_col = np.asarray(turbines, dtype=np.float32)[:, None]
    hour_col = np.full((len(turbines), 1), panel["hour"][o - 1], np.float32)

    lo = max(0, o - STA_DAYS * STEPS_PER_DAY)
    sw = v[turbines, lo:o, idx["Wspd"]]
    sp = v[turbines, lo:o, idx["Patv"]]
    sta = np.stack([np.nanmax(sw, 1), np.nanmin(sw, 1), np.nanmean(sw, 1),
                    np.nanmedian(sw, 1), np.nanmax(sp, 1), np.nanmin(sp, 1),
                    np.nanmean(sp, 1), np.nanmedian(sp, 1)], 1)

    out = np.concatenate([wspd, patv, point, turb_col, hour_col, sta], 1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_training_table(panel, turbines, stride=72, max_origin=None):
    """
    Description:
        Expand origins x turbines x DELTA horizons into a flat regression
        table. One row = one (turbine, origin, horizon) triple.
    Returns:
        (X, y) with X shape (n, 77) and y shape (n,).
    """
    steps = panel["steps"]
    last = (steps if max_origin is None else max_origin) - DELTA
    origins = np.arange(max(LAG, STA_DAYS * STEPS_PER_DAY), last + 1, stride)
    patv = _c(panel, "Patv")

    X, y = [], []
    for o in origins:
        base = _rows_for_origin(panel, int(o), turbines)     # (n, 76)
        for h in range(1, DELTA + 1):
            tgt = patv[turbines, o + h - 1]
            ok = np.isfinite(tgt) & (tgt >= 0)               # grader's rule
            if not ok.any():
                continue
            hid = np.full((ok.sum(), 1), h, np.float32)
            X.append(np.concatenate([base[ok], hid], 1))
            y.append(tgt[ok])
    if not X:
        raise ValueError("no training rows; check stride / max_origin")
    return np.concatenate(X), np.concatenate(y)


# --------------------------------------------------------------------------
# fit / predict
# --------------------------------------------------------------------------
def fit(fit_dir, fit_file=None, stride=72, seed=0):
    """Cluster the turbines, then train one LightGBM per cluster."""
    import lightgbm as lgb

    art = load_fit(fit_dir)
    src = fit_file or art["fit_file"]
    panel = load_panel(src)
    print("ST-Tree fitting on {} ({:.1f} days)".format(
        os.path.basename(src), panel["steps"] / STEPS_PER_DAY))

    labels = fit_clusters(panel, seed=seed)
    sizes = np.bincount(labels, minlength=N_CLUSTERS)
    print("clusters: {}".format(sizes.tolist()))

    out = os.path.join(fit_dir, "sttree")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "clusters.npy"), labels)

    for c in range(N_CLUSTERS):
        turbs = np.where(labels == c)[0]
        if turbs.size == 0:
            continue
        X, y = build_training_table(panel, turbs, stride=stride)
        model = lgb.train(LGB_PARAMS, lgb.Dataset(X, label=y),
                          num_boost_round=300)
        model.save_model(os.path.join(out, "lgb_{}.txt".format(c)))
        print("  cluster {}: {} turbines, {} rows, mae {:.1f} kW (in-sample)"
              .format(c, turbs.size, len(y),
                      np.abs(model.predict(X) - y).mean()))

    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump({"delta": DELTA, "n_clusters": N_CLUSTERS,
                   "fit_file": os.path.abspath(src), "stride": stride}, fh,
                  indent=2)
    print("wrote {}".format(out))


class STTree(object):
    """Loaded ST-Tree ready for inference."""

    def __init__(self, fit_dir):
        import lightgbm as lgb
        d = os.path.join(fit_dir, "sttree")
        self.labels = np.load(os.path.join(d, "clusters.npy"))
        self.models = {}
        for c in range(N_CLUSTERS):
            p = os.path.join(d, "lgb_{}.txt".format(c))
            if os.path.exists(p):
                self.models[c] = lgb.Booster(model_file=p)

    def predict(self, panel):
        """
        Args:
            panel: a window panel from load_panel(infile).
        Returns:
            (134, DELTA) predicted power in kW, clipped at zero.
        """
        o = panel["steps"]
        out = np.zeros((CAPACITY, DELTA), dtype=np.float64)
        for c, model in self.models.items():
            turbs = np.where(self.labels == c)[0]
            if turbs.size == 0:
                continue
            base = _rows_for_origin(panel, o, turbs)
            for h in range(1, DELTA + 1):
                hid = np.full((turbs.size, 1), h, np.float32)
                out[turbs, h - 1] = model.predict(
                    np.concatenate([base, hid], 1))
        return np.clip(out, 0.0, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--fit-dir", default=os.path.join(_HERE, "fit"))
    ap.add_argument("--fit-file", default=None)
    ap.add_argument("--stride", type=int, default=72)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    fit(a.fit_dir, a.fit_file, a.stride, a.seed)


if __name__ == "__main__":
    main()
