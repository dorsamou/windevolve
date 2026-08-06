# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/FDSTT/data.py

Description:
    Data pipeline for FDSTT (team HIK, rank 1, Baidu KDD Cup 2022). Turns the
    SDWPF CSVs into the tensors dmst.py expects, and assembles the same
    features at inference time so forecast(envs) matches training exactly.

    Split policy -- IDENTICAL to every other model in this repo:

        val_windows.py --val-days 48 --n-windows 24
            train  = days 1-197   (80.4%)   -> evaluation/splits/val_train.csv
            val    = days 198-245 (19.6%)   -> evaluation/splits/val_windows/
            test   = final_phase_test       -> locked, scored once

    Every model therefore reports two numbers, both from the official metric:
        evaluator.py           -> competition score on the 24 val windows
        competition_score.py   -> competition score on final_phase_test

    ANTI-LEAKAGE. FDSTT fits four things from data, and three of them would
    silently leak if built from the full 245-day CSV:

        G_d  spatial graph        location file only        cannot leak
        G_s  semantic graph       Wspd time series          LEAKS
        norm mean/std             all features              LEAKS
        k-means clusters          Wspd correlations         LEAKS (ST-Tree)

    All fitted artifacts are therefore built ONCE from --fit-file (default
    VAL_TRAIN_FILE, days 1-197), written to fit/, and LOADED at inference.
    Nothing is ever recomputed from a window at predict time.

    A window's 14 days of input history may reach back before step 28368 --
    that is data the forecaster genuinely has at prediction time. Only targets
    are forbidden from crossing the line, and val_windows.py enforces that.

    Fitting on 197 days rather than 245 costs some accuracy on
    final_phase_test relative to the competition teams, who had all 245. That
    is the price of one model producing both scores, and it is the same price
    every other model in this repo pays. Pass --fit-file <245-day csv> to
    build a second, leaderboard-comparable fit -- but never score THAT one on
    the val windows.

Feature layout (paper Section 4.1.2):
    encoder, 14 columns before the embedding is prepended:
        [0:4]   wspd block: own, spatial-neighbour mean, own, semantic mean
        [4:14]  Etmp Itmp Wdir Ndir Pab1 Prtv Patv Hour x y
    future covariates, 3 columns:
        Hour x y                    -- all that is knowable about the future
    target block, 7 columns, RAW and unnormalised:
        Patv Pab1 Pab2 Pab3 Wdir Ndir Wspd
        column 0 is the regression target; the other six exist only so
        FilterHuberLoss.valid_mask can reproduce the grader's exclusions.

Usage:
    python leaderboard/FDSTT/data.py fit        # build fit/ from val_train.csv
    python leaderboard/FDSTT/data.py check      # train/inference parity test
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from graphs import build_spatial_graph, build_semantic_graph, load_locations  # noqa: E402
from dmst import aggregate_wspd, INPUT_LEN, OUTPUT_LEN                        # noqa: E402

CAPACITY = 134
STEPS_PER_DAY = 144
FIT_DIR = os.path.join(_HERE, "fit")

RAW_COLS = ["TurbID", "Day", "Tmstamp", "Wspd", "Wdir", "Etmp", "Itmp",
            "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
OTHER_COLS = ["Etmp", "Itmp", "Wdir", "Ndir", "Pab1", "Prtv", "Patv",
              "Hour", "x", "y"]
FUT_COLS = ["Hour", "x", "y"]
TARGET_COLS = ["Patv", "Pab1", "Pab2", "Pab3", "Wdir", "Ndir", "Wspd"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_panel(csv_path, capacity=CAPACITY):
    """
    Description:
        Read an SDWPF CSV into a turbine-major panel. Sorts explicitly rather
        than trusting file order -- a mis-ordered panel would misalign every
        turbine against its graph row and produce plausible-looking nonsense.

    Returns:
        dict with
            values: (134, T, 10) float32, RAW_COLS[3:] in order, NaNs kept
            hour:   (T,) float32, 0-23.983
            steps:  T
    """
    df = pd.read_csv(csv_path, low_memory=False)
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError("CSV missing columns: {}".format(missing))
    df = df[RAW_COLS]

    hhmm = df["Tmstamp"].astype(str).str.split(":", expand=True).astype(int)
    slot = (hhmm[0] * 6 + hhmm[1] // 10).to_numpy()

    order = np.lexsort((slot, df["Day"].to_numpy(), df["TurbID"].to_numpy()))
    df, slot = df.iloc[order].reset_index(drop=True), slot[order]

    n = len(df)
    if n % capacity:
        raise ValueError("{} rows not divisible by {}".format(n, capacity))
    steps = n // capacity

    values = df[RAW_COLS[3:]].to_numpy(dtype=np.float32).reshape(
        capacity, steps, len(RAW_COLS) - 3)
    hour = (slot[:steps] / 6.0).astype(np.float32)
    return {"values": values, "hour": hour, "steps": steps}


def _col(panel, name):
    """Extract one raw sensor column as (134, T)."""
    return panel["values"][:, :, RAW_COLS[3:].index(name)]


# --------------------------------------------------------------------------
# fitted artifacts -- the anti-leakage boundary
# --------------------------------------------------------------------------
def fit(fit_file, out_dir=FIT_DIR, k=5, dup_self=True, normalise="none"):
    """
    Description:
        Build every data-derived artifact FDSTT needs, from the training
        portion ONLY, and write them to out_dir. Anything not produced here
        must not be computed at inference time.

    Args:
        fit_file: CSV to fit on. Default caller passes VAL_TRAIN_FILE
            (days 1-197). Pass the 245-day CSV only for a test-only fit.
        normalise: passed to build_semantic_graph; "none" is the paper.

    Returns:
        The artifact dict, also saved as npz + json.
    """
    os.makedirs(out_dir, exist_ok=True)
    panel = load_panel(fit_file)
    steps = panel["steps"]
    print("fitting on {} ({} steps = {:.1f} days)".format(
        os.path.basename(fit_file), steps, steps / STEPS_PER_DAY))

    loc = load_locations()
    g_d = build_spatial_graph(loc, k=k)
    wspd = _col(panel, "Wspd")                            # (134, T)
    g_s = build_semantic_graph(wspd, k=k, normalise=normalise)

    # Per-feature global normalisation over the fit period, NaNs ignored.
    wspd_block = aggregate_wspd(wspd.T, g_d, g_s, dup_self=dup_self)
    other = _assemble_other(panel, loc)                   # (T, 134, 10)
    feats = np.concatenate([wspd_block, other], axis=-1)  # (T, 134, 14)

    flat = feats.reshape(-1, feats.shape[-1])
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    # Power gets its own scalar so the Huber delta can be converted back into
    # kilowatts. Feeding raw kW into the decoder saturates the GRU gates.
    patv = _col(panel, "Patv")
    power_scale = float(np.nanpercentile(patv, 99.9))
    if not np.isfinite(power_scale) or power_scale <= 0:
        power_scale = 1500.0

    np.savez(os.path.join(out_dir, "artifacts.npz"),
             g_d=g_d, g_s=g_s, mean=mean, std=std, loc=loc)
    meta = {"fit_file": os.path.abspath(fit_file), "fit_steps": int(steps),
            "fit_days": round(steps / STEPS_PER_DAY, 2), "k": k,
            "dup_self": bool(dup_self), "semantic_normalise": normalise,
            "power_scale": power_scale, "n_feat": int(feats.shape[-1])}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("  power_scale {:.1f} kW | features {} | wrote {}".format(
        power_scale, feats.shape[-1], out_dir))
    return {"g_d": g_d, "g_s": g_s, "mean": mean, "std": std,
            "loc": loc, **meta}


def load_fit(fit_dir=FIT_DIR):
    """Load the artifacts written by fit(). Fails loudly if absent."""
    npz = os.path.join(fit_dir, "artifacts.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            "no fitted artifacts at {}. Run: python data.py fit".format(fit_dir))
    a = dict(np.load(npz))
    with open(os.path.join(fit_dir, "meta.json")) as fh:
        a.update(json.load(fh))
    return a


# --------------------------------------------------------------------------
# feature assembly, shared by training and inference
# --------------------------------------------------------------------------
def _assemble_other(panel, loc):
    """
    Description:
        Build the 10 non-wind-speed encoder columns in the paper's order.
    Returns:
        (T, 134, 10) float32.
    """
    v, steps = panel["values"], panel["steps"]
    idx = {n: RAW_COLS[3:].index(n) for n in
           ("Etmp", "Itmp", "Wdir", "Ndir", "Pab1", "Prtv", "Patv")}
    cols = [v[:, :, idx[n]].T for n in
            ("Etmp", "Itmp", "Wdir", "Ndir", "Pab1", "Prtv", "Patv")]
    cols.append(np.broadcast_to(panel["hour"][:, None], (steps, CAPACITY)))
    cols.append(np.broadcast_to(loc[:, 0][None, :], (steps, CAPACITY)))
    cols.append(np.broadcast_to(loc[:, 1][None, :], (steps, CAPACITY)))
    return np.stack(cols, axis=-1).astype(np.float32)


def build_features(panel, art, dup_self=True):
    """
    Description:
        The single code path both training and inference go through. Any
        divergence between the two is the classic silent train/serve skew, so
        there is deliberately only one implementation.

    Returns:
        feats: (T, 134, 14) normalised, NaN -> 0 (i.e. the feature mean)
        fut:   (T, 134, 3)  normalised Hour, x, y
        raw:   (T, 134, 7)  UNNORMALISED target block, NaNs preserved
    """
    wspd = _col(panel, "Wspd")
    wspd_block = aggregate_wspd(wspd.T, art["g_d"], art["g_s"],
                                dup_self=dup_self)
    other = _assemble_other(panel, art["loc"])
    feats = np.concatenate([wspd_block, other], axis=-1)

    feats = (feats - art["mean"]) / art["std"]
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    fut_idx = [4 + OTHER_COLS.index(c) for c in FUT_COLS]
    fut = feats[:, :, fut_idx]

    v = panel["values"]
    raw = np.stack([v[:, :, RAW_COLS[3:].index(c)].T for c in TARGET_COLS],
                   axis=-1)
    return feats.astype(np.float32), fut.astype(np.float32), raw.astype(np.float32)


# --------------------------------------------------------------------------
# training dataset
# --------------------------------------------------------------------------
class FDSTTWindows(Dataset):
    """
    Description:
        One sample = one (turbine, time origin) pair. All 134 turbines share
        one model, so the turbine index is just part of the sample, not a
        separate model.

        Full density is ~34,500 origins x 134 turbines = 4.6M samples per
        epoch, which the 288-step autoregressive decoder cannot chew through.
        Consecutive origins share 431 of their 432 input steps, so striding
        costs almost nothing: stride=6 gives ~770k.

    Args:
        stride: spacing between origins.
        max_origin: exclusive upper bound on the TARGET end. Leave None to use
            the whole file -- correct when fitting on val_train.csv, which is
            already truncated at the validation boundary.
    """

    def __init__(self, panel, art, input_len=INPUT_LEN, horizon=OUTPUT_LEN,
                 stride=6, dup_self=True, max_origin=None):
        self.feats, self.fut, self.raw = build_features(panel, art, dup_self)
        self.input_len, self.horizon = input_len, horizon
        self.power_scale = float(art["power_scale"])

        steps = panel["steps"]
        last = (steps if max_origin is None else max_origin) - horizon
        origins = np.arange(input_len, last + 1, stride)
        if origins.size == 0:
            raise ValueError("no valid origins: series too short")
        self.index = [(int(o), t) for o in origins for t in range(CAPACITY)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        o, t = self.index[i]
        enc = self.feats[o - self.input_len:o, t]              # (432, 14)
        fut = self.fut[o:o + self.horizon, t]                  # (288, 3)
        raw = self.raw[o:o + self.horizon, t]                  # (288, 7)
        last = self.raw[o - 1, t, 0] / self.power_scale        # y^t_i
        return (torch.tensor(t),
                torch.from_numpy(enc.copy()),
                torch.from_numpy(fut.copy()),
                torch.tensor([np.nan_to_num(last, nan=0.0)], dtype=torch.float32),
                torch.from_numpy(raw.copy()))


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------
def window_inputs(infile, art, input_len=INPUT_LEN, horizon=OUTPUT_LEN,
                  dup_self=True):
    """
    Description:
        Assemble model inputs for one competition window. Identical code path
        to training: same build_features, same fitted artifacts.

        The infile holds 14 days (2016 steps) per turbine; the encoder takes
        the last `input_len` of them. Future covariates are extrapolated from
        the window's own clock -- Hour advances, x and y are fixed -- which is
        legitimate, since neither depends on unseen data.

    Returns:
        turb_idx (134,), enc (134, input_len, 14), fut (134, horizon, 3),
        last_power (134, 1) already divided by power_scale.
    """
    panel = load_panel(infile)
    if panel["steps"] < input_len:
        raise ValueError("window has {} steps, need {}".format(
            panel["steps"], input_len))

    feats, _, raw = build_features(panel, art, dup_self)
    enc = feats[-input_len:].transpose(1, 0, 2)                # (134, L, 14)

    # future clock, continuing from the window's last timestamp
    last_hour = panel["hour"][-1]
    hours = (last_hour + (np.arange(1, horizon + 1) / 6.0)) % 24.0
    m, s = art["mean"], art["std"]
    hi = 4 + OTHER_COLS.index("Hour")
    xi, yi = 4 + OTHER_COLS.index("x"), 4 + OTHER_COLS.index("y")
    fut = np.empty((CAPACITY, horizon, 3), dtype=np.float32)
    fut[:, :, 0] = (hours[None, :] - m[hi]) / s[hi]
    fut[:, :, 1] = ((art["loc"][:, 0] - m[xi]) / s[xi])[:, None]
    fut[:, :, 2] = ((art["loc"][:, 1] - m[yi]) / s[yi])[:, None]

    last = np.nan_to_num(raw[-1, :, 0], nan=0.0) / art["power_scale"]
    return (torch.arange(CAPACITY),
            torch.from_numpy(np.ascontiguousarray(enc)),
            torch.from_numpy(fut),
            torch.from_numpy(last.astype(np.float32)).unsqueeze(1))


# --------------------------------------------------------------------------
# parity check
# --------------------------------------------------------------------------
def check(fit_dir=FIT_DIR, csv=None):
    """
    Description:
        Verify the training path and the inference path produce IDENTICAL
        encoder features for the same window. Train/serve skew here would be
        invisible in every other test -- both sides run without error and both
        produce plausible numbers.
    """
    from paths import VAL_TRAIN_FILE
    art = load_fit(fit_dir)
    src = csv or VAL_TRAIN_FILE
    panel = load_panel(src)

    ds = FDSTTWindows(panel, art, stride=1000)
    t_idx, enc_train, fut_train, last_train, raw = ds[0]
    o, turb = ds.index[0]
    print("origin step {} turbine {}".format(o, turb))

    tmp = "/tmp/_fdstt_parity.csv"
    df = pd.read_csv(src, low_memory=False)[RAW_COLS]
    steps = panel["steps"]
    lo, hi = o - INPUT_LEN, o
    idx = np.concatenate([np.arange(t * steps + lo, t * steps + hi)
                          for t in range(CAPACITY)])
    df.iloc[idx].to_csv(tmp, index=False, na_rep="")

    _, enc_inf, fut_inf, last_inf = window_inputs(tmp, art)
    d_enc = float((enc_inf[turb] - enc_train).abs().max())
    d_last = float((last_inf[turb] - last_train).abs().max())
    print("max |enc_train - enc_inference| = {:.3e}".format(d_enc))
    print("max |last_power difference|      = {:.3e}".format(d_last))
    assert d_enc < 1e-5 and d_last < 1e-5, "TRAIN/SERVE SKEW"

    from dmst import FilterHuberLoss
    keep = FilterHuberLoss.valid_mask(raw.unsqueeze(0))
    print("grader mask keeps {:.1f}% of this window's target steps".format(
        100 * float(keep.mean())))
    print("PARITY OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["fit", "check"])
    ap.add_argument("--fit-file", default=None,
                    help="default: VAL_TRAIN_FILE (days 1-197). Pass the "
                         "245-day CSV only for a test-only fit, and never "
                         "score that fit on the val windows.")
    ap.add_argument("--fit-dir", default=FIT_DIR)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--semantic-normalise", choices=["none", "corr"],
                    default="none")
    args = ap.parse_args()

    if args.cmd == "fit":
        from paths import VAL_TRAIN_FILE
        src = args.fit_file or VAL_TRAIN_FILE
        fit(src, out_dir=args.fit_dir, k=args.k,
            normalise=args.semantic_normalise)
    else:
        check(args.fit_dir)


if __name__ == "__main__":
    main()
