# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/FDSTT/program.py

Description:
    THE FULL FDSTT MODEL -- all four modules, behind this repo's standard
    forecast(envs) -> (134, 288, 1) contract. Runs unchanged under both
    graders:

        python evaluation/evaluator.py          leaderboard/FDSTT/program.py
        python evaluation/competition_score.py  leaderboard/FDSTT/program.py

    Modules:
        1. graphs.py  spatial + semantic graphs        (fitted, in fit/)
        2. dmst.py    GRU seq2seq, all 288 steps       (ckpt/best.pt)
        3. sttree.py  3x LightGBM, first 30 steps      (fit/sttree/)
        4. below      Algorithm 1, rule-based fusion

    Set FDSTT_DMST_ONLY=1 to bypass modules 3 and 4 and score DMST alone --
    that is the comparison against HIK's own ablation (41.41 DMST-only vs
    40.96 full, on their offline set).

Algorithm 1, from the paper Section 3.5:
    s = Mean(X_p[:, -5:, :])                    mean power over the last 5 steps
    if s < 350 or s > 700:                      extreme regime
        Y[:, :30]  = Y_tree[:, :30]
    else:
        Y[:, :30]  = 0.5*Y_tree + 0.5*Y_dmst
    Y[:, 30:] = Y_dmst[:, 30:] + 15             flat bias correction

    The +15 kW is a hardcoded correction for DMST systematically
    underpredicting at high power. Their ablation puts it at 0.086 --
    a fifth of their margin over their own DMST.

Environment:
    FDSTT_CKPT       checkpoint dir with best.pt   (default: ./ckpt)
    FDSTT_FIT        fitted-artifact dir           (default: from checkpoint)
    FDSTT_DEVICE     cuda / cuda:1 / cpu
    FDSTT_DMST_ONLY  1 to skip ST-Tree and the ensemble
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_HERE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from data import load_fit, load_panel, window_inputs, INPUT_LEN, RAW_COLS  # noqa: E402
from dmst import DMST, N_OTHER                                            # noqa: E402

CAPACITY = 134
DELTA_LOW, DELTA_UP = 350.0, 700.0     # paper: power-level thresholds
PHI = 15.0                             # paper: baseline value
BETA = 30                              # paper: handover step
_STATE = None


def _device():
    want = os.environ.get("FDSTT_DEVICE")
    if want:
        return torch.device(want)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _leakage_guard(art, path_to_test_x):
    """Refuse a full-245-day fit scored on val windows carved from those days."""
    if path_to_test_x is None:
        return
    if "val_window" in os.path.abspath(path_to_test_x) and \
            float(art.get("fit_days", 0)) > 200:
        raise RuntimeError(
            "LEAKAGE: fit covers {:.0f} days but scoring the val windows, "
            "whose targets lie inside that period. Use the search fit for "
            "evaluator.py and the full fit only for competition_score.py."
            .format(float(art["fit_days"])))


def _load_once(path_to_test_x=None):
    global _STATE
    if _STATE is not None:
        _leakage_guard(_STATE["art"], path_to_test_x)
        return _STATE

    ckpt_dir = os.environ.get("FDSTT_CKPT", os.path.join(_HERE, "ckpt"))
    ck = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
    fit_dir = os.environ.get(
        "FDSTT_FIT", ck.get("fit_dir", os.path.join(_HERE, "fit")))
    art = load_fit(fit_dir)
    _leakage_guard(art, path_to_test_x)

    dev = _device()
    model = DMST(n_wspd=ck["n_wspd"], n_other=N_OTHER, horizon=288).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    tree = None
    if os.environ.get("FDSTT_DMST_ONLY", "") != "1":
        try:
            from sttree import STTree
            tree = STTree(fit_dir)
        except Exception as exc:                       # noqa: BLE001
            sys.stderr.write(
                "ST-Tree unavailable ({}); running DMST alone. "
                "Fit it with: python sttree.py fit --fit-dir {}\n"
                .format(exc, fit_dir))

    _STATE = {"model": model, "tree": tree, "art": art, "device": dev,
              "power_scale": float(ck["power_scale"]),
              "dup_self": bool(art["dup_self"])}
    return _STATE


def ensemble(y_dmst, y_tree, recent_power, beta=BETA, phi=PHI,
             lo=DELTA_LOW, up=DELTA_UP):
    """
    Description:
        Algorithm 1. No learning, no fitted weights -- three hand-tuned
        constants. The paper's stated reason is that a meta-learner overfits,
        because the wind distribution shifts between train and test.
    Args:
        y_dmst: (134, 288) kW.
        y_tree: (134, 30) kW, or None to fall back to DMST alone.
        recent_power: scalar, mean Patv over the last 5 input steps.
    Returns:
        (134, 288) kW.
    """
    out = y_dmst.copy()
    if y_tree is not None:
        k = min(beta, y_tree.shape[1], out.shape[1])
        if recent_power < lo or recent_power > up:
            out[:, :k] = y_tree[:, :k]
        else:
            out[:, :k] = 0.5 * y_tree[:, :k] + 0.5 * y_dmst[:, :k]
    out[:, beta:] = out[:, beta:] + phi
    return np.clip(out, 0.0, None)


def forecast(envs):
    """
    Description:
        Predict 48 hours of active power for all 134 turbines from one window.
    Args:
        envs: settings dict containing "path_to_test_x".
    Returns:
        (134, 288, 1) float array, active power in kW.
    """
    st = _load_once(envs.get("path_to_test_x"))
    horizon = envs.get("output_len", 288)

    turb, enc, fut, last = window_inputs(
        envs["path_to_test_x"], st["art"], input_len=INPUT_LEN,
        horizon=horizon, dup_self=st["dup_self"])

    dev = st["device"]
    with torch.no_grad():
        y_dmst = (st["model"](turb.to(dev), enc.to(dev), fut.to(dev),
                              last.to(dev)) * st["power_scale"])
    y_dmst = y_dmst.cpu().numpy().astype(np.float64)[:, :, 0]   # (134, 288)

    y_tree, s = None, 0.0
    if st["tree"] is not None:
        panel = load_panel(envs["path_to_test_x"])
        y_tree = st["tree"].predict(panel)
        patv = panel["values"][:, -5:, RAW_COLS[3:].index("Patv")]
        s = float(np.nanmean(np.where(np.isfinite(patv), patv, 0.0)))

    out = ensemble(y_dmst, y_tree, s)
    return out[:, :horizon, None]
