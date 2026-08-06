# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/FDSTT/program.py

Description:
    The scoreable entry point for FDSTT's DMST module. Exposes the same
    forecast(envs) -> (134, 288, 1) contract every other model in this repo
    uses, so it runs unchanged under BOTH graders:

        python evaluation/evaluator.py          leaderboard/FDSTT/program.py
        python evaluation/competition_score.py  leaderboard/FDSTT/program.py

    This is DMST ALONE -- module 2 of 4. ST-Tree and the ensemble rule are not
    built yet, so expect this to land near HIK's own DMST-only ablation
    (41.41 on their offline set) rather than their full 44.917.

Environment:
    FDSTT_CKPT    checkpoint dir containing best.pt   (default: ./ckpt)
    FDSTT_FIT     fitted-artifact dir                 (default: from checkpoint)
    FDSTT_DEVICE  cuda / cuda:1 / cpu                 (default: cuda if present)

Guard:
    Scoring a full-245-day fit against the val windows would be leakage -- the
    val targets are inside those 245 days. That combination is refused rather
    than silently producing an inflated score.
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

from data import load_fit, window_inputs, INPUT_LEN                 # noqa: E402
from dmst import DMST, N_OTHER                                      # noqa: E402

CAPACITY = 134
_STATE = None


def _device():
    want = os.environ.get("FDSTT_DEVICE")
    if want:
        return torch.device(want)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _leakage_guard(art, path_to_test_x):
    """
    Description:
        Refuse the one combination that is silently wrong: a fit built on the
        full 245 days, scored on the val windows carved out of those same 245
        days. Nothing else about the run would look unusual.
    """
    if path_to_test_x is None:
        return
    scoring_val = "val_window" in os.path.abspath(path_to_test_x)
    fit_days = float(art.get("fit_days", 0))
    if scoring_val and fit_days > 200:
        raise RuntimeError(
            "LEAKAGE: fit covers {:.0f} days but scoring the val windows, "
            "whose targets lie inside that period. Use the search fit "
            "(FDSTT_FIT=fit_val) for evaluator.py, and the full fit only for "
            "competition_score.py.".format(fit_days))


def _load_once(path_to_test_x=None):
    """Build the model once; the graders call forecast() per window."""
    global _STATE
    if _STATE is not None:
        _leakage_guard(_STATE["art"], path_to_test_x)
        return _STATE

    ckpt_dir = os.environ.get("FDSTT_CKPT", os.path.join(_HERE, "ckpt"))
    ck = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location="cpu")
    fit_dir = os.environ.get("FDSTT_FIT", ck.get("fit_dir",
                                                 os.path.join(_HERE, "fit")))
    art = load_fit(fit_dir)
    _leakage_guard(art, path_to_test_x)

    dev = _device()
    model = DMST(n_wspd=ck["n_wspd"], n_other=N_OTHER, horizon=288).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    _STATE = {"model": model, "art": art, "device": dev,
              "power_scale": float(ck["power_scale"]),
              "dup_self": bool(art["dup_self"])}
    return _STATE


def forecast(envs):
    """
    Description:
        Predict 48 hours of active power for all 134 turbines from one
        competition window.
    Args:
        envs: settings dict containing "path_to_test_x".
    Returns:
        (134, 288, 1) float array, active power in kW.
    """
    st = _load_once(envs.get("path_to_test_x"))
    turb, enc, fut, last = window_inputs(
        envs["path_to_test_x"], st["art"],
        input_len=INPUT_LEN, horizon=envs.get("output_len", 288),
        dup_self=st["dup_self"])

    dev = st["device"]
    with torch.no_grad():
        pred = st["model"](turb.to(dev), enc.to(dev), fut.to(dev),
                           last.to(dev)) * st["power_scale"]
    out = pred.cpu().numpy().astype(np.float64)
    return np.clip(out, 0.0, None)          # power cannot be negative
