# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/bigscity/bigscity_ensemble.py

Description:
    Team BUAA_BIGSCity's KDD Cup 2022 submission (rank 11 of 2490), wrapped
    behind this repo's forecast(envs) contract. Runs their *actual submitted
    artefact* -- submit.zip from github.com/BUAABIGSCity/KDDCUP2022, with
    trained checkpoints, precomputed graphs and normalisation statistics --
    so it is inference only; no training required.

    REPRODUCTION CONFIRMED against final_phase_test:
        published leaderboard : -45.36026
        competition_score.py  :  45.360267
        (mae 41.150749, rmse 49.569786, accuracy 62.0971%, 142 windows)

    Their prediction is fully deterministic, so that match also validates
    the scorer end to end: competition_score.py is equivalent to Baidu's
    evaluation.py on a real submission.

    See bigscity_ensemble.md for the architecture walkthrough.

Inference recipe (verbatim from vendor/prepare.py + vendor/predict.py):
    * 5x AGCRN, one per DTW-graph fold, var_len 6 (5 features + first
      difference of Patv), add_apt True, fused by inverse-weight averaging
      with the per-fold weights below.
    * 1x MTGNN, geo graph (Gaussian kernel, thresholded at 0.8, binarised),
      var_len 5, add_apt False.
    * final = 0.4 * AGCRN_fused + 0.6 * MTGNN

Quirk worth knowing (it is theirs, not a bug here):
    data_mean_214.npy / data_scale_214.npy are shape (1, 134, 1, 1) -- ONE
    statistic per turbine, broadcast across all five feature channels. Wspd,
    Etmp, Itmp and Prtv are therefore standardised using Patv's per-turbine
    mean and std. Reproduced as-is; changing it would not be their model.

Setup (once):
    cd leaderboard/bigscity
    mkdir -p vendor && cd vendor
    unzip /path/to/submit.zip
    mv submit/* . && rmdir submit
    # keeps: AGCRN.py MTGNN.py npy/ kfold_dtw_5_data_diff/
    # the rest of their repo (training loop, dataset, losses) is unused here

Usage:
    SDWPF_TEST_DIR=/path/to/final_phase_test \
        python evaluation/competition_score.py \
               leaderboard/bigscity/bigscity_ensemble.py

Environment:
    SDWPF_BIGSCITY_DIR   vendored artefact dir (default: ./vendor beside this file)
    SDWPF_DEVICE         "cuda", "cuda:1", "cpu" (default: cuda if available)

Citation:
    Jiang, Han & Wang. BUAA_BIGSCity: Spatial-Temporal Graph Neural Network
    for Wind Power Forecasting in Baidu KDD CUP 2022. arXiv:2302.11159.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CAPACITY = 134
INPUT_LEN = 144
OUTPUT_LEN = 288
TRAIN_DAYS = 214          # selects npy/data_mean_214.npy and data_scale_214.npy

_HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.environ.get("SDWPF_BIGSCITY_DIR", os.path.join(_HERE, "vendor"))

# Their column order after preprocessing, and the seven they select.
COLUMNS = ["weekday", "time", "Wspd", "Wdir", "Etmp", "Itmp",
           "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
SELECT = ["weekday", "time", "Wspd", "Etmp", "Itmp", "Prtv", "Patv"]
SELECT_IND = [COLUMNS.index(c) for c in SELECT]   # -> [0, 1, 2, 4, 5, 10, 11]

# --- prepare.py: prep_env() model_list / model_list2 -------------------------
AGCRN_MODELS = [
    {"checkpoints_in": "31543_AGCRN", "best": 2, "ind": 0, "weight": 0.321763283},
    {"checkpoints_in": "94237_AGCRN", "best": 0, "ind": 1, "weight": 0.495282746},
    {"checkpoints_in": "43292_AGCRN", "best": 1, "ind": 2, "weight": 0.356973800},
    {"checkpoints_in": "98201_AGCRN", "best": 1, "ind": 3, "weight": 0.298808416},
    {"checkpoints_in": "8204_AGCRN",  "best": 0, "ind": 4, "weight": 0.408473526},
]
MTGNN_MODELS = [
    {"checkpoints_in": "55237_MTGNN", "best": 0, "ind": -1, "weight": 1.0},
]
AGCRN_MIX, MTGNN_MIX = 0.4, 0.6

_STATE = None             # populated once, on the first forecast() call


def _device():
    """Resolve the inference device; honours SDWPF_DEVICE, falls back to CPU."""
    want = os.environ.get("SDWPF_DEVICE")
    if want:
        return torch.device(want)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _geo_graph(epsilon=0.8, binary=True):
    """
    Description:
        Gaussian-kernel the pairwise distance matrix, threshold it, binarise.
        Verbatim from predict.py's graph_type == "geo" branch.
    """
    g = np.load(os.path.join(VENDOR, "npy", "geo_graph.npy"))
    g = np.exp(-np.square(g / g.flatten().std()))
    g[g < epsilon] = 0
    if binary:
        g[g >= epsilon] = 1
    return g


def _dtw_graph(ind, topk=5, K=5):
    """Load the fold-specific top-k DTW similarity graph."""
    return np.load(os.path.join(
        VENDOR, "npy", "dtw_graph_top{}_{}_{}.npy".format(topk, ind, K)))


def _build(model_cls, cfg, adj, ckpt_dir, best, device):
    """Instantiate one model, load its checkpoint, put it in eval mode."""
    model = model_cls(config=cfg, adj_mx=adj).to(device)
    path = os.path.join(VENDOR, "kfold_dtw_5_data_diff", ckpt_dir,
                        "model_{}.pt".format(best))
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_once():
    """
    Description:
        Build all six models and load the normalisation statistics. Cached at
        module level: competition_score.py calls forecast() once per window,
        and re-instantiating six graph networks 142 times would dominate the
        runtime.
    Returns:
        A dict with the model list, data_mean/data_scale tensors, and device.
    """
    global _STATE
    if _STATE is not None:
        return _STATE

    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)          # AGCRN.py / MTGNN.py live there
    from AGCRN import AGCRN
    from MTGNN import MTGNN

    device = _device()

    base = {
        "input_len": INPUT_LEN, "output_len": OUTPUT_LEN, "capacity": CAPACITY,
        "dropout": 0, "rnn_units": 64, "embed_dim": 10, "num_layers": 2,
        "cheb_order": 2, "device": device,
    }

    models = []
    for m in AGCRN_MODELS:
        cfg = dict(base, var_len=6, data_diff=True, add_apt=True)
        models.append((
            _build(AGCRN, cfg, _dtw_graph(m["ind"]),
                   m["checkpoints_in"], m["best"], device),
            m["weight"], "agcrn"))
    for m in MTGNN_MODELS:
        cfg = dict(base, var_len=5, data_diff=False, add_apt=False)
        models.append((
            _build(MTGNN, cfg, _geo_graph(),
                   m["checkpoints_in"], m["best"], device),
            m["weight"], "mtgnn"))

    mean = np.load(os.path.join(VENDOR, "npy",
                                "data_mean_{}.npy".format(TRAIN_DAYS)))
    scale = np.load(os.path.join(VENDOR, "npy",
                                 "data_scale_{}.npy".format(TRAIN_DAYS)))

    _STATE = {
        "models": models,
        "data_mean": torch.FloatTensor(mean).to(device),    # (1, 134, 1, 1)
        "data_scale": torch.FloatTensor(scale).to(device),
        "device": device,
    }
    return _STATE


def _read_window(path, capacity):
    """
    Description:
        Reproduce TestPGL4WPFDataset's preprocessing without importing their
        dataset module: reorder columns, prepend weekday and time-of-day
        indices, zero-fill NaNs, reshape turbine-major.
    Args:
        path: the window infile CSV.
        capacity: number of turbines (134).
    Returns:
        A float32 array of shape (1, capacity, T, 12).
    """
    df = pd.read_csv(path)

    feats = [c for c in df.columns
             if "Patv" not in c and c not in ("Day", "Tmstamp", "TurbID")]
    out = df[feats].copy()
    out["Patv"] = df["Patv"]

    # 'HH:MM' -> 0..143. Their time_dict is exactly this arithmetic.
    hhmm = df["Tmstamp"].astype(str).str.split(":", expand=True).astype(int)
    out.insert(0, "time", hhmm[0] * 6 + hhmm[1] // 10)
    out.insert(0, "weekday", df["Day"] % 7)

    assert list(out.columns) == COLUMNS, \
        "column order drifted: {}".format(list(out.columns))

    arr = out.fillna(0).values.astype(np.float32)
    arr = arr.reshape(capacity, -1, len(COLUMNS))       # turbine-major
    return np.expand_dims(arr, 0)                        # (1, N, T, F)


def forecast(envs):
    """
    Description:
        Run BUAA_BIGSCity's six-model ensemble over one test window and
        return their fused prediction.
    Args:
        envs: settings dict; must contain "path_to_test_x".
    Returns:
        A float array of shape (capacity, output_len, 1), active power in kW.
    """
    capacity = envs.get("capacity", CAPACITY)
    output_len = envs.get("output_len", OUTPUT_LEN)

    st = _load_once()
    device, mean, scale = st["device"], st["data_mean"], st["data_scale"]

    x = _read_window(envs["path_to_test_x"], capacity)
    x = torch.FloatTensor(x[:, :, -INPUT_LEN:, SELECT_IND]).to(device)

    agcrn, agcrn_w, mtgnn, mtgnn_w = [], [], [], []
    with torch.no_grad():
        for model, weight, kind in st["models"]:
            pred = model(x, None, mean, scale)                    # (B, N, T)
            pred = F.relu(pred * scale[:, :, :, -1] + mean[:, :, :, -1])
            pred = pred.cpu().numpy()[0]                          # (N, T)
            (agcrn if kind == "agcrn" else mtgnn).append(pred)
            (agcrn_w if kind == "agcrn" else mtgnn_w).append(weight)

    def _fuse(preds, weights):
        """Inverse-weight average, exactly as predict.py's fusion loop."""
        inv = np.array([1.0 / w for w in weights], dtype=np.float64)
        return np.tensordot(inv, np.stack(preds), axes=(0, 0)) / inv.sum()

    fused = AGCRN_MIX * _fuse(agcrn, agcrn_w) + MTGNN_MIX * _fuse(mtgnn, mtgnn_w)
    return fused[:, :output_len, None]                            # (N, T, 1)