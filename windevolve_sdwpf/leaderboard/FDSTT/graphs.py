# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/hik/graphs.py

Description:
    Module 1 of 4 of FDSTT (team HIK, rank 1, Baidu KDD Cup 2022). Builds the
    two graphs the DMST encoder aggregates over. No training, no checkpoints --
    pure feature engineering, so this is the one part of FDSTT that can be
    verified by looking at it.

    HIK published no code. Everything here is read off the paper
    (arXiv/KDD '22 workshop, "Complementary Fusion of Deep Spatio-Temporal
    Network and Tree Model for Wind Power Forecasting"), Section 3.2.

    Spatial-aware graph G_d (Eq. 2):
        Euclidean distance between turbine coordinates, top-K nearest
        neighbours per node, binarised.

    Semantic-aware graph G_s (Eq. 3):
        "Differential similarity" -- the dot product of wind-speed FIRST
        DIFFERENCES between two turbines, summed over time:
            Sim(i, j) = sum_t (x_i^t - x_i^{t-1}) * (x_j^t - x_j^{t-1})
        then top-K most similar per node, binarised.

        Note this is UNNORMALISED. It is not Pearson correlation and not
        cosine similarity: a turbine whose wind speed swings harder gets a
        larger similarity to everything. The paper writes it this way, so it
        is written that way here. `normalise="corr"` is provided to test
        whether that was deliberate -- it is an ablation, not the paper.

    The paper's rationale for G_s: two turbines on opposite slopes can be
    metres apart yet anticorrelated, while distant turbines under the same
    weather front move together. G_d captures local wake structure, G_s
    captures global regime similarity.

Open question (ambiguity #4 in our notes):
    Figure 1's caption says the constructor "uses input turbine series to
    generate the graphs", which reads as rebuilding G_s per inference window.
    Section 3.2 reads as building it once. Both are supported here:
        build_semantic_graph(wspd_from_train)    -> static, default
        build_semantic_graph(wspd_from_window)   -> per-window
    G_d is static either way; turbines do not move.

Usage:
    python leaderboard/hik/graphs.py            # build static graphs, save npy
    python leaderboard/hik/graphs.py --inspect  # + sanity report

Outputs (to ./npy beside this file):
    g_spatial_k5.npy    (134, 134) float32, binary
    g_semantic_k5.npy   (134, 134) float32, binary
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from paths import TRAIN_FILE, LOCATION_FILE          # noqa: E402

CAPACITY = 134
TOP_K = 5                                            # paper: K = 5 for both
OUT_DIR = os.path.join(_HERE, "npy")


def _topk_binarise(score, k, larger_is_closer, include_self=False):
    """
    Description:
        Keep the k best neighbours per row and binarise, per Eq. 2. Self is
        excluded before ranking -- the paper defines N(i) as the k nearest
        *other* nodes, and a self-loop would be redundant with the CONCAT of
        x_i in Eqs. 4-6.
    Args:
        score: (N, N) similarity or distance matrix.
        k: neighbours per node.
        larger_is_closer: True for similarity, False for distance.
        include_self: add the diagonal back after ranking.
    Returns:
        (N, N) float32 binary adjacency. Row i marks the neighbours of i.
        Not symmetric -- top-k is a directed relation, and the paper does not
        symmetrise.
    """
    n = score.shape[0]
    s = score.astype(np.float64).copy()
    np.fill_diagonal(s, -np.inf if larger_is_closer else np.inf)

    order = np.argsort(-s if larger_is_closer else s, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=np.float32)
    adj[np.arange(n)[:, None], order] = 1.0
    if include_self:
        np.fill_diagonal(adj, 1.0)
    return adj


def load_locations(path=None):
    """
    Description:
        Read the turbine layout, ordered by TurbID so row i is turbine i+1
        and indices line up with every other (134, ...) array in the repo.
    Returns:
        (134, 2) float array of x, y coordinates in metres.
    """
    df = pd.read_csv(path or LOCATION_FILE).sort_values("TurbID")
    assert len(df) == CAPACITY, "expected {} turbines, got {}".format(
        CAPACITY, len(df))
    return df[["x", "y"]].to_numpy(dtype=np.float64)


def build_spatial_graph(locations=None, k=TOP_K, include_self=False):
    """
    Description:
        G_d, Eq. 2. Pairwise Euclidean distance, top-k nearest, binarised.
    Returns:
        (134, 134) float32 binary adjacency.
    """
    loc = load_locations() if locations is None else locations
    diff = loc[:, None, :] - loc[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return _topk_binarise(dist, k, larger_is_closer=False,
                          include_self=include_self)


def differential_similarity(wspd, normalise="none"):
    """
    Description:
        Eq. 3 -- similarity between wind-speed *variation* patterns.
    Args:
        wspd: (134, T) wind speed per turbine over time. NaNs are treated as
            no-change (diff contribution zero) rather than dropped, so a gappy
            turbine is pulled toward neutral instead of toward nonsense.
        normalise:
            "none" -- the paper's formula verbatim, unnormalised dot product.
            "corr" -- divide by the product of diff norms, i.e. cosine
                similarity of the difference series. ABLATION ONLY; this is
                not what the paper specifies.
    Returns:
        (134, 134) similarity matrix.
    """
    x = np.asarray(wspd, dtype=np.float64)
    assert x.shape[0] == CAPACITY, "expected (134, T), got {}".format(x.shape)

    d = np.diff(x, axis=1)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    sim = d @ d.T
    if normalise == "corr":
        norm = np.linalg.norm(d, axis=1)
        sim = sim / np.maximum(np.outer(norm, norm), 1e-12)
    elif normalise != "none":
        raise ValueError("normalise must be 'none' or 'corr'")
    return sim


def build_semantic_graph(wspd, k=TOP_K, normalise="none", include_self=False):
    """
    Description:
        G_s, Eq. 3 followed by top-k selection.
    Args:
        wspd: (134, T) wind speed. Pass the training series for the static
            graph, or one window's input series for the per-window variant.
    Returns:
        (134, 134) float32 binary adjacency.
    """
    sim = differential_similarity(wspd, normalise=normalise)
    return _topk_binarise(sim, k, larger_is_closer=True,
                          include_self=include_self)


def wspd_from_train(path=None, capacity=CAPACITY):
    """
    Description:
        Pull the wind-speed panel out of the 245-day training CSV.
    Returns:
        (134, T) float array; NaNs left in place for the caller to handle.
    """
    df = pd.read_csv(path or TRAIN_FILE, usecols=["TurbID", "Wspd"])
    w = df["Wspd"].to_numpy(dtype=np.float64)
    assert w.size % capacity == 0, \
        "row count {} not divisible by {}".format(w.size, capacity)
    return w.reshape(capacity, -1)            # file is turbine-major


def wspd_from_window(path, capacity=CAPACITY):
    """Same, for a single final_phase_test infile (per-window G_s variant)."""
    df = pd.read_csv(path, usecols=["TurbID", "Wspd"])
    return df["Wspd"].to_numpy(dtype=np.float64).reshape(capacity, -1)


def _inspect(g_d, g_s, loc):
    """
    Description:
        Sanity checks worth running once. With no reference implementation to
        diff against, these are the evidence that the graphs are sane.
    """
    n = CAPACITY
    diff = loc[:, None, :] - loc[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))

    print("degree (out) : G_d {:.1f}  G_s {:.1f}   (should equal K)".format(
        g_d.sum(1).mean(), g_s.sum(1).mean()))
    print("symmetric    : G_d {}  G_s {}   (top-k need not be)".format(
        bool((g_d == g_d.T).all()), bool((g_s == g_s.T).all())))
    print("in-degree    : G_d max {:.0f}  G_s max {:.0f}   "
          "(a hub means one turbine is everyone's neighbour)".format(
              g_d.sum(0).max(), g_s.sum(0).max()))

    overlap = (g_d * g_s).sum() / g_d.sum()
    print("G_d n G_s    : {:.1%} of spatial edges are also semantic".format(
        overlap))
    print("             -> low overlap is the point; if this is near 100%,")
    print("                G_s adds nothing and the paper's premise fails")

    md = dist[g_d > 0].mean()
    ms = dist[g_s > 0].mean()
    print("mean edge distance: G_d {:.0f} m   G_s {:.0f} m".format(md, ms))
    print("             -> G_s should be much larger; it is meant to reach")
    print("                past the local neighbourhood")

    iso = int((g_s.sum(0) == 0).sum())
    print("nodes no one selects as a semantic neighbour: {}/{}".format(iso, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--normalise", choices=["none", "corr"], default="none",
                    help="'none' is the paper; 'corr' is an ablation")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    loc = load_locations()

    g_d = build_spatial_graph(loc, k=args.k)
    g_s = build_semantic_graph(wspd_from_train(), k=args.k,
                               normalise=args.normalise)

    suffix = "" if args.normalise == "none" else "_" + args.normalise
    for name, g in (("g_spatial_k{}".format(args.k), g_d),
                    ("g_semantic_k{}{}".format(args.k, suffix), g_s)):
        p = os.path.join(args.out_dir, name + ".npy")
        np.save(p, g)
        print("wrote {}  {}".format(p, g.shape))

    if args.inspect:
        print()
        _inspect(g_d, g_s, loc)


if __name__ == "__main__":
    main()
