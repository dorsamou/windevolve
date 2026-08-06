# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/analysis/plots.py

Description:
    Figures for the baseline report. Three stages, run in order:

      predict   run each baseline over chosen validation windows and cache its
                raw (134, 288) prediction to disk, so every figure below is
                drawn from ONE set of forecasts rather than re-running models
      figures   prediction plots (per turbine, per window, and farm-aggregate),
                the error-vs-horizon curve, and the masking picture
      sweeps    parameter curves read straight from runs/sweeps/*.tsv

    Separating predict from figures matters for the GRU: its dropout is live at
    inference, so calling forecast() twice gives two different answers. Caching
    once means the GRU curve in the per-turbine plot and the GRU curve in the
    horizon plot are the same forecast, not two draws.

    THE MASK IS DRAWN, NOT HIDDEN. The official metric excludes ground-truth
    steps that are missing or invalid -- NaN, negative power, a turbine stopped
    while the wind blows, feathered blades, an out-of-range sensor. Those steps
    are shaded in every prediction plot, because a model can look wrong on a
    stretch that was never scored, and a reader who does not know that will draw
    the wrong conclusion from the picture.

    Configuration selection reuses analysis/sweep.py's grids: a model is plotted
    at its REFERENCE configuration by default, or at its best validation
    configuration with --best, which reads the sweep TSV. Nothing is duplicated
    between the two files.

Citation:
    Masking rule reproduced from the official Baidu KDD Cup 2022 SDWPF grader
    (PaddlePaddle/PaddleSpatial, .../kddcup22-sdwpf-evaluation), turbine_scores.
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # from the package root (windevolve_sdwpf/)
    python analysis/plots.py predict --models naive,climatology,arma,gbdt
    python analysis/plots.py predict --models gru --best
    python analysis/plots.py figures --windows 1,12,24 --turbines 0,42,97
    python analysis/plots.py sweeps
    # everything, default selections
    python analysis/plots.py all
"""
import os                                  # paths
import sys                                 # sys.path, exit codes
import glob                                # window discovery
import json                                # config round-trip
import argparse                            # CLI

import numpy as np                         # arrays
import pandas as pd                        # TSV / CSV reading
import matplotlib
matplotlib.use("Agg")                      # headless: write files, open no window
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sweep as sw                         # grids, module loading, overrides
from test_data import TestData             # official per-window loader
from paths import VAL_DIR, LOCATION_FILE, RUNS_DIR

PLOT_DIR = os.path.join(RUNS_DIR, "plots")
PRED_DIR = os.path.join(PLOT_DIR, "preds")
CAPACITY, OUTPUT_LEN, DAY_LEN = 134, 288, 144

# One colour per baseline, fixed across every figure in the report so a reader
# learns the legend once.
COLOURS = {"naive": "#d62728", "climatology": "#ff7f0e", "arma": "#2ca02c",
           "gbdt": "#1f77b4", "gru": "#9467bd", "actual": "#111111"}
NICE = {"naive": "Seasonal naive", "climatology": "Climatology",
        "arma": "ARMA", "gbdt": "XGBoost", "gru": "GRU"}


# ---------------------------------------------------------------------------
# Window discovery and ground truth
# ---------------------------------------------------------------------------
def list_windows(val_dir=None):
    """
    Description:
        Pair the validation infiles and outfiles, in the order the evaluator
        scores them, so a window index here means the same window there.
    Args:
        val_dir: validation window directory (default: paths.VAL_DIR)
    Returns:
        A list of (index, infile_path, outfile_path), 1-based.
    """
    vd = val_dir or VAL_DIR
    ins = sorted(glob.glob(os.path.join(vd, "infile", "*in.csv")))
    outs = sorted(glob.glob(os.path.join(vd, "outfile", "*out.csv")))
    if not ins:
        raise SystemExit("no validation windows under {}. Build the split "
                         "first: python evaluation/val_windows.py".format(vd))
    return [(i, a, b) for i, (a, b) in enumerate(zip(ins, outs), start=1)]


def ground_truth(outfile):
    """
    Description:
        Load one window's ground truth and the official validity mask.
    Args:
        outfile: path to a window's NNNNout.csv
    Returns:
        (truth, valid) -- two (134, 288) arrays; truth is Patv in kW, valid is
        boolean and True where the official grader would score the step.
    """
    td = TestData(path_to_data=outfile, start_col=3, farm_capacity=CAPACITY)
    _, raw_turbines = td.get_all_turbines()
    truth = np.full((CAPACITY, OUTPUT_LEN), np.nan)
    valid = np.zeros((CAPACITY, OUTPUT_LEN), dtype=bool)
    for t in range(CAPACITY):
        raw = raw_turbines[t].iloc[:OUTPUT_LEN].reset_index(drop=True)
        truth[t] = raw["Patv"].to_numpy(dtype=float)
        nan_cond = pd.isna(raw).any(axis=1).to_numpy()
        bad = ((raw["Patv"] < 0)
               | ((raw["Patv"] == 0) & (raw["Wspd"] > 2.5))
               | (raw["Pab1"] > 89) | (raw["Pab2"] > 89) | (raw["Pab3"] > 89)
               | (raw["Wdir"] < -180) | (raw["Wdir"] > 180)
               | (raw["Ndir"] < -720) | (raw["Ndir"] > 720)).to_numpy()
        valid[t] = ~nan_cond & ~bad
    return truth, valid


# ---------------------------------------------------------------------------
# Configuration selection -- reuses sweep.py's grids
# ---------------------------------------------------------------------------
def pick_point(model, best=False, label=None):
    """
    Description:
        Choose which configuration of a model to plot. Default is the grid's
        reference point. --best reads the model's sweep TSV and takes the
        lowest valid score; that point is selected ON VALIDATION, which is the
        correct thing to do here and must be reported as such.
    Args:
        model: key into sweep.MODELS
        best: pick the best validation score from the sweep TSV
        label: explicit grid label, overriding both
    Returns:
        A grid point dict.
    """
    spec = sw.MODELS[model]
    points = spec["grid"]()
    want = label
    if want is None and best:
        tsv = os.path.join(sw.SWEEP_DIR, "sweep_{}.tsv".format(model))
        if os.path.exists(tsv):
            df = pd.read_csv(tsv, sep="\t")
            df = df[df["valid"] == 1]
            if len(df):
                want = df.loc[df["score"].idxmin(), "label"]
                print("  {}: best on validation = {!r} ({:.3f} MW)".format(
                    model, want, df["score"].min()))
        else:
            print("  {}: no sweep TSV yet, using the reference point".format(model))
    if want is None:
        want = spec["reference"]
    for p in points:
        if p["label"] == want:
            return p
    raise SystemExit("{}: no grid point labelled {!r}".format(model, want))


def build_module(model, point):
    """
    Description:
        Load a baseline and apply one grid point's parameters, exactly as the
        sweep harness does, so a plotted forecast matches its scored row.
    Args:
        model: key into sweep.MODELS
        point: a grid point dict
    Returns:
        The configured module.
    """
    spec = sw.MODELS[model]
    mod = sw.load_baseline(spec["path"])
    overrides = dict(point["overrides"])
    if point["fit"] is not None and spec["prepare"] is not None:
        cache_dir = os.path.join(sw.SWEEP_DIR, "artifacts", model)
        os.makedirs(cache_dir, exist_ok=True)
        fit_over, _ = spec["prepare"](point["fit"], mod, cache_dir)
        overrides.update(fit_over)
    if model == "gru":
        sw.post_load_gru(mod, overrides)
    sw.apply_overrides(mod, overrides)
    sw.wrap_scale(mod, point["scale"])
    return mod


# ---------------------------------------------------------------------------
# Stage 1: predict
# ---------------------------------------------------------------------------
def stage_predict(args):
    """
    Description:
        Run each requested model over each requested window and cache the
        (134, 288) prediction as .npy, alongside a JSON note recording which
        configuration produced it.
    Args:
        args: parsed CLI namespace
    Returns:
        0
    """
    os.makedirs(PRED_DIR, exist_ok=True)
    windows = list_windows()
    wanted = ([int(x) for x in args.windows.split(",")] if args.windows
              else [w[0] for w in windows])
    by_idx = {i: (a, b) for i, a, b in windows}

    for model in args.models.split(","):
        model = model.strip()
        if model not in sw.MODELS:
            raise SystemExit("unknown model {!r}".format(model))
        point = pick_point(model, best=args.best, label=args.label)
        print("{}: configuration {!r}".format(model, point["label"]))
        mod = build_module(model, point)
        for w in wanted:
            if w not in by_idx:
                raise SystemExit("window {} does not exist (have 1..{})".format(
                    w, len(windows)))
            infile, _ = by_idx[w]
            envs = {"path_to_test_x": infile, "path_to_train": sw.VAL_TRAIN_FILE,
                    "path_to_locations": LOCATION_FILE, "start_col": 3,
                    "out_var": 1, "output_len": OUTPUT_LEN, "capacity": CAPACITY}
            pred = np.asarray(mod.forecast(envs), dtype=float).reshape(
                CAPACITY, OUTPUT_LEN)
            np.save(os.path.join(PRED_DIR, "{}__w{:02d}.npy".format(model, w)), pred)
            print("  window {:>3}  ok".format(w))
        with open(os.path.join(PRED_DIR, "{}.json".format(model)), "w") as fh:
            json.dump({"label": point["label"], "scale": point["scale"],
                       "overrides": {k: str(v) for k, v in point["overrides"].items()}},
                      fh, indent=2)
    print("\npredictions cached in {}".format(PRED_DIR))
    return 0


def load_pred(model, w):
    """Load a cached prediction, or None if that model/window was not run."""
    p = os.path.join(PRED_DIR, "{}__w{:02d}.npy".format(model, w))
    return np.load(p) if os.path.exists(p) else None


def available_models(windows):
    """Which models have cached predictions for every requested window."""
    out = []
    for m in sw.MODELS:
        if all(load_pred(m, w) is not None for w in windows):
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Stage 2: figures
# ---------------------------------------------------------------------------
def _hours(n=OUTPUT_LEN):
    """Horizon axis in hours, so the reader sees 0-48 rather than 0-288."""
    return np.arange(n) / 6.0


def _shade_mask(ax, valid_row):
    """Shade the horizon stretches the official metric does not score."""
    bad = ~valid_row
    if not bad.any():
        return
    h = _hours(len(valid_row))
    start = None
    for i, b in enumerate(bad):
        if b and start is None:
            start = i
        elif not b and start is not None:
            ax.axvspan(h[start], h[i], color="0.85", zorder=0, lw=0)
            start = None
    if start is not None:
        ax.axvspan(h[start], h[-1], color="0.85", zorder=0, lw=0)


def fig_single(models, w, turbine, outpath):
    """
    Description:
        The core prediction plot: one turbine, one window, every model's
        forecast against the truth, with unscored steps shaded.
    Args:
        models: list of model keys with cached predictions
        w: window index
        turbine: turbine index (0-based)
        outpath: PNG path
    Returns:
        None
    """
    _, _, outfile = [x for x in list_windows() if x[0] == w][0]
    truth, valid = ground_truth(outfile)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    _shade_mask(ax, valid[turbine])
    ax.plot(_hours(), truth[turbine], color=COLOURS["actual"], lw=2.0,
            label="Actual", zorder=5)
    for m in models:
        p = load_pred(m, w)
        ax.plot(_hours(), p[turbine], color=COLOURS[m], lw=1.3, alpha=0.9,
                label=NICE[m])
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Active power, Patv (kW)")
    ax.set_title("Window {} | turbine {} | shaded = excluded by the official mask"
                 .format(w, turbine))
    ax.set_xlim(0, 48)
    ax.legend(ncol=3, fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_grid(models, windows, turbines, outpath):
    """
    Description:
        The same plot tiled: turbines down the rows, windows across the columns.
        This is the honest way to show more than one case -- a single panel can
        always be cherry-picked, a grid cannot.
    Args:
        models: list of model keys
        windows: list of window indices
        turbines: list of turbine indices
        outpath: PNG path
    Returns:
        None
    """
    gts = {w: ground_truth(dict((i, c) for i, _, c in list_windows())[w])
           for w in windows}
    nrow, ncol = len(turbines), len(windows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.5 * nrow),
                             squeeze=False, sharex=True)
    for r, t in enumerate(turbines):
        for c, w in enumerate(windows):
            ax = axes[r][c]
            truth, valid = gts[w]
            _shade_mask(ax, valid[t])
            ax.plot(_hours(), truth[t], color=COLOURS["actual"], lw=1.6, zorder=5)
            for m in models:
                ax.plot(_hours(), load_pred(m, w)[t], color=COLOURS[m],
                        lw=1.0, alpha=0.85)
            ax.set_xlim(0, 48)
            ax.grid(alpha=0.2, lw=0.4)
            if r == 0:
                ax.set_title("window {}".format(w), fontsize=10)
            if c == 0:
                ax.set_ylabel("turbine {}\nPatv (kW)".format(t), fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel("horizon (h)", fontsize=9)
    handles = [plt.Line2D([], [], color=COLOURS["actual"], lw=1.6, label="Actual")]
    handles += [plt.Line2D([], [], color=COLOURS[m], lw=1.2, label=NICE[m])
                for m in models]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_farm(models, windows, outpath):
    """
    Description:
        Farm-aggregate power: every turbine summed, per window. This is the
        picture that matches what the project is actually forecasting -- the
        official score sums errors across all 134 turbines, so farm total is
        the quantity a reader should judge, and single-turbine panels are
        illustration rather than evidence.
    Args:
        models: list of model keys
        windows: list of window indices
        outpath: PNG path
    Returns:
        None
    """
    by_idx = dict((i, c) for i, _, c in list_windows())
    n = len(windows)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 3.4), squeeze=False, sharey=True)
    for c, w in enumerate(windows):
        ax = axes[0][c]
        truth, valid = ground_truth(by_idx[w])
        # Sum only over turbines the mask keeps at each step, then rescale to a
        # full-farm equivalent, so a window with many masked turbines is not
        # drawn as a power dip that never happened.
        keep = valid.astype(float)
        n_keep = np.maximum(keep.sum(axis=0), 1.0)
        tot = np.nansum(np.where(valid, truth, 0.0), axis=0) / n_keep * CAPACITY
        ax.plot(_hours(), tot / 1000.0, color=COLOURS["actual"], lw=2.0,
                label="Actual", zorder=5)
        for m in models:
            p = load_pred(m, w)
            pt = np.sum(np.where(valid, p, 0.0), axis=0) / n_keep * CAPACITY
            ax.plot(_hours(), pt / 1000.0, color=COLOURS[m], lw=1.3, label=NICE[m])
        ax.set_title("window {}".format(w), fontsize=10)
        ax.set_xlabel("horizon (h)")
        ax.set_xlim(0, 48)
        ax.grid(alpha=0.25, lw=0.5)
        if c == 0:
            ax.set_ylabel("Farm total power (MW)")
            ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_horizon(models, windows, outpath):
    """
    Description:
        Mean absolute error against horizon step, averaged over every turbine
        and every requested window, masked steps excluded. This is the figure
        that explains the score table: two models with the same total score can
        fail in completely different places, and a sweep that improves the
        first six hours while wrecking the last twelve looks like no change at
        all in the aggregate.
    Args:
        models: list of model keys
        windows: list of window indices
        outpath: PNG path
    Returns:
        None
    """
    by_idx = dict((i, c) for i, _, c in list_windows())
    fig, ax = plt.subplots(figsize=(9, 4.0))
    for m in models:
        num = np.zeros(OUTPUT_LEN)
        den = np.zeros(OUTPUT_LEN)
        for w in windows:
            truth, valid = ground_truth(by_idx[w])
            err = np.abs(load_pred(m, w) - truth) / 1000.0     # kW -> MW
            num += np.nansum(np.where(valid, err, 0.0), axis=0)
            den += valid.sum(axis=0)
        ax.plot(_hours(), num / np.maximum(den, 1), color=COLOURS[m],
                lw=1.6, label=NICE[m])
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Mean |error| per turbine (MW)")
    ax.set_title("Where each baseline loses: error against horizon "
                 "({} windows, masked steps excluded)".format(len(windows)))
    ax.set_xlim(0, 48)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def stage_figures(args):
    """Draw every prediction figure from the cached forecasts."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    windows = [int(x) for x in args.windows.split(",")]
    turbines = [int(x) for x in args.turbines.split(",")]
    models = available_models(windows)
    if not models:
        raise SystemExit("no cached predictions for windows {}. Run: "
                         "python analysis/plots.py predict".format(windows))
    print("models with cached predictions: {}".format(", ".join(models)))

    fig_single(models, windows[0], turbines[0],
               os.path.join(PLOT_DIR, "pred_single.png"))
    fig_grid(models, windows, turbines, os.path.join(PLOT_DIR, "pred_grid.png"))
    fig_farm(models, windows, os.path.join(PLOT_DIR, "pred_farm.png"))
    fig_horizon(models, windows, os.path.join(PLOT_DIR, "error_vs_horizon.png"))
    print("wrote pred_single.png, pred_grid.png, pred_farm.png, "
          "error_vs_horizon.png -> {}".format(PLOT_DIR))
    return 0


# ---------------------------------------------------------------------------
# Stage 3: sweep curves
# ---------------------------------------------------------------------------
def stage_sweeps(args):
    """
    Description:
        One figure per model: score against each swept parameter, with the
        measured noise floor drawn as a band where one exists. A sweep curve
        without that band invites the reader to believe every wiggle.
    Args:
        args: parsed CLI namespace
    Returns:
        0
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    tsvs = sorted(glob.glob(os.path.join(sw.SWEEP_DIR, "sweep_*.tsv")))
    if not tsvs:
        raise SystemExit("no sweep results under {}".format(sw.SWEEP_DIR))

    for tsv in tsvs:
        model = os.path.basename(tsv)[len("sweep_"):-len(".tsv")]
        df = pd.read_csv(tsv, sep="\t")
        ok = df[df["valid"] == 1].copy()
        if not len(ok):
            print("{}: no valid rows, skipped".format(model))
            continue

        # Noise floor: spread of the repeated identical configuration, if the
        # grid has a seed axis. Everything else is judged against it.
        floor = None
        seeds = ok[ok["axis"] == "seed"]
        if len(seeds) >= 2:
            floor = (float(seeds["score"].mean()), float(seeds["score"].std(ddof=1)))
            print("{}: noise floor {:.3f} +/- {:.3f} MW over {} repeats".format(
                model, floor[0], floor[1], len(seeds)))

        axes_names = [a for a in ok["axis"].unique() if a != "seed"]
        if not axes_names:
            axes_names = list(ok["axis"].unique())
        n = len(axes_names)
        fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
        for i, axis in enumerate(axes_names):
            ax = axs[0][i]
            sub = ok[ok["axis"] == axis]
            xs = _axis_values(sub)
            if xs is None:                       # non-numeric axis -> bar chart
                ax.bar(range(len(sub)), sub["score"], color=COLOURS.get(model, "0.4"))
                ax.set_xticks(range(len(sub)))
                ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=7)
            else:
                order = np.argsort(xs)
                ax.plot(np.asarray(xs)[order], sub["score"].to_numpy()[order],
                        "o-", color=COLOURS.get(model, "0.4"), lw=1.5, ms=4)
                ax.set_xlabel(axis)
            if floor is not None:
                ax.axhspan(floor[0] - floor[1], floor[0] + floor[1],
                           color="0.85", zorder=0, lw=0,
                           label="noise floor (+/-1 sd)")
                if i == 0:
                    ax.legend(fontsize=7)
            ax.set_ylabel("Score, 0.5(RMSE+MAE)  MW" if i == 0 else "")
            ax.set_title(axis, fontsize=10)
            ax.grid(alpha=0.25, lw=0.5)
        fig.suptitle("{} -- validation sweep (lower is better)".format(
            NICE.get(model, model)), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(PLOT_DIR, "sweep_{}.png".format(model))
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote {}".format(out))

    _sweep_tables(tsvs)
    return 0


def stage_summary(args):
    """
    Description:
        Build section 5.4 of the report -- the cross-model summary -- directly
        from the sweep TSVs, so the headline table is never transcribed by hand
        and can never disagree with the results it claims to summarise.

        Two score columns, deliberately:
          reference   the default configuration, unselected. The honest
                      baseline number, and the score the seed program starts
                      from.
          swept-best  the lowest validation score in the grid. Selected ON
                      VALIDATION and therefore optimistically biased -- but the
                      evolved champion is selected on validation too, so this is
                      the like-for-like comparator.

        The scale axis is reported separately from the rest of the grid, because
        shrinkage is a metric-calibration control rather than a hyperparameter,
        and folding it into "swept-best" would hide exactly the thing it exists
        to expose.
    Args:
        args: parsed CLI namespace
    Returns:
        0
    """
    tsvs = sorted(glob.glob(os.path.join(sw.SWEEP_DIR, "sweep_*.tsv")))
    if not tsvs:
        raise SystemExit("no sweep results under {}".format(sw.SWEEP_DIR))

    rows = []
    for tsv in tsvs:
        model = os.path.basename(tsv)[len("sweep_"):-len(".tsv")]
        df = pd.read_csv(tsv, sep="\t")
        ok = df[df["valid"] == 1].copy()
        if not len(ok):
            continue
        spec = sw.MODELS.get(model, {})
        ref_label = spec.get("reference", "")

        ref = ok[ok["label"] == ref_label]
        ref_score = float(ref["score"].iloc[0]) if len(ref) else float("nan")

        # Best excluding the scale axis: pure hyperparameter headroom.
        tuned = ok[ok["axis"] != "scale"]
        if len(tuned):
            b = tuned.loc[tuned["score"].idxmin()]
            best_label, best_score = str(b["label"]), float(b["score"])
        else:
            best_label, best_score = "--", float("nan")

        # Best on the scale axis: what uniform shrinkage alone is worth.
        sc = ok[ok["axis"] == "scale"]
        if len(sc):
            s = sc.loc[sc["score"].idxmin()]
            scale_label, scale_score = str(s["label"]), float(s["score"])
        else:
            scale_label, scale_score = "--", float("nan")

        seeds = ok[ok["axis"] == "seed"]
        floor = ("{:.3f}".format(float(seeds["score"].std(ddof=1)))
                 if len(seeds) >= 2 else "deterministic")

        rows.append({
            "model": NICE.get(model, model),
            "ref_label": ref_label, "ref": ref_score,
            "best_label": best_label, "best": best_score,
            "scale_label": scale_label, "scale": scale_score,
            "floor": floor,
            "n_points": len(ok), "n_invalid": int((df["valid"] == 0).sum()),
        })

    if not rows:
        raise SystemExit("no valid rows in any sweep TSV")
    rows.sort(key=lambda r: r["ref"] if r["ref"] == r["ref"] else 1e9)

    def _f(x):
        return "{:.3f}".format(x) if x == x else "--"

    out = os.path.join(PLOT_DIR, "summary_table.md")
    os.makedirs(PLOT_DIR, exist_ok=True)
    with open(out, "w") as fh:
        fh.write("### 5.4 Cross-model summary\n\n")
        fh.write("Scores are 0.5(RMSE+MAE) in MW on the validation split, "
                 "lower is better.\n\n")
        fh.write("| Model | Reference config | Reference | Best tuned config | "
                 "Best tuned | Tuning headroom | Best shrinkage config | "
                 "Shrunk score | Shrinkage gain | Noise floor (1 sd) |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            head = (r["ref"] - r["best"]
                    if r["ref"] == r["ref"] and r["best"] == r["best"]
                    else float("nan"))
            gain = (r["ref"] - r["scale"]
                    if r["ref"] == r["ref"] and r["scale"] == r["scale"]
                    else float("nan"))
            fh.write("| {} | `{}` | {} | `{}` | {} | {} | `{}` | {} | {} | {} |\n"
                     .format(r["model"], r["ref_label"], _f(r["ref"]),
                             r["best_label"], _f(r["best"]), _f(head),
                             r["scale_label"], _f(r["scale"]), _f(gain),
                             r["floor"]))
        fh.write("\n**Tuning headroom** = reference minus best tuned. This is the "
                 "bar the evolutionary search has to clear before *discovery* is "
                 "the right word for what it did.\n\n")
        fh.write("**Shrinkage gain** = reference minus the best point on the "
                 "scale axis. Expect this to be large for the poor forecasters "
                 "and near zero for the good ones: a well-calibrated forecast "
                 "has nothing to shrink toward. If that pattern holds it is the "
                 "cleanest evidence available that shrinkage is metric "
                 "calibration, not forecasting skill.\n\n")
        fh.write("Grid points scored: {}. Rejected by the official guards: {}.\n"
                 .format(sum(r["n_points"] for r in rows),
                         sum(r["n_invalid"] for r in rows)))

    print(open(out).read())
    print("wrote {}".format(out))
    return 0


# ---------------------------------------------------------------------------
# Stage 4: regime analysis -- where in the horizon, and in what power state,
# does each model actually win?
# ---------------------------------------------------------------------------
# Bins on the turbine's power at the forecast origin. Physical rather than
# quantile bins, because the question is about named regimes -- "parked",
# "near rated" -- and quantile edges would move between runs and make two
# reports incomparable. SDWPF turbines are ~1500 kW rated.
POWER_BINS = [(-1e9, 30.0, "parked (<30 kW)"),
              (30.0, 300.0, "low (30-300)"),
              (300.0, 800.0, "mid (300-800)"),
              (800.0, 1200.0, "high (800-1200)"),
              (1200.0, 1e9, "near rated (>1200)")]


def origin_power(infile, capacity=CAPACITY):
    """
    Description:
        Each turbine's power state at the forecast origin, taken as the mean of
        the final hour of the input window rather than the single last reading.
        One 10-minute sample is noisy and occasionally a dropout; an hourly mean
        is the same quantity with less measurement noise, which matters because
        it is being used to define regime boundaries.
    Args:
        infile: path to a window's NNNNin.csv
        capacity: number of turbines
    Returns:
        A (capacity,) float array of recent mean Patv in kW.
    """
    df = pd.read_csv(infile, usecols=["Patv"])
    total = len(df) // capacity
    patv = np.nan_to_num(df["Patv"].to_numpy(dtype=float), nan=0.0)
    return patv.reshape(capacity, total)[:, -6:].mean(axis=1)


def _bin_index(values):
    """Map origin powers onto POWER_BINS indices."""
    idx = np.full(len(values), -1, dtype=int)
    for b, (lo, hi, _) in enumerate(POWER_BINS):
        idx[(values >= lo) & (values < hi)] = b
    return idx


def _regime_tensors(models, windows):
    """
    Description:
        Accumulate masked absolute error, and the mean predicted and actual
        trajectories, split by origin-power regime and horizon step.
    Args:
        models: list of model keys with cached predictions
        windows: list of window indices
    Returns:
        (err, pred, truth, count) where err/pred are
        {model: (n_bins, 288)} in MW, truth is (n_bins, 288), and count is
        (n_bins,) turbine-windows per regime.
    """
    by_idx = {i: (a, b) for i, a, b in list_windows()}
    nb = len(POWER_BINS)
    err = {m: np.zeros((nb, OUTPUT_LEN)) for m in models}
    pred = {m: np.zeros((nb, OUTPUT_LEN)) for m in models}
    den = np.zeros((nb, OUTPUT_LEN))
    truth_sum = np.zeros((nb, OUTPUT_LEN))
    count = np.zeros(nb)

    for w in windows:
        infile, outfile = by_idx[w]
        t, valid = ground_truth(outfile)
        bins = _bin_index(origin_power(infile))
        for b in range(nb):
            rows = np.where(bins == b)[0]
            if not len(rows):
                continue
            v = valid[rows]
            den[b] += v.sum(axis=0)
            truth_sum[b] += np.nansum(np.where(v, t[rows], 0.0), axis=0)
            count[b] += len(rows)
            for m in models:
                p = load_pred(m, w)[rows]
                err[m][b] += np.nansum(np.where(v, np.abs(p - t[rows]), 0.0), axis=0)
                pred[m][b] += np.nansum(np.where(v, p, 0.0), axis=0)

    d = np.maximum(den, 1.0)
    for m in models:
        err[m] = err[m] / d / 1000.0            # kW -> MW
        pred[m] = pred[m] / d
    return err, pred, truth_sum / d, count


def fig_regime_horizon(models, err, count, outpath, smooth=6):
    """
    Description:
        Error against horizon, one panel per origin-power regime. This is the
        figure that tests the claim that a neural net and a tree fail in
        different places: if the GRU wins in mid-power states and loses at the
        extremes, the curves will cross, and they will cross at different
        horizons in different panels.
    Args:
        models: list of model keys
        err: {model: (n_bins, 288)} mean |error| in MW
        count: (n_bins,) turbine-windows per regime
        outpath: PNG path
    Returns:
        None
    """
    nb = len(POWER_BINS)
    fig, axes = plt.subplots(1, nb, figsize=(3.4 * nb, 3.6), squeeze=False, sharex=True)
    for b in range(nb):
        ax = axes[0][b]
        for m in models:
            ax.plot(_hours(), _smooth(err[m][b], smooth), color=COLOURS[m],
                    lw=1.5, label=NICE[m])
        ax.set_title("{}\nn={:,}".format(POWER_BINS[b][2], int(count[b])), fontsize=9)
        ax.set_xlabel("horizon (h)")
        ax.set_xlim(0, 48)
        ax.grid(alpha=0.25, lw=0.5)
        if b == 0:
            ax.set_ylabel("Mean |error| per turbine (MW)")
            ax.legend(fontsize=7)
    fig.suptitle("Where each model fails: error by horizon and power state at the "
                 "forecast origin", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_regime_winner(models, err, outpath):
    """
    Description:
        Which model has the lowest error in each (regime, horizon) cell, and by
        how much. The colour is the winner; the annotation is its margin over
        the runner-up, in MW. A margin near zero means the cell is a tie and the
        winner should not be read as a result.
    Args:
        models: list of model keys
        err: {model: (n_bins, 288)} mean |error| in MW
        outpath: PNG path
    Returns:
        None
    """
    buckets = [(0, 36, "0-6h"), (36, 72, "6-12h"), (72, 144, "12-24h"),
               (144, 216, "24-36h"), (216, 288, "36-48h")]
    nb, nh = len(POWER_BINS), len(buckets)
    stack = np.stack([np.array([err[m][:, lo:hi].mean(axis=1)
                                for lo, hi, _ in buckets]).T for m in models])
    win = np.argmin(stack, axis=0)                       # (n_bins, n_buckets)
    srt = np.sort(stack, axis=0)
    margin = srt[1] - srt[0] if len(models) > 1 else np.zeros_like(srt[0])

    fig, ax = plt.subplots(figsize=(1.5 * nh + 3, 0.75 * nb + 2.2))
    cmap = matplotlib.colors.ListedColormap([COLOURS[m] for m in models])
    ax.imshow(win, cmap=cmap, vmin=0, vmax=max(len(models) - 1, 1), aspect="auto")
    for b in range(nb):
        for h in range(nh):
            ax.text(h, b, "{}\n+{:.3f}".format(NICE[models[win[b, h]]].split()[0],
                                                margin[b, h]),
                    ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(nh))
    ax.set_xticklabels([lab for _, _, lab in buckets])
    ax.set_yticks(range(nb))
    ax.set_yticklabels([lab for _, _, lab in POWER_BINS], fontsize=8)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Power at forecast origin")
    ax.set_title("Best model per regime (margin over runner-up, MW)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_regime_decay(models, pred, truth, outpath, smooth=6):
    """
    Description:
        The mean forecast TRAJECTORY from each starting regime, with the actual
        mean trajectory overlaid. This tests the steepness claim directly: a
        model whose short-horizon curve is "too steep" from a high or low
        starting state will revert toward the middle faster than the truth
        does, and that shows up here as a curve that falls (or rises) away from
        the black line in the first few hours.

        This is a different question from the error plot. A model can have a
        good average error while systematically mis-shaping its trajectory,
        because over- and under-shoots cancel in the mean absolute error but not
        in the mean prediction.
    Args:
        models: list of model keys
        pred: {model: (n_bins, 288)} mean predicted power in kW
        truth: (n_bins, 288) mean actual power in kW
        outpath: PNG path
    Returns:
        None
    """
    nb = len(POWER_BINS)
    fig, axes = plt.subplots(1, nb, figsize=(3.4 * nb, 3.6), squeeze=False, sharex=True)
    for b in range(nb):
        ax = axes[0][b]
        ax.plot(_hours(), _smooth(truth[b], smooth), color=COLOURS["actual"],
                lw=2.0, label="Actual", zorder=5)
        for m in models:
            ax.plot(_hours(), _smooth(pred[m][b], smooth), color=COLOURS[m],
                    lw=1.4, label=NICE[m])
        ax.set_title(POWER_BINS[b][2], fontsize=9)
        ax.set_xlabel("horizon (h)")
        ax.set_xlim(0, 48)
        ax.grid(alpha=0.25, lw=0.5)
        if b == 0:
            ax.set_ylabel("Mean power (kW)")
            ax.legend(fontsize=7)
    fig.suptitle("Mean forecast trajectory by starting regime -- is the "
                 "short-horizon curve too steep?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _regime_tables(models, err, pred, truth, count):
    """Write the regime numbers as markdown, ready to paste into the report."""
    out = os.path.join(PLOT_DIR, "regime_tables.md")
    buckets = [(0, 36, "0-6h"), (36, 72, "6-12h"), (72, 144, "12-24h"),
               (144, 216, "24-36h"), (216, 288, "36-48h")]
    with open(out, "w") as fh:
        fh.write("### Mean |error| (MW) by power regime and horizon\n\n")
        fh.write("| regime | n | model | " + " | ".join(l for _, _, l in buckets) + " |\n")
        fh.write("|---|---|---|" + "---|" * len(buckets) + "\n")
        for b, (_, _, lab) in enumerate(POWER_BINS):
            for m in models:
                cells = ["{:.4f}".format(err[m][b, lo:hi].mean()) for lo, hi, _ in buckets]
                fh.write("| {} | {:,} | {} | {} |\n".format(
                    lab, int(count[b]), NICE[m], " | ".join(cells)))

        fh.write("\n### Crossover: first horizon where each model overtakes the "
                 "best-at-h=1 model\n\n")
        fh.write("| regime | leader at h=1 | overtaken by | at horizon |\n")
        fh.write("|---|---|---|---|\n")
        for b, (_, _, lab) in enumerate(POWER_BINS):
            first = min(models, key=lambda m: err[m][b, :6].mean())
            row = "| {} | {} | ".format(lab, NICE[first])
            beat = None
            for h in range(6, OUTPUT_LEN):
                better = [m for m in models if err[m][b, h] < err[first][b, h]]
                if better:
                    beat = (min(better, key=lambda m: err[m][b, h]), h / 6.0)
                    break
            fh.write(row + ("{} | {:.1f} h |\n".format(NICE[beat[0]], beat[1])
                            if beat else "never | -- |\n"))

        fh.write("\n### Trajectory bias at short horizon (mean predicted minus "
                 "mean actual, kW, averaged over 0-6h)\n\n")
        fh.write("A negative value means the model decays away from its starting "
                 "state faster than reality does; positive means it holds on too "
                 "long.\n\n")
        fh.write("| regime | " + " | ".join(NICE[m] for m in models) + " |\n")
        fh.write("|---|" + "---|" * len(models) + "\n")
        for b, (_, _, lab) in enumerate(POWER_BINS):
            cells = ["{:+.1f}".format((pred[m][b, :36] - truth[b, :36]).mean())
                     for m in models]
            fh.write("| {} | {} |\n".format(lab, " | ".join(cells)))
    print("wrote {}".format(out))


def _smooth(y, steps):
    """
    Description:
        Light moving average along the horizon axis. Conditional means are taken
        over far fewer samples than the aggregate ones -- a regime holding a few
        dozen turbine-windows is visibly noisy -- and the question being asked
        is about the SHAPE of the curve, which survives smoothing while the
        sampling jitter does not. Edges use a shrinking window rather than
        padding, so the first and last points are not dragged toward zero.
    Args:
        y: 1-D array over horizon steps
        steps: window width in 10-minute steps (0 or 1 disables)
    Returns:
        The smoothed array.
    """
    if not steps or steps < 2:
        return y
    k = int(steps)
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def stage_regimes(args):
    """
    Description:
        Run the regime analysis: split every turbine-window by the power at its
        forecast origin, then compare models within each regime rather than in
        aggregate. A single headline score hides the possibility that two models
        with near-identical totals are winning in completely different places --
        which is exactly what a portfolio or per-regime ensemble would exploit.
    Args:
        args: parsed CLI namespace
    Returns:
        0
    """
    os.makedirs(PLOT_DIR, exist_ok=True)
    windows = [int(x) for x in args.windows.split(",")]
    models = available_models(windows)
    if not models:
        raise SystemExit("no cached predictions for windows {}. Run: "
                         "python analysis/plots.py predict".format(windows))
    print("models: {}".format(", ".join(models)))
    err, pred, truth, count = _regime_tensors(models, windows)
    for b, (_, _, lab) in enumerate(POWER_BINS):
        print("  {:<22} {:>7,} turbine-windows".format(lab, int(count[b])))

    fig_regime_horizon(models, err, count,
                       os.path.join(PLOT_DIR, "regime_error.png"), args.smooth)
    fig_regime_winner(models, err, os.path.join(PLOT_DIR, "regime_winner.png"))
    fig_regime_decay(models, pred, truth,
                     os.path.join(PLOT_DIR, "regime_decay.png"), args.smooth)
    _regime_tables(models, err, pred, truth, count)
    print("wrote regime_error.png, regime_winner.png, regime_decay.png -> {}".format(
        PLOT_DIR))
    return 0


def _axis_values(sub):
    """Extract a numeric x value from each label, or None if that fails."""
    xs = []
    for lab in sub["label"]:
        tok = lab.split("=")[-1].strip()
        try:
            xs.append(float(tok))
        except ValueError:
            return None
    return xs


def _sweep_tables(tsvs):
    """Write every sweep as a markdown table, ready to paste into the report."""
    out = os.path.join(PLOT_DIR, "sweep_tables.md")
    with open(out, "w") as fh:
        for tsv in tsvs:
            model = os.path.basename(tsv)[len("sweep_"):-len(".tsv")]
            df = pd.read_csv(tsv, sep="\t")
            fh.write("\n### {}\n\n".format(NICE.get(model, model)))
            fh.write("| axis | setting | score (MW) | RMSE | MAE | valid |\n")
            fh.write("|---|---|---|---|---|---|\n")
            for _, r in df.iterrows():
                fh.write("| {} | {} | {} | {} | {} | {} |\n".format(
                    r["axis"], r["label"],
                    "{:.3f}".format(r["score"]) if r["valid"] else "--",
                    "{:.3f}".format(r["rmse"]) if r["valid"] else "--",
                    "{:.3f}".format(r["mae"]) if r["valid"] else "--",
                    "yes" if r["valid"] else str(r["reason"])[:60]))
    print("wrote {}".format(out))


# ---------------------------------------------------------------------------
def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Baseline report figures.")
    ap.add_argument("stage",
                    choices=["predict", "figures", "sweeps", "summary",
                             "regimes", "all"])
    ap.add_argument("--models", default="naive,climatology,arma,gbdt,gru",
                    help="comma-separated model keys (predict stage)")
    ap.add_argument("--windows", default="1,12,24",
                    help="comma-separated window indices")
    ap.add_argument("--turbines", default="0,42,97",
                    help="comma-separated turbine indices (figures stage)")
    ap.add_argument("--best", action="store_true",
                    help="plot each model at its best VALIDATION configuration "
                         "from its sweep TSV instead of the reference point")
    ap.add_argument("--smooth", type=int, default=6,
                    help="moving-average width in 10-min steps for the regime "
                         "curves (6 = 1 hour; 0 disables). Tables are never smoothed")
    ap.add_argument("--label", default=None,
                    help="plot one explicit grid label (single model only)")
    args = ap.parse_args()

    if args.stage in ("predict", "all"):
        stage_predict(args)
    if args.stage in ("figures", "all"):
        stage_figures(args)
    if args.stage in ("sweeps", "all"):
        stage_sweeps(args)
    if args.stage in ("regimes", "all"):
        stage_regimes(args)
    if args.stage in ("summary", "all"):
        stage_summary(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
