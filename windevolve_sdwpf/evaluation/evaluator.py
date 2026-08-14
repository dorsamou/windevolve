# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/evaluator.py

Description:
    The OpenEvolve-facing evaluator for the evolve loop. It scores a candidate
    program on the VALIDATION split (a held-out slice of the 245-day training
    data, built by val_windows.py) -- NEVER on final_phase_test. This is the
    critical anti-leakage rule: the search signal must be held-out data, so that
    final_phase_test stays locked for a single end-of-project measurement via
    competition_score.py.

    Scoring is delegated to official_metrics.py -- the REAL Baidu KDD Cup 2022
    submission grader, imported verbatim -- exactly like competition_score.py.
    So the evolve loop and the final measurement use the SAME ruler; the only
    difference is which windows they point at (validation vs final_phase_test).
    Using the real grader also gives us its degeneracy guards for free, which
    matters here because evolution WILL try to produce all-zero or constant
    programs that game the metric. The grader raises MetricsError on those; we
    catch it and return a large-negative penalty so OpenEvolve discards them.

    OpenEvolve MAXIMIZES `combined_score`, so a VALID score is NEGATIVE
    (combined_score = -competition_score). Any failure -- bad import, missing
    forecast(), wrong shape, NaN, degenerate prediction, runtime error, or
    timeout -- collapses to PENALTY_COMBINED_SCORE and valid=False, and this
    function NEVER raises.

    Contract: the candidate exposes
        forecast(envs) -> np.ndarray of shape (134, 288, 1) in kW,
    reading the infile at envs["path_to_test_x"] itself.

Citation:
    Scoring via the official Baidu KDD Cup 2022 SDWPF submission grader
    (PaddlePaddle/PaddleSpatial, .../kddcup22-sdwpf-evaluation/paddlepaddle),
    imported verbatim as official_metrics.py.
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # build the split once:
    python val_windows.py --train-file /path/to/sdwpf_245days_v1.csv
    # then the evolve loop (config.yaml points OpenEvolve at this file):
    python -m openevolve.cli initial_program.py evaluator.py --config config.yaml
    # or score one program directly for a quick check:
    python evaluator.py initial_program.py
"""
import os                          # paths and env vars
import sys                         # program loading, argv
import glob                        # discover windows
import time                        # timing
import importlib.util             # load candidate by path

import numpy as np                 # arrays

# This file is loaded BY PATH (OpenEvolve uses importlib.util), which does not
# put its own directory on sys.path. Add this directory -- for the sibling
# scorer modules -- and the repo root -- for paths.py -- explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_data import TestData     # official per-window loader (faithful copy)
import official_metrics as om      # the REAL competition grader, verbatim
from paths import VAL_DIR, VAL_TRAIN_FILE, LOCATION_FILE

CAPACITY, OUTPUT_LEN, OUT_VAR, START_COL, DAY_LEN = 134, 288, 1, 3, 144

# Guard thresholds: official competition defaults.
MIN_NON_ZERO_RATIO = float(os.environ.get("SDWPF_MIN_NON_ZERO_RATIO", "0.9"))
MIN_DISTINCT_RATIO = float(os.environ.get("SDWPF_MIN_DISTINCT_RATIO", "0.1"))

# Returned on ANY failure; far below any real score (~ -40..-60) so a broken
# candidate can never outrank a working one.
PENALTY_COMBINED_SCORE = -1.0e6

# Farm rated capacity in MW: 134 turbines x 1500 kW. The competition score is a
# SUM of per-turbine errors in MW, so dividing by this gives error as a percent
# of installed capacity -- the normalisation the wind-forecasting literature
# uses (nMAE/nRMSE), and the one that stays comparable across periods because
# capacity is fixed while mean wind is not.
FARM_CAPACITY_MW = 134 * 1.5

# Soft budget warning: OpenEvolve kills candidates that exceed its per-eval
# timeout, and a killed eval collapses to fitness 0.0 -- which would OUTRANK
# valid negative scores. Warn if we get close.
TIMEOUT_WARN_SECONDS = float(os.environ.get("SDWPF_TIMEOUT_WARN", "250"))

# Hard in-evaluator wall-clock budget. MUST sit comfortably below
# evaluator.timeout in config.yaml: if OpenEvolve's own timeout fires first the
# candidate collapses to fitness 0.0, which OUTRANKS every valid (negative)
# score and poisons the run. We raise instead, and evaluate() converts that into
# PENALTY_COMBINED_SCORE like any other failure.
EVAL_BUDGET_SECONDS = float(os.environ.get("SDWPF_EVAL_BUDGET", "600"))


class EvalBudgetExceeded(RuntimeError):
    """Raised when a candidate exceeds the in-evaluator wall-clock budget."""


def _load_program(path):
    """Import a candidate program by path; must expose forecast(envs)."""
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prep_env(path_to_test_x=None, train_file=None, fold=None):
    """Build the envs dict for forecast(). Fitters must use the TRUNCATED csv.

    train_file overrides VAL_TRAIN_FILE so each fold fits on its own prefix.
    fold is the fold name ("F1".."F5"), or None on the single split. Programs
    that persist fitted artefacts resolve them per fold from this, so one
    process can score five folds without reloading the wrong weights.
    """
    return {
        "path_to_test_x": path_to_test_x,
        "path_to_train": train_file or VAL_TRAIN_FILE,
        "fold": fold,
        "path_to_locations": LOCATION_FILE,
        "start_col": START_COL,
        "out_var": OUT_VAR,
        "output_len": OUTPUT_LEN,
        "capacity": CAPACITY,
    }


def _grader_settings(path_to_test_x):
    """Settings dict for the official grader (identifier + guard thresholds)."""
    return {
        "capacity": CAPACITY,
        "output_len": OUTPUT_LEN,
        "day_len": DAY_LEN,
        "path_to_test_x": path_to_test_x,
        "min_non_zero_ratio": MIN_NON_ZERO_RATIO,
        "min_distinct_ratio": MIN_DISTINCT_RATIO,
    }


def _list_windows(val_dir):
    """Discover and pair the infile/outfile validation windows."""
    infiles = sorted(glob.glob(os.path.join(val_dir, "infile", "*in.csv")))
    outfiles = sorted(glob.glob(os.path.join(val_dir, "outfile", "*out.csv")))
    if not infiles:
        raise ValueError(
            "No validation windows under {!r}. Build them first: "
            "python val_windows.py --train-file <245-day csv>".format(val_dir))
    if len(infiles) != len(outfiles):
        raise ValueError("infile/outfile mismatch: {} vs {}".format(len(infiles), len(outfiles)))
    return infiles, outfiles


def _penalty(reason, n_windows=None):
    """Build the standard failure result (large-negative combined_score)."""
    out = {"combined_score": PENALTY_COMBINED_SCORE, "valid": False, "reason": reason}
    if n_windows is not None:
        out["n_windows"] = n_windows
    return out


def _score_forecaster(module, val_dir, train_file=None, fold=None, n_windows=None):
    """
    Description:
        Score a loaded candidate on the validation windows through the OFFICIAL
        grader, one window at a time, then average. Raises MetricsError (from the
        grader) on degenerate predictions; the caller converts that to a penalty.
    Args:
        module: loaded candidate exposing forecast(envs)
        val_dir: folder with infile/ and outfile/ validation windows
        train_file: fold_train.csv for this fold (None = the global VAL_TRAIN_FILE)
        fold: fold name passed through to forecast() as envs["fold"]
        n_windows: optional cap (None = all)
    Returns:
        A metrics dict with combined_score, competition_score, rmse, mae,
        latest_rmse, window_scores, n_windows, eval_seconds, valid.
    """
    infiles, outfiles = _list_windows(val_dir)
    n = len(infiles) if n_windows is None else min(n_windows, len(infiles))

    maes, rmses, lrmses = [], [], []
    t0 = time.perf_counter()
    for w in range(n):
        # Bail out ourselves rather than letting OpenEvolve's timeout fire.
        if time.perf_counter() - t0 > EVAL_BUDGET_SECONDS:
            raise EvalBudgetExceeded(
                "exceeded {:.0f}s after {}/{} windows".format(
                    EVAL_BUDGET_SECONDS, w, n))
        infile = infiles[w]
        pred = np.asarray(module.forecast(_prep_env(infile, train_file, fold)))  # (134,288,1)

        td = TestData(path_to_data=outfiles[w], start_col=START_COL, farm_capacity=CAPACITY)
        turbines, raw_turbines = td.get_all_turbines()

        # Per-window, per-turbine arrays of shape (output_len, out_var) -- the
        # shape the real grader's turbine_scores expects (no cnt_sample axis).
        predictions, grounds, raw_df_lst = [], [], []
        for i in range(CAPACITY):
            gt = np.asarray(turbines[i])[:OUTPUT_LEN, -OUT_VAR:]
            predictions.append(np.asarray(pred[i]))
            grounds.append(gt)
            raw_df_lst.append(raw_turbines[i].iloc[:OUTPUT_LEN].reset_index(drop=True))

        settings = _grader_settings(infile)
        tm, tr, _lm, lr = om.regressor_detailed_scores(predictions, grounds, raw_df_lst, settings)
        maes.append(tm)
        rmses.append(tr)
        lrmses.append(lr)

    elapsed = time.perf_counter() - t0
    avg_mae = float(np.mean(maes))
    avg_rmse = float(np.mean(rmses))
    score = 0.5 * (avg_rmse + avg_mae)          # official ranking score, lower better

    if elapsed > TIMEOUT_WARN_SECONDS:
        sys.stderr.write(
            "WARNING: eval took {:.1f}s for {} windows -- near OpenEvolve's "
            "timeout (a timeout collapses to fitness 0.0, outranking valid "
            "negative scores). Reduce --n-windows or speed up the candidate.\n".format(elapsed, n))

    return {
        "combined_score": -score,               # OpenEvolve MAXIMIZES -> negate
        "competition_score": score,             # 0.5*(RMSE+MAE), lower better
        "rmse": avg_rmse,
        "mae": avg_mae,
        "latest_rmse": float(np.mean(lrmses)),
        # Per-window scores, same order as the window files. Already computed
        # above -- previously averaged away. Paired comparisons between two
        # candidates need these; comparing aggregate means does not cancel
        # window difficulty and overstates the comparison noise several-fold.
        "window_scores": [0.5 * (m + r) for m, r in zip(maes, rmses)],
        "pct_capacity": round(100.0 * score / FARM_CAPACITY_MW, 3),
        "n_windows": n,
        "eval_seconds": round(elapsed, 2),
        "valid": True,
    }


def evaluate(program_path, val_dir=None, n_windows=None):
    """
    Description:
        OpenEvolve entry point. Load and score the candidate on the VALIDATION
        split. NEVER raises: any failure (bad import, missing forecast, wrong
        shape, NaN, degenerate/guard-rejected prediction, runtime error) is
        returned as the penalty result.
    Args:
        program_path: path to the candidate program's .py file
        val_dir: validation window folder (defaults to VAL_DIR / SDWPF_VAL_WINDOWS)
        n_windows: optional cap on windows scored (None = all)
    Returns:
        A metrics dict; combined_score is the value OpenEvolve maximizes.
    """
    vdir = val_dir if val_dir is not None else VAL_DIR
    if n_windows is None:
        # OpenEvolve calls evaluate(program_path) with no cap, so without this
        # the env var only worked from the CLI and there was no way to shorten
        # an evaluation during a search run.
        n_windows = int(os.environ.get("SDWPF_MAX_WINDOWS", "0")) or None
    try:
        module = _load_program(program_path)
    except Exception as e:
        return _penalty("import error: {}".format(e))
    if not hasattr(module, "forecast"):
        return _penalty("program exposes no forecast(envs)")
    try:
        return _score_forecaster(module, vdir, n_windows=n_windows)
    except EvalBudgetExceeded as e:
        return _penalty("eval budget exceeded: {}".format(e))
    except om.MetricsError as e:
        # Degenerate prediction rejected by the official guards -- exactly the
        # gaming behaviour we want evolution to be penalised for.
        return _penalty("degenerate prediction: {}".format(e))
    except Exception as e:
        return _penalty("runtime error: {}".format(e))


def _list_folds(folds_root):
    """Discover fold directories written by `val_windows.py --layout rolling`."""
    dirs = sorted(d for d in glob.glob(os.path.join(folds_root, "F*"))
                  if os.path.isdir(os.path.join(d, "windows")))
    if not dirs:
        raise ValueError(
            "No folds under {!r}. Build them first: python val_windows.py "
            "--layout rolling --out-dir {}".format(folds_root, folds_root))
    return dirs


def evaluate_folds(program_path, folds_root=None, n_windows=None):
    """
    Description:
        Score a candidate across every fold of a rolling-origin layout, each
        fold fitting on its own fold_train.csv, and report the mean. Like
        evaluate(), this NEVER raises -- any failure collapses to the penalty.

        A candidate that fails on ANY fold is penalised outright rather than
        scored on the folds that happened to work. Scoring the survivors would
        silently change the number of folds behind a score and make candidates
        incomparable, which is the same class of bug as the official grader's
        variable turbine count.
    Args:
        program_path: candidate to score
        folds_root: directory containing F1/, F2/, ... (default SDWPF_FOLDS_ROOT)
        n_windows: optional per-fold cap on windows scored (None = all)
    Returns:
        A metrics dict. combined_score is what OpenEvolve maximizes; fold_scores
        and window_scores carry the extra information folds make available.
    """
    root = folds_root if folds_root is not None else os.environ.get("SDWPF_FOLDS_ROOT")
    if not root:
        return _penalty("no folds_root given and SDWPF_FOLDS_ROOT is unset")
    if n_windows is None:
        n_windows = int(os.environ.get("SDWPF_MAX_WINDOWS", "0")) or None

    try:
        fold_dirs = _list_folds(root)
    except Exception as e:
        return _penalty("fold discovery failed: {}".format(e))

    try:
        module = _load_program(program_path)
    except Exception as e:
        return _penalty("import error: {}".format(e))
    if not hasattr(module, "forecast"):
        return _penalty("program exposes no forecast(envs)")

    fold_scores, window_scores = {}, []
    rmses, maes, lrmses, seconds = [], [], [], 0.0

    for fdir in fold_dirs:
        fname = os.path.basename(fdir)
        try:
            r = _score_forecaster(
                module,
                os.path.join(fdir, "windows"),
                train_file=os.path.join(fdir, "fold_train.csv"),
                fold=fname,
                n_windows=n_windows)
        except EvalBudgetExceeded as e:
            return _penalty("{}: eval budget exceeded: {}".format(fname, e))
        except om.MetricsError as e:
            return _penalty("{}: degenerate prediction: {}".format(fname, e))
        except Exception as e:
            return _penalty("{}: runtime error: {}".format(fname, e))

        fold_scores[fname] = r["competition_score"]
        window_scores.extend(r["window_scores"])
        rmses.append(r["rmse"]); maes.append(r["mae"]); lrmses.append(r["latest_rmse"])
        seconds += r["eval_seconds"]

    # Mean over FOLDS, not over windows: folds differ in window count, and a
    # fold-weighted mean keeps every period contributing equally.
    scores = list(fold_scores.values())
    score = float(np.mean(scores))

    return {
        "combined_score": -score,               # OpenEvolve MAXIMIZES -> negate
        "competition_score": score,
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "latest_rmse": float(np.mean(lrmses)),
        "fold_scores": fold_scores,             # per-period breakdown
        "fold_std": float(np.std(scores)),      # this IS sigma_period
        "pct_capacity": round(100.0 * score / FARM_CAPACITY_MW, 3),
        "pct_capacity_std": round(100.0 * float(np.std(scores)) / FARM_CAPACITY_MW, 3),
        "fold_worst": max(scores),
        "window_scores": window_scores,         # for paired acceptance
        "n_folds": len(fold_scores),
        "n_windows": len(window_scores),
        "eval_seconds": round(seconds, 2),
        "valid": True,
    }


def accept_paired(cand_windows, parent_windows, k=1.0, block=3, n_boot=2000, seed=0):
    """
    Description:
        Noise-aware acceptance test for the evolve loop. Accepts a candidate
        only when it beats its parent by more than the noise in THAT COMPARISON.

        Both programs are scored on the same windows, so a window that is hard
        for everyone contributes nothing to distinguishing them -- it cancels in
        the difference. Testing the paired difference, rather than comparing two
        aggregate means, is what makes the threshold meaningful.
    Args:
        cand_windows / parent_windows: per-window score vectors (lower better)
        k: threshold in standard errors. 1.0 permissive, 2.0 ~ 95% one-sided
        block: moving-block length, to respect correlation between nearby windows
    Returns:
        (accept: bool, mean_delta: float, se: float) -- mean_delta negative =
        candidate better. Log all three; the rejected ones are the interesting data.
    """
    d = np.asarray(cand_windows, dtype=float) - np.asarray(parent_windows, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return False, float("nan"), float("nan")
    mean_d = float(d.mean())
    if d.size < block + 1:
        return bool(mean_d < 0), mean_d, float("nan")

    rng = np.random.default_rng(seed)
    blocks = np.array([d[i:i + block] for i in range(d.size - block + 1)])
    nb = int(np.ceil(d.size / float(block)))
    se = float(np.array([
        np.concatenate(blocks[rng.integers(0, len(blocks), nb)])[:d.size].mean()
        for _ in range(n_boot)]).std())
    return bool(mean_d < 0 and abs(mean_d) > k * se), mean_d, se


def main():
    """CLI: score one program on the validation split (or a fold layout)."""
    argv = [a for a in sys.argv[1:]]
    folds_root = None
    if "--folds-root" in argv:
        i = argv.index("--folds-root")
        try:
            folds_root = argv[i + 1]
        except IndexError:
            print("--folds-root needs a directory")
            sys.exit(1)
        del argv[i:i + 2]
    elif os.environ.get("SDWPF_FOLDS_ROOT"):
        folds_root = os.environ["SDWPF_FOLDS_ROOT"]

    if not argv:
        print("usage: python evaluator.py <program.py> [--folds-root <dir>]")
        sys.exit(1)

    n_cap = int(os.environ.get("SDWPF_MAX_WINDOWS", "0")) or None
    print("program : {}".format(argv[0]))
    print("metric  : OFFICIAL competition grader (kddcup22-sdwpf-evaluation, verbatim)")

    if folds_root:
        print("folds   : {}".format(folds_root))
        res = evaluate_folds(argv[0], folds_root=folds_root, n_windows=n_cap)
    else:
        print("val_dir : {}".format(VAL_DIR))
        res = evaluate(argv[0], n_windows=n_cap)

    for k in ("combined_score", "competition_score", "pct_capacity",
              "pct_capacity_std", "rmse", "mae", "latest_rmse",
              "fold_std", "fold_worst", "n_folds", "n_windows", "eval_seconds",
              "valid", "reason"):
        if k in res:
            print("  {}: {}".format(k, res[k]))
    if res.get("fold_scores"):
        print("  fold_scores:")
        for name, sc in res["fold_scores"].items():
            print("    {:<6} {:.4f}".format(name, sc))


if __name__ == "__main__":
    main()