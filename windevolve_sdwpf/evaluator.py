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

from test_data import TestData     # official per-window loader (faithful copy)
import official_metrics as om      # the REAL competition grader, verbatim

CAPACITY, OUTPUT_LEN, OUT_VAR, START_COL, DAY_LEN = 134, 288, 1, 3, 144

# ---------------------------------------------------------------------------
# The evolve loop scores on the VALIDATION split, not final_phase_test.
# val_windows.py writes <repo>/val_split/val_windows (infile/ + outfile/) and
# <repo>/val_split/val_train.csv. Both are overridable via env vars.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_VAL_DIR = os.path.join(_HERE, "val_split", "val_windows")
_DEFAULT_VAL_TRAIN = os.path.join(_HERE, "val_split", "val_train.csv")

VAL_DIR = os.environ.get("SDWPF_VAL_WINDOWS", _DEFAULT_VAL_DIR)
# Candidates that fit a model must train on the TRUNCATED csv (pre-val rows
# only), never the full 245 days -- otherwise they see their own val targets.
VAL_TRAIN_FILE = os.environ.get("SDWPF_VAL_TRAIN", _DEFAULT_VAL_TRAIN)
LOCATION_FILE = os.environ.get(
    "SDWPF_LOCATION_FILE",
    os.path.join(_HERE, "..", "wind_forecasting_data", "sdwpf_kddcup",
                 "sdwpf_baidukddcup2022_turb_location.csv"))

# Guard thresholds: official competition defaults.
MIN_NON_ZERO_RATIO = float(os.environ.get("SDWPF_MIN_NON_ZERO_RATIO", "0.5"))
MIN_DISTINCT_RATIO = float(os.environ.get("SDWPF_MIN_DISTINCT_RATIO", "0.1"))

# Returned on ANY failure; far below any real score (~ -40..-60) so a broken
# candidate can never outrank a working one.
PENALTY_COMBINED_SCORE = -1.0e6

# Soft budget warning: OpenEvolve kills candidates that exceed its per-eval
# timeout, and a killed eval collapses to fitness 0.0 -- which would OUTRANK
# valid negative scores. Warn if we get close.
TIMEOUT_WARN_SECONDS = float(os.environ.get("SDWPF_TIMEOUT_WARN", "250"))


def _load_program(path):
    """Import a candidate program by path; must expose forecast(envs)."""
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prep_env(path_to_test_x=None):
    """Build the envs dict for forecast(). Fitters must use the TRUNCATED csv."""
    return {
        "path_to_test_x": path_to_test_x,
        "path_to_train": VAL_TRAIN_FILE,
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


def _score_forecaster(module, val_dir, n_windows=None):
    """
    Description:
        Score a loaded candidate on the validation windows through the OFFICIAL
        grader, one window at a time, then average. Raises MetricsError (from the
        grader) on degenerate predictions; the caller converts that to a penalty.
    Args:
        module: loaded candidate exposing forecast(envs)
        val_dir: folder with infile/ and outfile/ validation windows
        n_windows: optional cap (None = all)
    Returns:
        A metrics dict with combined_score, competition_score, rmse, mae,
        latest_rmse, n_windows, eval_seconds, valid.
    """
    infiles, outfiles = _list_windows(val_dir)
    n = len(infiles) if n_windows is None else min(n_windows, len(infiles))

    maes, rmses, lrmses = [], [], []
    t0 = time.perf_counter()
    for w in range(n):
        infile = infiles[w]
        pred = np.asarray(module.forecast(_prep_env(infile)))       # (134, 288, 1)

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
    try:
        module = _load_program(program_path)
    except Exception as e:
        return _penalty("import error: {}".format(e))
    if not hasattr(module, "forecast"):
        return _penalty("program exposes no forecast(envs)")
    try:
        return _score_forecaster(module, vdir, n_windows=n_windows)
    except om.MetricsError as e:
        # Degenerate prediction rejected by the official guards -- exactly the
        # gaming behaviour we want evolution to be penalised for.
        return _penalty("degenerate prediction: {}".format(e))
    except Exception as e:
        return _penalty("runtime error: {}".format(e))


def main():
    """CLI: score one program on the validation split and print the result."""
    if len(sys.argv) < 2:
        print("usage: python evaluator.py <program.py>")
        sys.exit(1)
    n_cap = int(os.environ.get("SDWPF_MAX_WINDOWS", "0")) or None
    print("program : {}".format(sys.argv[1]))
    print("metric  : OFFICIAL competition grader (kddcup22-sdwpf-evaluation, verbatim)")
    print("val_dir : {}".format(VAL_DIR))
    res = evaluate(sys.argv[1], n_windows=n_cap)
    for k in ("combined_score", "competition_score", "rmse", "mae",
              "latest_rmse", "n_windows", "eval_seconds", "valid", "reason"):
        if k in res:
            print("  {}: {}".format(k, res[k]))


if __name__ == "__main__":
    main()
