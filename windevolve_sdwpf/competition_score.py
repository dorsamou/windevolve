# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/competition_score.py

Description:
    THE FINAL, LEADERBOARD-COMPARABLE SCORER. Run this exactly once, at the end
    of the project, on the champion program, against the locked-away
    final_phase_test set.

    Scoring is delegated to official_metrics.py, which is the REAL Baidu KDD Cup
    2022 submission grader copied verbatim from
        PaddleSpatial/apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/paddlepaddle/metrics.py
    -- i.e. the exact code that produced the competition leaderboard numbers.
    That file (not the simpler apps/wpf_baseline_gru/metrics.py demo scorer)
    carries the true masking rule (Patv < 0 OR (Patv == 0 & Wspd > 2.5)), the
    degeneracy guards (is_zero_prediction / is_identical_prediction /
    is_valid_prediction), and the latest-24h score. We import it unchanged so
    the number here is Baidu's number by construction.

    This scorer mirrors the official evaluation.py: it runs the program on each
    window, hands that window's per-turbine (predictions, grounds, raw frames)
    to the grader's regressor_detailed_scores, and averages the per-window
    totals. The final ranking score is 0.5 * (RMSE + MAE).

    The grader RAISES MetricsError on degenerate submissions (all-zero or
    near-constant predictions). For a real forecaster this never fires; for a
    gamed one it does -- exactly as in the competition.

    Contract: the program exposes
        forecast(envs) -> np.ndarray of shape (134, 288, 1) in kW,
    reading the infile at envs["path_to_test_x"] itself.

Citation:
    Scoring via the official Baidu KDD Cup 2022 SDWPF submission grader
    (PaddlePaddle/PaddleSpatial, .../kddcup22-sdwpf-evaluation/paddlepaddle),
    imported verbatim as official_metrics.py.
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    SDWPF_TEST_DIR=/path/to/final_phase_test \
    SDWPF_GRU_CHECKPOINTS=baseline_models/gru/checkpoints \
        python competition_score.py baseline_models/gru/gru_baseline.py
"""
import os                          # paths and env vars
import sys                         # program loading, argv
import glob                        # discover windows
import time                        # timing
import importlib.util             # load the forecaster by path

import numpy as np                 # arrays

from test_data import TestData     # official per-window loader (faithful copy)
import official_metrics as om      # the REAL competition grader, verbatim

CAPACITY, OUTPUT_LEN, OUT_VAR, START_COL, DAY_LEN = 134, 288, 1, 3, 144

TEST_DIR = os.environ.get(
    "SDWPF_TEST_DIR",
    "/home/dorsa/srip/wind_forecasting_data/sdwpf_kddcup/final_phase_test")
TRAIN_FILE = os.environ.get(
    "SDWPF_TRAIN_FILE",
    "/home/dorsa/srip/wind_forecasting_data/sdwpf_kddcup/sdwpf_245days_v1.csv")
LOCATION_FILE = os.environ.get(
    "SDWPF_LOCATION_FILE",
    "/home/dorsa/srip/wind_forecasting_data/sdwpf_kddcup/"
    "sdwpf_baidukddcup2022_turb_location.csv")

# Guard thresholds: the official competition defaults. A submission is rejected
# if too many turbines are all-zero or (near-)constant.
MIN_NON_ZERO_RATIO = float(os.environ.get("SDWPF_MIN_NON_ZERO_RATIO", "0.5"))
MIN_DISTINCT_RATIO = float(os.environ.get("SDWPF_MIN_DISTINCT_RATIO", "0.1"))


def _load_program(path):
    """Load a forecaster module from a .py path (must expose forecast(envs))."""
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prep_env(path_to_test_x=None):
    """Build the envs dict for forecast(); path_to_test_x is filled per window."""
    return {
        "path_to_test_x": path_to_test_x,
        "path_to_train": TRAIN_FILE,
        "path_to_locations": LOCATION_FILE,
        "start_col": START_COL,
        "out_var": OUT_VAR,
        "output_len": OUTPUT_LEN,
        "capacity": CAPACITY,
    }


def _grader_settings(path_to_test_x):
    """
    Build the settings dict the official grader expects. It parses
    path_to_test_x for the window identifier and reads the guard thresholds.
    """
    return {
        "capacity": CAPACITY,
        "output_len": OUTPUT_LEN,
        "day_len": DAY_LEN,
        "path_to_test_x": path_to_test_x,
        "min_non_zero_ratio": MIN_NON_ZERO_RATIO,
        "min_distinct_ratio": MIN_DISTINCT_RATIO,
    }


def score(program_path, test_dir=TEST_DIR, n_windows=None):
    """
    Description:
        Score a program on the given test set through the OFFICIAL competition
        grader, one window at a time, then average the per-window totals.
    Args:
        program_path: path to the program exposing forecast(envs)
        test_dir: folder containing infile/ and outfile/
        n_windows: cap on windows (None = all)
    Returns:
        A dict: total_mae, total_rmse, competition_score, latest_mae,
        latest_rmse, n_windows.
    """
    module = _load_program(program_path)
    infiles = sorted(glob.glob(os.path.join(test_dir, "infile", "*in.csv")))
    outfiles = sorted(glob.glob(os.path.join(test_dir, "outfile", "*out.csv")))
    if not infiles:
        raise ValueError("No windows under {!r}".format(test_dir))
    if len(infiles) != len(outfiles):
        raise ValueError("infile/outfile mismatch: {} vs {}".format(len(infiles), len(outfiles)))
    n = len(infiles) if n_windows is None else min(n_windows, len(infiles))

    maes, rmses, lmaes, lrmses = [], [], [], []
    t0 = time.perf_counter()
    for w in range(n):
        infile = infiles[w]
        # Run the program on this window.
        pred = np.asarray(module.forecast(_prep_env(infile)))       # (134, 288, 1)

        # Load this window's ground truth + raw per-turbine frames.
        td = TestData(path_to_data=outfiles[w], start_col=START_COL, farm_capacity=CAPACITY)
        turbines, raw_turbines = td.get_all_turbines()

        # Assemble the grader's contract for THIS window: predictions[i] and
        # gts[i] are (cnt_sample=1, output_len, out_var); raw_df_lst[i] is the
        # turbine's raw frame (first output_len rows).
        # The real grader is called ONCE PER WINDOW: predictions[i] and gts[i]
        # are this turbine's single-window arrays of shape (output_len, out_var)
        # -- NO cnt_sample axis. turbine_scores masks pred[indices] directly.
        predictions, grounds, raw_df_lst = [], [], []
        for i in range(CAPACITY):
            gt = np.asarray(turbines[i])[:OUTPUT_LEN, -OUT_VAR:]
            predictions.append(np.asarray(pred[i]))
            grounds.append(gt)
            raw_df_lst.append(raw_turbines[i].iloc[:OUTPUT_LEN].reset_index(drop=True))

        # OFFICIAL grader: returns (total_mae, total_rmse, latest_mae, latest_rmse).
        settings = _grader_settings(infile)
        tm, tr, lm, lr = om.regressor_detailed_scores(predictions, grounds, raw_df_lst, settings)
        maes.append(tm)
        rmses.append(tr)
        lmaes.append(lm)
        lrmses.append(lr)
        if (w + 1) % 40 == 0:
            print("  {}/{} windows ({:.0f}s)".format(w + 1, n, time.perf_counter() - t0))

    avg_mae = float(np.mean(maes))
    avg_rmse = float(np.mean(rmses))
    return {
        "total_mae": avg_mae,
        "total_rmse": avg_rmse,
        "competition_score": float(0.5 * (avg_rmse + avg_mae)),
        "latest_mae": float(np.mean(lmaes)),
        "latest_rmse": float(np.mean(lrmses)),
        "n_windows": n,
    }


def main():
    """Score the given program and print the official competition result."""
    if len(sys.argv) < 2:
        print("usage: python competition_score.py <program.py>")
        sys.exit(1)
    n_cap = int(os.environ.get("SDWPF_MAX_WINDOWS", "0")) or None
    print("program : {}".format(sys.argv[1]))
    print("metric  : OFFICIAL competition grader (kddcup22-sdwpf-evaluation, verbatim)")
    print("test_dir: {}".format(TEST_DIR))
    res = score(sys.argv[1], TEST_DIR, n_cap)
    print()
    print("=== FINAL COMPETITION SCORE ===")
    print("  total_mae         : {:.6f}".format(res["total_mae"]))
    print("  total_rmse        : {:.6f}".format(res["total_rmse"]))
    print("  competition_score : {:.6f}   (0.5*(rmse+mae), lower is better)".format(
        res["competition_score"]))
    print("  latest-24h mae    : {:.6f}".format(res["latest_mae"]))
    print("  latest-24h rmse   : {:.6f}".format(res["latest_rmse"]))
    print("  windows           : {}".format(res["n_windows"]))


if __name__ == "__main__":
    main()
