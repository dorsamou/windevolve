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
    carries the true masking rule (NaN rows, Patv < 0, Patv == 0 & Wspd > 2.5),
    the degeneracy guards (is_zero_prediction / is_identical_prediction /
    is_valid_prediction), and the latest-24h score. We import it unchanged so
    the number here is Baidu's number by construction.
    It reproduces:

      initiate_env()  capacity 134, output_len 288, out_var 1, day_len 144,
                      min_distinct_ratio 0.1, min_non_zero_ratio 0.9
      per window      metrics.regressor_detailed_scores(...)
                      performance(): raise if latest_rmse < 0
                                     acc = 1 - latest_rmse / capacity
                      accuracy gate: raise if acc <= 0
                      per-window print of RMSE / MAE / Score / Accuracy
                      MAX_TIMEOUT projection check
      after the loop  total_score keeps its 65535 sentinel unless exactly
                      NUM_MAX_RUNS (142) windows were graded
                      MIN_NOISE_LEVEL std floor on rmses / maes / accuracies
                      total_score = (avg_mae + avg_rmse) / 2

    HARNESS DIFFERENCES:
      * Official takes a submitted .zip, loads prepare.py, calls prep_env(), and
        checks the declared `framework` against the evaluator's own directory
        name. We take a path to a .py exposing forecast(envs).
      * Official unzips test_x.zip / test_y.zip and reads one NNNNin.zip /
        NNNNout.zip per window. We read already-unzipped CSVs from
        <test_dir>/infile and <test_dir>/outfile.
      * Official returns {"score": -total_score} because its leaderboard is
        higher-is-better. We return total_score positive, lower-is-better.

    DEVIATION:
      * MAX_TIMEOUT. Official raises when the projected remaining time will not
        fit the 10-hour budget. That budget was measured on the competition's
        4-core / Tesla V100 box; raising on different hardware would reject
        programs the competition would have accepted (and vice versa). The check
        is computed exactly as official does and printed as a WARNING instead of
        raising. Nothing else here is softened.

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
import time                        # timing / MAX_TIMEOUT budget
import importlib.util             # load the forecaster by path

import numpy as np                 # arrays

from test_data import TestData     # official per-window loader (faithful copy)
import official_metrics as om      # the REAL competition grader, verbatim

# --- evaluation.py: initiate_env() ------------------------------------------
CAPACITY, OUTPUT_LEN, OUT_VAR, START_COL, DAY_LEN = 134, 288, 1, 3, 144
MIN_NON_ZERO_RATIO = float(os.environ.get("SDWPF_MIN_NON_ZERO_RATIO", "0.9"))
MIN_DISTINCT_RATIO = float(os.environ.get("SDWPF_MIN_DISTINCT_RATIO", "0.1"))

# --- evaluation.py: module-level constants ----------------------------------
NUM_MAX_RUNS = int(os.environ.get("SDWPF_NUM_MAX_RUNS", "142"))
MAX_TIMEOUT = float(os.environ.get("SDWPF_MAX_TIMEOUT", str(3600 * 10)))  # 10 hours
MIN_TIME = 3                # 3 secs
MIN_NOISE_LEVEL = 0.05      # 5 %

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


class EvaluationError(Exception):
    """
    Description:
        Mirrors evaluation.py's EvaluationError: raised when the official grader
        would have refused to emit a score. Distinct from
        official_metrics.MetricsError, which the grader itself raises for
        degenerate per-window predictions.
    """


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
    Values are exactly those set by evaluation.py initiate_env().
    """
    return {
        "capacity": CAPACITY,
        "output_len": OUTPUT_LEN,
        "day_len": DAY_LEN,
        "path_to_test_x": path_to_test_x,
        "min_non_zero_ratio": MIN_NON_ZERO_RATIO,
        "min_distinct_ratio": MIN_DISTINCT_RATIO,
    }


def performance(settings, idx, prediction, ground_truth, ground_truth_df):
    """
    Description:
        Grade one window. Mirrors evaluation.py performance(): call the official
        grader, reject a negative latest-24h RMSE, and derive accuracy from it.
    Args:
        settings: the grader settings dict for this window
        idx: window index, used only in messages
        prediction: per-turbine predictions for this window
        ground_truth: per-turbine target arrays for this window
        ground_truth_df: per-turbine raw frames for this window
    Returns:
        A tuple (overall_mae, overall_rmse, acc, latest_mae, latest_rmse).
    """
    overall_mae, overall_rmse, overall_latest_mae, overall_latest_rmse = \
        om.regressor_detailed_scores(prediction, ground_truth, ground_truth_df, settings)
    if overall_latest_rmse < 0:
        raise EvaluationError("The RMSE of the last 24 hours is negative ({}) in the {}-th "
                              "prediction".format(overall_latest_rmse, idx))
    acc = 1 - overall_latest_rmse / settings["capacity"]
    return overall_mae, overall_rmse, acc, overall_latest_mae, overall_latest_rmse


def score(program_path, test_dir=TEST_DIR, n_windows=None, is_debug=False):
    """
    Description:
        Score a program through the OFFICIAL grader, reproducing evaluation.py's
        evaluate() control flow: per-window grading, per-window accuracy gate,
        MAX_TIMEOUT projection, the NUM_MAX_RUNS completeness gate with its
        65535 sentinel, and the MIN_NOISE_LEVEL across-window std floor.
    Args:
        program_path: path to the program exposing forecast(envs)
        test_dir: folder containing infile/ and outfile/
        n_windows: cap on windows (None = all). A capped run is incomplete by
            definition, so total_score keeps the sentinel -- exactly as official
            behaves when it collects fewer than NUM_MAX_RUNS results.
        is_debug: mirror evaluation.py's is_debug timing prints
    Returns:
        A dict: competition_score, total_mae, total_rmse, latest_mae,
        latest_rmse, accuracy, n_windows.
    """
    begin_time = time.time()
    left_time = MAX_TIMEOUT
    start_test_time = begin_time

    module = _load_program(program_path)
    infiles = sorted(glob.glob(os.path.join(test_dir, "infile", "*in.csv")))
    outfiles = sorted(glob.glob(os.path.join(test_dir, "outfile", "*out.csv")))
    if not infiles:
        raise EvaluationError("No windows under {!r}".format(test_dir))
    if len(infiles) != len(outfiles):
        raise EvaluationError("infile/outfile mismatch: {} vs {}".format(
            len(infiles), len(outfiles)))
    n = len(infiles) if n_windows is None else min(n_windows, len(infiles))

    maes, rmses, accuracies, lmaes, lrmses = [], [], [], [], []
    start_forecast_time = start_test_time
    end_forecast_time = start_forecast_time

    for i in range(n):
        infile = infiles[i]
        # Run the program on this window.
        prediction = np.asarray(module.forecast(_prep_env(infile)))     # (134, 288, 1)
        if is_debug:
            end_forecast_time = time.time()
            print("\nElapsed time for {}-th prediction is: "
                  "{} secs \n".format(i, end_forecast_time - start_forecast_time))
            start_forecast_time = end_forecast_time

        # Ground truth for this window, exactly as official
        # exec_predict_and_test(flag='test') builds it.
        td = TestData(path_to_data=outfiles[i], start_col=START_COL, farm_capacity=CAPACITY)
        turbines, raw_turbines = td.get_all_turbines()
        gt_ys = np.array([t[:OUTPUT_LEN, -OUT_VAR:] for t in turbines])

        settings = _grader_settings(infile)
        tmp_mae, tmp_rmse, tmp_acc, tmp_lmae, tmp_lrmse = \
            performance(settings, i, prediction, gt_ys, raw_turbines)

        # evaluation.py: accuracy gate. acc <= 0 means the last-24h summed RMSE
        # reached farm capacity (134 MW); the submission is rejected outright.
        if tmp_acc <= 0:
            print('\n\tThe {}-th prediction -- '
                  'RMSE: {}, MAE: {}, and Accuracy: {}'.format(i, tmp_mae, tmp_rmse, tmp_acc))
            raise EvaluationError("Accuracy ({:.3f}) is lower than Zero, which means that "
                                  "the RMSE (in latest 24 hours) of the {}th prediction "
                                  "is too large!".format(tmp_acc, i))
        else:
            print('\n\tThe {}-th prediction -- '
                  'RMSE: {}, MAE: {}, Score: {}, '
                  'and Accuracy: {:.4f}%'.format(i, tmp_rmse, tmp_mae,
                                                 (tmp_rmse + tmp_mae) / 2, tmp_acc * 100))
        maes.append(tmp_mae)
        rmses.append(tmp_rmse)
        accuracies.append(tmp_acc)
        lmaes.append(tmp_lmae)
        lrmses.append(tmp_lrmse)

        # evaluation.py: MAX_TIMEOUT projection. DEVIATION: warn, do not raise.
        cost_time = time.time() - begin_time
        left_time -= cost_time
        cnt_left_runs = NUM_MAX_RUNS - (i + 1)
        if i > 1 and left_time < MIN_TIME * (cnt_left_runs + 1):
            print("\nWARNING: the official evaluator would have raised TIMEOUT here -- on its "
                  "hardware, {} remaining predictions would not fit in {:.2f} secs. Not raised: "
                  "the 10-hour budget was set on the competition's 4-core/V100 box, so this "
                  "projection is not transferable to other hardware.\n".format(
                      cnt_left_runs, left_time))
        begin_time = time.time()

    # --- evaluation.py: completeness gate + 65535 sentinel -------------------
    # total_score is only overwritten when exactly NUM_MAX_RUNS windows were
    # graded. Otherwise it keeps the sentinel, which is what the official
    # evaluator hands the leaderboard for an incomplete run (as -65535).
    avg_mae, avg_rmse, total_score = -1, -1, 65535
    if len(maes) == NUM_MAX_RUNS:
        if np.std(np.array(rmses)) < MIN_NOISE_LEVEL or np.std(np.array(maes)) < MIN_NOISE_LEVEL \
                or np.std(np.array(accuracies)) < MIN_NOISE_LEVEL:
            raise EvaluationError("Std of rmses ({:.4f}) or std of maes ({:.4f}) or std of accs "
                                  "({:.4f}) is too small! ".format(
                                      np.std(np.array(rmses)), np.std(np.array(maes)),
                                      np.std(np.array(accuracies))))
        avg_mae = np.array(maes).mean()
        avg_rmse = np.array(rmses).mean()
        total_score = (avg_mae + avg_rmse) / 2
        print('\n --- Final MAE: {}, RMSE: {} ---'.format(avg_mae, avg_rmse))
        print('--- Final Score --- \n\t{}'.format(total_score))

    if is_debug:
        print("\nElapsed time for prediction is {} secs\n".format(
            end_forecast_time - start_test_time))
        print("\nTotal time for evaluation is {} secs\n".format(time.time() - start_test_time))

    return {
        "competition_score": float(total_score),
        "total_mae": float(avg_mae),
        "total_rmse": float(avg_rmse),
        "latest_mae": float(np.mean(lmaes)) if lmaes else -1.0,
        "latest_rmse": float(np.mean(lrmses)) if lrmses else -1.0,
        "accuracy": float(np.mean(accuracies)) if accuracies else -1.0,
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
    print("  accuracy          : {:.4f}%".format(res["accuracy"] * 100))
    print("  windows           : {}".format(res["n_windows"]))
    if res["competition_score"] == 65535:
        print("\n  *** 65535 is the OFFICIAL SENTINEL for an incomplete run: only {} of {} "
              "windows were graded, so no score was computed. This is exactly what the "
              "official evaluator hands the leaderboard here (as -65535). ***".format(
                  res["n_windows"], NUM_MAX_RUNS))


if __name__ == "__main__":
    main()
