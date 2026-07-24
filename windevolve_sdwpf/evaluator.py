# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/evaluator.py

Description:
    OpenEvolve-facing evaluator for the SDWPF / KDD Cup 2022 task. Mirrors the
    official grader but scores a GIVEN program (an OpenEvolve candidate or
    initial_program.py) instead of an uploaded submission. For each of the 142 windows it runs
    the program's forecast() on the infile, loads ground truth from the outfile,
    and scores with metrics.regressor_detailed_scores; the final score is the
    mean over windows of the summed-per-turbine (MAE, RMSE), combined as
    0.5*(RMSE+MAE) (lower is better). evaluate() never raises: any failure is
    returned as a large negative penalty so it cannot corrupt OpenEvolve's
    fitness ranking (a raised exception would be replaced by fitness 0.0, which
    would wrongly outrank valid negative scores).

    Program contract (kept from the official grader):
      forecast(envs) reads the infile at envs["path_to_test_x"] itself and
      returns predictions of shape (134, 288, 1) in kW. As in the official
      evaluator the array is passed through unchanged, so a wrong shape fails
      on the ground-truth comparison inside metrics.mae.
      envs also exposes "path_to_train" (the 245-day training CSV) and
      "path_to_locations" (turbine x/y), so a candidate may fit a model. Because
      forecast() is called once per window (142 times per evaluation), any such
      loading or fitting MUST be cached at module level and done once.

Citation:
    official Baidu KDD Cup 2022 SDWPF evaluator:
      PaddlePaddle/PaddleSpatial,
      apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/paddlepaddle/evaluation.py
      https://github.com/PaddlePaddle/PaddleSpatial  (accessed 2026-07-22)
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # Score the persistence seed over all 142 windows:
    #   python evaluator.py initial_program.py
    # Score just the first 10 windows of a given test dir:
    #   python evaluator.py initial_program.py "/path/to/final_phase_test" 10
"""
import os                        # path handling and env-var lookup
import sys                       # add this package to the import path
import glob                      # discover the infile/outfile windows
import time                      # measure per-eval wall-clock time
import logging                   # warn when an eval approaches the timeout
import importlib.util            # load the candidate program from its path

import numpy as np               # array handling / mean / std

# Make this package importable no matter where OpenEvolve invokes the file from.
_HERE = os.path.dirname(os.path.abspath(__file__))   # this file's directory
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import metrics                   # faithful scorer copy (this package)
from test_data import TestData   # faithful loader copy (this package)

# Module logger for the slow-eval warning.
logger = logging.getLogger(__name__)

# Path to the unzipped final_phase_test folder (must contain infile/ and outfile/).
# Overridable via the SDWPF_TEST_DIR environment variable.
TEST_DIR = os.environ.get(
    "SDWPF_TEST_DIR",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/final_phase_test",
)

# Path to the 245-day training CSV. Passed to the program via envs so a candidate
# may fit a model on it. NOTE: this file is ~4.7M rows -- a program MUST load/fit
# it once and cache the result, since forecast() is called once per window (142
# times per evaluation). Overridable via SDWPF_TRAIN_FILE.
TRAIN_FILE = os.environ.get(
    "SDWPF_TRAIN_FILE",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/sdwpf_245days_v1.csv",
)

# Path to the turbine location CSV (TurbID, x, y), enabling spatial models.
# Overridable via SDWPF_LOCATION_FILE.
LOCATION_FILE = os.environ.get(
    "SDWPF_LOCATION_FILE",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/sdwpf_baidukddcup2022_turb_location.csv",
)

# Score returned on ANY failure; far below any real score (~ -40..-60) so
# failed candidates always rank last. Returned, never raised, so OpenEvolve
# honors it instead of substituting fitness 0.0.
PENALTY_COMBINED_SCORE = -1.0e6

# Official threshold on the std of the per-window scores (cross-window guard).
MIN_NOISE_LEVEL = 0.05

# Only apply the cross-window std guard once at least this many windows exist
# (the std of a couple of numbers on a debug subset is meaningless).
STD_CHECK_MIN_WINDOWS = 10

# Warn if a single full eval exceeds this, approaching OpenEvolve's 300s timeout.
SLOW_EVAL_WARN_SECONDS = 150.0


def prep_env():
    """
    Description:
        Build the settings dict shared by every window (matches the official
        prep_env / initiate_env values). "path_to_test_x" is filled per window.
    Args:
        None
    Returns:
        A settings dict of dataset/scoring constants.
    """
    return {
        "task": "MS",                 # multivariate-in, single-out
        "target": "Patv",             # target column name
        "capacity": 134,              # number of turbines
        "start_col": 3,               # first modeled column (Wspd)
        "in_var": 10,                 # number of modeled input variables
        "out_var": 1,                 # number of output variables (Patv)
        "max_input_len": 2016,        # official Lx cap: 14 days x 144 steps/day.
                                      # The full history is present in every
                                      # infile; unused in scoring, available to
                                      # a program as a hint via envs.
        "output_len": 288,            # forecast horizon = 48h -> the ranking score
        "day_len": 144,               # 24h -> latest-day diagnostic only
        "min_non_zero_ratio": 0.9,    # farm-level all-zero guard threshold
        "min_distinct_ratio": 0.1,    # farm-level constant-output guard threshold
        "path_to_train": TRAIN_FILE,      # 245-day training CSV (load+fit ONCE, cache)
        "path_to_locations": LOCATION_FILE,  # turbine x/y positions for spatial models
    }


def _load_program(program_path):
    """
    Description:
        Import a candidate program file by path and return the module object.
    Args:
        program_path: filesystem path to the program's .py file
    Returns:
        The imported module (expected to expose forecast(envs)).
    """
    spec = importlib.util.spec_from_file_location("evolved_program", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exec_predict_and_test(envs, test_file, forecast_module, flag='predict'):
    """
    Description:
        Handle one window file: either run the program on the infile
        ('predict') or load ground truth from the outfile ('test').
    Args:
        envs: settings dict; "path_to_test_x" is set here in the predict branch
        test_file: path to the window CSV (an infile for predict, outfile for test)
        forecast_module: the loaded candidate program (used in the predict branch)
        flag: 'predict' to run the forecaster, 'test' to load ground truth
    Returns:
        For 'predict': {"prediction": <array>}.
        For 'test': {"ground_truth_y": <array>, "ground_truth_df": <list of frames>}.
    """
    if flag == 'predict':
        envs["path_to_test_x"] = test_file            # the program reads this path
        return {"prediction": forecast_module.forecast(envs)}
    elif flag == 'test':
        test_data = TestData(path_to_data=test_file, start_col=envs["start_col"])
        turbines, raw_turbines = test_data.get_all_turbines()
        # Ground truth = first output_len rows, last out_var columns (Patv), per turbine.
        test_ys = [turbine[:envs["output_len"], -envs["out_var"]:] for turbine in turbines]
        return {"ground_truth_y": np.array(test_ys), "ground_truth_df": raw_turbines}
    else:
        raise ValueError("Unsupported flag (only 'predict' or 'test').")


def performance(settings, idx, prediction, ground_truth, ground_truth_df):
    """
    Description:
        Score one window and repackage the raw metric outputs into
        (MAE, RMSE, accuracy). Accuracy is a diagnostic only, not the ranking.
    Args:
        settings: the settings dict for this run
        idx: window index (used only for error messages)
        prediction: this window's predictions, shape (134, 288, 1)
        ground_truth: this window's ground-truth Patv, shape (134, 288, 1)
        ground_truth_df: this window's 134 raw per-turbine frames
    Returns:
        A tuple (overall_mae, overall_rmse, accuracy) for this window.
    """
    overall_mae, overall_rmse, _, overall_latest_rmse = \
        metrics.regressor_detailed_scores(prediction, ground_truth, ground_truth_df, settings)
    # A negative latest RMSE means too many invalid turbines leaked -1 sentinels.
    if overall_latest_rmse < 0:
        raise metrics.MetricsError(
            "Negative latest-24h RMSE ({}) in the {}-th prediction".format(overall_latest_rmse, idx))
    acc = 1 - overall_latest_rmse / settings["capacity"]   # convenience 0..1 diagnostic
    return overall_mae, overall_rmse, acc


def _list_windows(test_dir):
    """
    Description:
        Discover and pair the infile/outfile window CSVs under a test dir.
    Args:
        test_dir: folder containing infile/*in.csv and outfile/*out.csv
    Returns:
        A tuple (infiles, outfiles) of sorted, equal-length path lists.
    """
    infiles = sorted(glob.glob(os.path.join(test_dir, "infile", "*in.csv")))
    outfiles = sorted(glob.glob(os.path.join(test_dir, "outfile", "*out.csv")))
    if len(infiles) != len(outfiles):
        raise ValueError("infile/outfile count mismatch: {} vs {}".format(len(infiles), len(outfiles)))
    if len(infiles) == 0:
        raise ValueError("No test windows found under {!r}".format(test_dir))
    return infiles, outfiles


def _penalty(reason, n_windows=None):
    """
    Description:
        Build the standard failure result (large negative combined_score).
    Args:
        reason: short human-readable failure reason
        n_windows: optional number of windows attempted
    Returns:
        A metrics dict with combined_score = PENALTY_COMBINED_SCORE and valid=False.
    """
    out = {"combined_score": PENALTY_COMBINED_SCORE, "valid": False, "reason": reason}
    if n_windows is not None:
        out["n_windows"] = n_windows
    return out


def evaluate_forecaster(forecast_module, test_dir, n_windows=None):
    """
    Description:
        Score a loaded program over the test windows: loop each window, run the
        forecaster, load ground truth, score, then average across windows and
        form 0.5*(RMSE+MAE). Applies the cross-window std guard for full runs.
    Args:
        forecast_module: the loaded candidate program exposing forecast(envs)
        test_dir: folder with infile/ and outfile/ windows
        n_windows: cap on the number of windows (None = all; official = 142)
    Returns:
        A metrics dict (see evaluate) with combined_score, competition_score,
        rmse, mae, accuracy, n_windows, eval_seconds, valid.
    """
    settings = prep_env()
    infiles, outfiles = _list_windows(test_dir)
    n = len(infiles) if n_windows is None else min(n_windows, len(infiles))

    start_time = time.perf_counter()
    maes, rmses, accs = [], [], []
    for i in range(n):
        # Run the program on window i's infile (passed straight through, as in
        # the official evaluator -- forecast() must return (134, 288, 1) or
        # metrics.mae raises on the shape mismatch).
        pred_res = exec_predict_and_test(settings, infiles[i], forecast_module, flag='predict')
        prediction = pred_res["prediction"]
        # Load window i's ground truth + raw frames from the outfile.
        test_res = exec_predict_and_test(settings, outfiles[i], forecast_module, flag='test')
        gt = test_res["ground_truth_y"]
        gt_df = test_res["ground_truth_df"]
        # Score this window and collect its metrics.
        m, r, a = performance(settings, i, prediction, gt, gt_df)
        maes.append(m)
        rmses.append(r)
        accs.append(a)

    # Cross-window guard: implausibly identical per-window scores == degenerate.
    # Skipped for small debug subsets where the std is just noise.
    if n >= STD_CHECK_MIN_WINDOWS and (np.std(rmses) < MIN_NOISE_LEVEL or np.std(maes) < MIN_NOISE_LEVEL):
        return _penalty("cross-window scores nearly identical (std below threshold)", n)

    # Mean over windows, then the competition combination.
    avg_mae = float(np.mean(maes))
    avg_rmse = float(np.mean(rmses))
    score = 0.5 * (avg_rmse + avg_mae)            # competition score (lower is better)

    # Warn if we are approaching OpenEvolve's per-candidate timeout.
    elapsed = time.perf_counter() - start_time
    if elapsed > SLOW_EVAL_WARN_SECONDS:
        logger.warning(
            "SDWPF eval took %.1fs for %d windows -- approaching OpenEvolve's 300s "
            "per-candidate timeout (a timeout collapses to fitness 0.0, which "
            "outranks valid negative scores). Consider speeding up the eval.",
            elapsed, n,
        )

    return {
        # OpenEvolve MAXIMIZES -> negate. A VALID combined_score is NEGATIVE.
        "combined_score": -score,           # fitness key OpenEvolve maximizes
        "competition_score": score,         # official 0.5*(RMSE+MAE), lower better
        "rmse": avg_rmse,                   # mean summed-per-turbine RMSE (MW)
        "mae": avg_mae,                     # mean summed-per-turbine MAE (MW)
        "accuracy": float(np.mean(accs)),   # 0..1 diagnostic only
        "n_windows": n,                     # number of windows scored
        "eval_seconds": round(elapsed, 2),  # wall-clock cost (timeout monitor)
        "valid": True,                      # this candidate scored successfully
    }


def evaluate(program_path, test_dir=None, n_windows=None):
    """
    Description:
        OpenEvolve entry point. Load the candidate program and score it. Never
        raises: any failure (bad import, missing forecast, degenerate/NaN
        prediction, runtime error) is returned as the penalty result.
    Args:
        program_path: path to the candidate program's .py file
        test_dir: test folder (defaults to TEST_DIR / SDWPF_TEST_DIR)
        n_windows: optional cap on windows scored (None = all)
    Returns:
        A metrics dict; combined_score is the value OpenEvolve maximizes.
    """
    test_dir = test_dir or TEST_DIR
    try:
        program = _load_program(program_path)
    except Exception as e:  # noqa: BLE001
        return _penalty("load error: {}".format(e))
    if not hasattr(program, "forecast"):
        return _penalty("program has no forecast(envs) function")
    try:
        return evaluate_forecaster(program, test_dir, n_windows=n_windows)
    except Exception as e:  # noqa: BLE001 -- includes MetricsError from degenerate candidates
        return _penalty("runtime error: {}".format(e))


if __name__ == "__main__":
    # Standalone: python evaluator.py <program.py> [test_dir] [n_windows]
    prog = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "initial_program.py")
    td = sys.argv[2] if len(sys.argv) > 2 else TEST_DIR
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print("program :", prog)
    print("test_dir:", td)
    result = evaluate(prog, test_dir=td, n_windows=nw)
    for k, v in result.items():
        print("  {}: {}".format(k, v))