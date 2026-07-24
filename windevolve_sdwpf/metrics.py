# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/metrics.py

Description:
    copy of the official Baidu KDD Cup 2022 SDWPF scorer.
    Implements the per-turbine MAE/RMSE with the official masking rules
    (missing / zero / unknown / abnormal), the kW->MW conversion, the
    degeneracy guards, and the sum-across-turbines aggregation. The guards 
    RAISE MetricsError; the thin evaluator (evaluator.py) catches those and
    converts them into an OpenEvolve penalty score.

Citation:
    official Baidu KDD Cup 2022 SDWPF evaluation code:
      PaddlePaddle/PaddleSpatial,
      apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation/paddlepaddle/metrics.py
      https://github.com/PaddlePaddle/PaddleSpatial  (accessed 2026-07-22)
    Dataset / methodology:
      Zhou et al. "SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting
      Challenge at KDD Cup 2022." arXiv:2208.04360, 2022.
      Zhou et al. "A Dataset for Spatial Dynamic Wind Power Forecasting over a
      Large Turbine Array." Scientific Data 11:649, 2024.
      doi:10.1038/s41597-024-03427-5
"""
import os                        # split the infile path to derive the window id
import logging                   # replaces the original stdout prints in the guards
import traceback                 # print full stack on unexpected ValueErrors

import numpy as np               # array math for the metrics
import pandas as pd              # NaN detection / column access on the raw frame

# Module logger. Guard messages go to DEBUG, so they are silent unless the
# caller explicitly lowers the level (logging.getLogger("metrics").setLevel(...)).
logger = logging.getLogger(__name__)


class MetricsError(Exception):
    """
    Description:
        Custom exception raised by the scorer when a prediction is invalid or
        degenerate (NaN, empty, all-zero, or (near-)constant across the farm).
    """
    def __init__(self, err_message):
        """
        Description:
            Store the error message on the exception.
        Args:
            err_message: human-readable reason the prediction was rejected
        Returns:
            None
        """
        Exception.__init__(self, err_message)


def is_valid_prediction(prediction, idx=None):
    """
    Description:
        Reject a prediction that contains NaN values or is empty.
    Args:
        prediction: one turbine's predicted array (2-D, e.g. shape (288, 1))
        idx: optional window index, used only to label the error message
    Returns:
        True if the prediction is usable; otherwise raises MetricsError.
    """
    try:
        # A NaN anywhere would poison the downstream arithmetic, so reject it.
        if prediction.ndim > 1:
            nan_prediction = pd.isna(prediction).any(axis=1)
            if nan_prediction.any():
                msg = ("NaN in predicted values!" if idx is None
                       else "NaN in predicted values ({}th prediction)!".format(idx))
                raise MetricsError(msg)
        # An empty array means the forecaster returned nothing.
        if prediction.size == 0:
            msg = ("Empty prediction!" if idx is None
                   else "Empty predicted values ({}th prediction)! ".format(idx))
            raise MetricsError(msg)
    except ValueError as e:
        # Normalize any unexpected error into a single MetricsError type.
        traceback.print_exc()
        if idx is None:
            raise MetricsError("Value Error: {}. ".format(e))
        raise MetricsError("Value Error: {} in {}th prediction. ".format(e, idx))
    return True


def mae(pred, gt, run_id=0):
    """
    Description:
        Mean Absolute Error of two already-cleaned, MW-scaled arrays.
    Args:
        pred: predicted values (validated, masked, in MW)
        gt: ground-truth values (same shape as pred)
        run_id: optional window index, used only for error messages
    Returns:
        The MAE as a float (or -1 if the prediction failed validation).
    """
    _mae = -1
    if is_valid_prediction(pred, idx=run_id):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction ({}) and Ground Truth ({}) "
                               "in {}th prediction! ".format(pred.shape, gt.shape, run_id))
        _mae = np.mean(np.abs(pred - gt))
    return _mae


def mse(pred, gt, run_id=0):
    """
    Description:
        Mean Square Error of two already-cleaned, MW-scaled arrays.
    Args:
        pred: predicted values (validated, masked, in MW)
        gt: ground-truth values (same shape as pred)
        run_id: optional window index, used only for error messages
    Returns:
        The MSE as a float (or -1 if the prediction failed validation).
    """
    _mse = -1
    if is_valid_prediction(pred, idx=run_id):
        if pred.shape != gt.shape:
            raise MetricsError("Different shapes between Prediction ({}) and Ground Truth ({}) "
                               "in {}th prediction! ".format(pred.shape, gt.shape, run_id))
        _mse = np.mean((pred - gt) ** 2)
    return _mse


def rmse(pred, gt, run_id=0):
    """
    Description:
        Root Mean Square Error of two already-cleaned, MW-scaled arrays.
    Args:
        pred: predicted values (validated, masked, in MW)
        gt: ground-truth values (same shape as pred)
        run_id: optional window index, used only for error messages
    Returns:
        The RMSE as a float (or -1 if the underlying MSE was invalid).
    """
    _mse = mse(pred, gt, run_id=run_id)
    if _mse < 0:
        return -1
    return np.sqrt(_mse)


def regressor_scores(prediction, gt, idx=0):
    """
    Description:
        Compute MAE and RMSE together for two already-cleaned, MW-scaled arrays.
    Args:
        prediction: predicted values (validated, masked, in MW)
        gt: ground-truth values (same shape as prediction)
        idx: optional window index, used only for error messages
    Returns:
        A tuple (mae, rmse) of floats.
    """
    _mae = mae(prediction, gt, run_id=idx)
    _rmse = rmse(prediction, gt, run_id=idx)
    return _mae, _rmse


def turbine_scores(pred, gt, raw_data, examine_len, idx=0):
    """
    Description:
        MAE and RMSE (in MW) of ONE turbine over one window. Builds the missing
        mask (any-NaN row) and the invalid mask (zero/unknown/abnormal rules),
        excludes those rows, converts kW->MW, then scores the surviving last
        `examine_len` steps.
    Args:
        pred: this turbine's predicted array over the horizon (kW)
        gt: this turbine's ground-truth Patv over the horizon (kW)
        raw_data: this turbine's raw outfile frame (all columns, NaNs kept)
        examine_len: number of trailing steps to score (288 real, 144 latest)
        idx: optional window index, used only for error messages
    Returns:
        A tuple (mae, rmse) in MW, or (-1, -1) if nothing survives the mask.
    """
    # "Missing" mask: any row that has a NaN in any column is excluded.
    nan_cond = pd.isna(raw_data).any(axis=1)
    # "Invalid" mask: the zero / unknown / abnormal caveats rules.
    invalid_cond = (raw_data['Patv'] < 0) | \
                   ((raw_data['Patv'] == 0) & (raw_data['Wspd'] > 2.5)) | \
                   ((raw_data['Pab1'] > 89) | (raw_data['Pab2'] > 89) | (raw_data['Pab3'] > 89)) | \
                   ((raw_data['Wdir'] < -180) | (raw_data['Wdir'] > 180) | (raw_data['Ndir'] < -720) |
                    (raw_data['Ndir'] > 720))
    # Keep only rows that are neither missing nor invalid (rows are removed, not zeroed).
    indices = np.where(~nan_cond & ~invalid_cond)
    prediction = pred[indices]
    targets = gt[indices]
    # kW -> MW conversion happens HERE, after masking, right before the metric.
    _mae, _rmse = -1, -1
    if np.any(prediction) and np.any(targets):
        _mae, _rmse = regressor_scores(prediction[-examine_len:] / 1000, targets[-examine_len:] / 1000, idx=idx)
    return _mae, _rmse


def check_zero_prediction(prediction, idx=None):
    """
    Description:
        Flag whether ONE turbine's prediction is entirely zero.
    Args:
        prediction: one turbine's predicted array
        idx: optional window index, used only for the log message
    Returns:
        -1 if the prediction is all zeros, otherwise 0.
    """
    if not np.any(prediction):
        msg = ("Zero prediction!" if idx is None
               else "Zero predicted values ({}th prediction)! ".format(idx))
        logger.debug(msg)          # silenced by default (was print in the original)
        return -1
    return 0


def is_zero_prediction(predictions, identifier, settings):
    """
    Description:
        Farm-level all-zero guard: reject the window if too many turbines
        (fewer than min_non_zero_ratio remain non-zero) predict all zeros.
    Args:
        predictions: sequence of 134 per-turbine predicted arrays
        identifier: window index, used only for the error message
        settings: dict holding "capacity" and "min_non_zero_ratio"
    Returns:
        False if acceptable; otherwise raises MetricsError.
    """
    wind_farm_statuses = []
    for i in range(settings["capacity"]):
        # -1 for an all-zero turbine, 0 otherwise.
        status = check_zero_prediction(predictions[i], idx=identifier)
        wind_farm_statuses.append(status)
    statuses = np.array(wind_farm_statuses)
    non_zero_predictions = statuses[statuses == 0]                 # the healthy turbines
    non_zero_ratio = non_zero_predictions.size / settings["capacity"]
    if non_zero_ratio < settings["min_non_zero_ratio"]:
        msg = "{:.2f}% turbines with zero predicted values " \
              "(in {}th prediction)!".format((1 - non_zero_ratio) * 100, identifier)
        raise MetricsError(msg)
    return False


def check_identical_prediction(prediction, min_std=0.1, min_distinct_ratio=0.1, idx=None):
    """
    Description:
        Flag whether ONE turbine's prediction is constant or nearly constant
        (exactly flat, or tiny spread with very few distinct values).
    Args:
        prediction: one turbine's predicted array
        min_std: spread below which the series is considered nearly flat
        min_distinct_ratio: unique-value fraction below which it is "identical"
        idx: optional window index, used only for the log message
    Returns:
        -1 if the prediction is (near-)constant, otherwise 0.
    """
    try:
        # Exactly constant: min equals max.
        if np.min(prediction) == np.max(prediction):
            msg = ("All predicted values are the same as {:.4f}!".format(np.min(prediction)) if idx is None
                   else "All predicted values are same as {:.4f} ({}th prediction)!".format(np.min(prediction), idx))
            logger.debug(msg)      # silenced by default (was print in the original)
            return -1
        # Near-constant: tiny spread AND very few distinct values.
        if np.std(prediction) <= min_std:
            prediction = np.ravel(prediction)
            distinct_prediction = set(prediction)
            distinct_ratio = len(distinct_prediction) / np.size(prediction)
            samples = list(distinct_prediction)[:3]
            samples = ",".join("{:.5f}".format(s) for s in samples)
            if distinct_ratio < min_distinct_ratio:
                msg = ("{:.2f}% of predicted values are same! Some predicted values are: "
                       "{},...".format((1 - distinct_ratio) * 100, samples) if idx is None
                       else "{:.2f}% of predicted values are same ({}th run)! Some predicted values are:"
                            "{},...".format((1 - distinct_ratio) * 100, idx, samples))
                logger.debug(msg)  # silenced by default (was print in the original)
                return -1
    except ValueError as e:
        # Normalize any unexpected error into a single MetricsError type.
        traceback.print_exc()
        if idx is None:
            raise MetricsError("Value Error: {}. ".format(e))
        raise MetricsError("Value Error: {} in {}th prediction. ".format(e, idx))
    return 0


def is_identical_prediction(predictions, identifier, settings):
    """
    Description:
        Farm-level constant-output guard: reject the window if too few turbines
        (fewer than min_distinct_ratio vary) show real variation.
    Args:
        predictions: sequence of 134 per-turbine predicted arrays
        identifier: window index, used only for the error message
        settings: dict holding "capacity" and "min_distinct_ratio"
    Returns:
        False if acceptable; otherwise raises MetricsError.
    """
    farm_check_statuses = []
    for i in range(settings["capacity"]):
        # -1 for a (near-)constant turbine, 0 otherwise.
        status = check_identical_prediction(predictions[i],
                                             min_distinct_ratio=settings["min_distinct_ratio"],
                                             idx=identifier)
        farm_check_statuses.append(status)
    statuses = np.array(farm_check_statuses)
    variational_predictions = statuses[statuses == 0]              # the turbines that vary
    variation_ratio = variational_predictions.size / settings["capacity"]
    if variation_ratio < settings["min_distinct_ratio"]:
        msg = "{:.2f}% turbines with (almost) identical predicted values " \
              "({}th prediction)!".format((1 - variation_ratio) * 100, identifier)
        raise MetricsError(msg)
    return False


def regressor_detailed_scores(predictions, gts, raw_df_lst, settings):
    """
    Description:
        Score ONE window across all turbines. Runs the farm-level degeneracy
        guards, scores each turbine with turbine_scores, skips NaN / fully
        masked turbines, then SUMS the per-turbine MAE and RMSE over the farm.
        Also returns the last-24h (day_len) sums used only for the diagnostic.
    Args:
        predictions: sequence of 134 per-turbine predicted arrays (kW)
        gts: sequence of 134 per-turbine ground-truth Patv arrays (kW)
        raw_df_lst: sequence of 134 per-turbine raw frames (NaNs kept)
        settings: dict with capacity, output_len, day_len, guard thresholds,
                  and path_to_test_x (used to derive the window identifier)
    Returns:
        A tuple (total_mae, total_rmse, total_latest_mae, total_latest_rmse);
        raises MetricsError on degenerate or fully-invalid windows.
    """
    # Derive the window index from the infile name ("0005in.csv" -> 4).
    path_to_test_x = settings["path_to_test_x"]
    tokens = os.path.split(path_to_test_x)
    identifier = int(tokens[-1][:-6]) - 1
    all_mae, all_rmse = [], []
    all_latest_mae, all_latest_rmse = [], []
    # Called for their side effect: they raise on degenerate (constant/zero) input.
    if not is_identical_prediction(predictions, identifier, settings) and \
            not is_zero_prediction(predictions, identifier, settings):
        pass
    for i in range(settings["capacity"]):
        prediction = predictions[i]
        # Reject NaN/empty predictions for this turbine (raises inside).
        if not is_valid_prediction(prediction, idx=identifier):
            continue
        gt = gts[i]
        raw_df = raw_df_lst[i]
        # Full-horizon (288) score for this turbine.
        _mae, _rmse = turbine_scores(prediction, gt, raw_df, settings["output_len"], idx=identifier)
        if _mae != _mae or _rmse != _rmse:   # NaN check (NaN != NaN)
            continue
        if -1 == _mae or -1 == _rmse:         # everything masked out for this turbine
            continue
        all_mae.append(_mae)
        all_rmse.append(_rmse)
        # Last-24h (144) score, kept only for the diagnostic "accuracy".
        latest_mae, latest_rmse = turbine_scores(prediction, gt, raw_df, settings["day_len"], idx=identifier)
        all_latest_mae.append(latest_mae)
        all_latest_rmse.append(latest_rmse)
    # Farm score for this window = SUM of the surviving per-turbine scores.
    total_mae = np.array(all_mae).sum()
    total_rmse = np.array(all_rmse).sum()
    # A negative sum means -1 sentinels leaked in -> too many invalid turbines.
    if total_mae < 0 or total_rmse < 0:
        raise MetricsError("{}th prediction: summed MAE ({:.2f}) or RMSE ({:.2f}) is negative, "
                           "which indicates too many invalid values "
                           "in the prediction! ".format(identifier, total_mae, total_rmse))
    # No turbine produced a valid score.
    if len(all_mae) == 0 or len(all_rmse) == 0 or total_mae == 0 or total_rmse == 0:
        raise MetricsError("No valid MAE or RMSE for "
                           "all of the turbines in {}th prediction! ".format(identifier))
    total_latest_mae = np.array(all_latest_mae).sum()
    total_latest_rmse = np.array(all_latest_rmse).sum()
    return total_mae, total_rmse, total_latest_mae, total_latest_rmse
