"""
OpenEvolve evaluator: SDWPF FULL FARM, KDD Cup 2022 protocol.

Scores exactly like the competition so results sit on the published leaderboard
scale (the paper's GRU baseline = 42.32 on the 214/31 split):
  - forecast output_len=288 steps (48h) ahead from input_len=144 history, no future weather;
  - per turbine: roll K origins over the validation days, compute RMSE and MAE in
    MEGAWATTS on the unmasked steps, average over origins;
  - FARM score = sum over all 134 turbines of (RMSE + MAE)/2. Lower is better.
  - KDD masking (curtailed/abnormal excluded) and Patv<0 -> 0 applied.

The SAME forecast() is applied to every turbine, so the farm score measures
cross-turbine generalization.

Split: train first TRAIN_DAYS, validate on the next VAL_DAYS (default 214/31 to
match the paper's baseline). The sealed final_phase_test.zip is NEVER used here --
that's the final leaderboard number, scored once on the best program separately.

The per-turbine data/windows are cached on first call (built once, reused every
iteration) so only forecast() re-runs per candidate.

Env (defaults = competition-comparable):
  SDWPF_CSV (the full 245-day file), SDWPF_INPUT_LEN=144, SDWPF_OUTPUT_LEN=288,
  SDWPF_TRAIN_DAYS=214, SDWPF_VAL_DAYS=31, SDWPF_DAY_LEN=144, SDWPF_N_ORIGINS=64,
  SDWPF_MAX_TURBINES (limit turbines for speed while developing; default all).
"""
import os
import importlib.util
import traceback
import concurrent.futures
import numpy as np
import pandas as pd

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    EvaluationResult = None

DATA_PATH = os.environ.get("SDWPF_CSV", "sdwpf_245days_v1.csv")
INPUT_LEN = int(os.environ.get("SDWPF_INPUT_LEN", "144"))
OUTPUT_LEN = int(os.environ.get("SDWPF_OUTPUT_LEN", "288"))
TRAIN_DAYS = int(os.environ.get("SDWPF_TRAIN_DAYS", "214"))
VAL_DAYS = int(os.environ.get("SDWPF_VAL_DAYS", "31"))
DAY_LEN = int(os.environ.get("SDWPF_DAY_LEN", "144"))
N_ORIGINS = int(os.environ.get("SDWPF_N_ORIGINS", "64"))
MAX_TURBINES = os.environ.get("SDWPF_MAX_TURBINES", "")

FEATURE_COLS = ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
TARGET_COL = "Patv"
_IDX = {c: i for i, c in enumerate(FEATURE_COLS)}
_CACHE = None


def _abnormal_mask(block):
    P, W = block[:, _IDX["Patv"]], block[:, _IDX["Wspd"]]
    cond = (P <= 0) & (W > 2.5)
    for c in ["Pab1", "Pab2", "Pab3"]:
        cond |= block[:, _IDX[c]] > 89
    cond |= (block[:, _IDX["Wdir"]] < -180) | (block[:, _IDX["Wdir"]] > 180)
    cond |= (block[:, _IDX["Ndir"]] < -720) | (block[:, _IDX["Ndir"]] > 720)
    return cond


def _build_cache():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values([c for c in ["TurbID", "Day", "Tmstamp"] if c in df.columns]).reset_index(drop=True)
    turbines = list(df["TurbID"].unique())
    if MAX_TURBINES:
        turbines = turbines[: int(MAX_TURBINES)]

    train_end = TRAIN_DAYS * DAY_LEN
    val_end = (TRAIN_DAYS + VAL_DAYS) * DAY_LEN
    per_turbine = []
    origins = None

    for tid in turbines:
        sub = df[df["TurbID"] == tid]
        raw = sub[FEATURE_COLS].to_numpy(dtype=float)
        raw[:, _IDX["Patv"]] = np.clip(raw[:, _IDX["Patv"]], 0, None)
        filled = pd.DataFrame(raw, columns=FEATURE_COLS).ffill().bfill().fillna(0.0).to_numpy()
        n = len(raw)
        if n < val_end:
            continue
        if origins is None:
            lo = max(train_end, INPUT_LEN)
            hi = min(val_end, n) - OUTPUT_LEN
            origins = np.unique(np.linspace(lo, hi, N_ORIGINS).astype(int))

        hw = np.stack([filled[t - INPUT_LEN:t] for t in origins])
        gts = np.stack([raw[t:t + OUTPUT_LEN, _IDX["Patv"]] for t in origins])
        drops = np.stack([_abnormal_mask(raw[t:t + OUTPUT_LEN]) for t in origins])
        valid = (~drops) & np.isfinite(gts)
        train_series = pd.DataFrame(filled[:train_end], columns=FEATURE_COLS)
        pers = np.repeat(hw[:, -1, _IDX["Patv"]][:, None], OUTPUT_LEN, axis=1)
        per_turbine.append({"hw": hw, "gts": gts, "valid": valid, "train": train_series, "pers": pers})

    if not per_turbine:
        raise RuntimeError("No turbine had enough rows for the split")
    return {"turbines": per_turbine, "n": len(per_turbine)}


def _turbine_score(preds, gts, valid):
    maes, rmses = [], []
    for i in range(len(gts)):
        m = valid[i] & np.isfinite(preds[i])
        if m.sum() == 0:
            continue
        p, g = preds[i][m] / 1000.0, gts[i][m] / 1000.0
        maes.append(float(np.mean(np.abs(p - g))))
        rmses.append(float(np.sqrt(np.mean((p - g) ** 2))))
    if not maes:
        return None
    return float(np.mean(maes)), float(np.mean(rmses))


def _run_with_timeout(func, args, timeout):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(func, *args).result(timeout=timeout)


def _result(metrics, artifacts):
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return metrics


def _fail(msg, extra=None):
    art = {"error_message": msg}
    if extra:
        art.update(extra)
    return _result({"combined_score": -1.0, "skill_vs_persistence": -1.0, "error": msg}, art)


def evaluate(program_path):
    global _CACHE
    try:
        if _CACHE is None:
            _CACHE = _build_cache()

        spec = importlib.util.spec_from_file_location("candidate", program_path)
        prog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prog)
        if not hasattr(prog, "forecast"):
            return _fail("Program missing forecast()")

        tot_mae = tot_rmse = tot_mae_p = tot_rmse_p = 0.0
        scored = 0
        for T in _CACHE["turbines"]:
            preds = _run_with_timeout(
                prog.forecast,
                (T["train"], T["hw"], FEATURE_COLS, TARGET_COL, INPUT_LEN, OUTPUT_LEN),
                timeout=float(os.environ.get("SDWPF_FC_TIMEOUT", "120")),
            )
            preds = np.asarray(preds, dtype=float)
            if preds.shape != T["gts"].shape:
                return _fail("Prediction shape mismatch",
                             {"expected": list(T["gts"].shape), "got": list(preds.shape)})
            ms = _turbine_score(preds, T["gts"], T["valid"])
            mp = _turbine_score(T["pers"], T["gts"], T["valid"])
            if ms is None or mp is None:
                continue
            tot_mae += ms[0]; tot_rmse += ms[1]
            tot_mae_p += mp[0]; tot_rmse_p += mp[1]
            scored += 1

        if scored == 0:
            return _fail("No turbine produced a scorable result")

        farm_score = (tot_mae + tot_rmse) / 2.0          # summed over turbines, MW
        pers_score = (tot_mae_p + tot_rmse_p) / 2.0
        skill = 1.0 - (farm_score / pers_score) if pers_score > 0 else 0.0

        metrics = {
            "combined_score": float(skill),
            "skill_vs_persistence": float(skill),
            "farm_score_mw": float(farm_score),       # compare to baseline 42.32
            "farm_rmse_mw": float(tot_rmse),
            "farm_mae_mw": float(tot_mae),
            "turbines_scored": int(scored),
        }
        return _result(metrics, {"input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
                                 "n_origins": int(len(_CACHE["turbines"][0]["gts"]))})
    except concurrent.futures.TimeoutError:
        return _fail("forecast() timed out")
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()[-1200:]})


if __name__ == "__main__":
    import json
    r = evaluate(os.environ.get("SDWPF_PROGRAM", "initial_program.py"))
    print(json.dumps(getattr(r, "metrics", r), indent=2))
