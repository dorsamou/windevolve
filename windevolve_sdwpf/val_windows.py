# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/val_windows.py

Description:
    Builds a VALIDATION window set from the tail of the 245-day training CSV,
    in the exact same on-disk format as the official final_phase_test
    (infile/NNNNin.csv + outfile/NNNNout.csv, turbine-major). This lets both
    harness arms search against held-out data and keep final_phase_test locked
    for a single end-of-project measurement.

    Also writes a TRUNCATED training CSV covering only the days BEFORE the
    validation period. Candidates that fit a model must be pointed at this file
    (via SDWPF_TRAIN_FILE) rather than the full 245 days, otherwise they train
    on their own validation targets.

    Why this exists:
        Scoring candidates on final_phase_test makes that set the search signal.
        The resulting score is no longer held out and is not comparable to the
        KDD Cup leaderboard, where each team got one shot at those windows.

Citation:
    Window layout mirrors the official Baidu KDD Cup 2022 SDWPF evaluation
    bundle (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    python val_windows.py                       # defaults below
    python val_windows.py --val-days 30 --n-windows 40 --seed 1
"""
import os                          # path handling and env-var lookup
import shutil                      # clear a stale output directory
import argparse                    # CLI options
from pathlib import Path           # path arithmetic

import numpy as np                 # RNG for window origins
import pandas as pd                # CSV reading / writing

# ---------------------------------------------------------------------------
# Fixed protocol constants. These mirror the official settings and are NOT
# part of any search space.
# ---------------------------------------------------------------------------
CAPACITY = 134                     # turbines in the farm
STEPS_PER_DAY = 144                # 10-min sampling -> 144 steps/day
INPUT_LEN = 2016                   # official max_input_len: 14 days of history
OUTPUT_LEN = 288                   # official horizon: 48 h

# Defaults chosen so validation is large enough to be stable but small enough
# that a search iteration stays fast.
DEFAULT_VAL_DAYS = 30              # last N days of the 245-day file held out
DEFAULT_N_WINDOWS = 40             # official test set uses 142; 40 keeps evals quick
DEFAULT_SEED = 1                   # fixed: the validation set must never change

_HERE = Path(__file__).resolve().parent

TRAIN_FILE = Path(os.environ.get(
    "SDWPF_TRAIN_FILE_FULL",
    os.environ.get(
        "SDWPF_TRAIN_FILE",
        "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
        "sdwpf_kddcup/sdwpf_245days_v1.csv",
    ),
))

# Everything this script produces lands here.
OUT_DIR = Path(os.environ.get("SDWPF_VAL_DIR", _HERE / "val_split"))
VAL_WINDOWS_DIR = OUT_DIR / "val_windows"      # infile/ + outfile/
VAL_TRAIN_FILE = OUT_DIR / "val_train.csv"     # truncated training CSV


def _load_sorted(path: Path) -> pd.DataFrame:
    """
    Description:
        Read the 245-day CSV and sort it turbine-major, then chronologically,
        matching the row order the official window files use.
    Args:
        path: path to sdwpf_245days_v1.csv
    Returns:
        The sorted DataFrame with a stable RangeIndex.
    """
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")
    print(f"reading {path} ...", flush=True)
    df = pd.read_csv(path)
    for col in ("TurbID", "Day", "Tmstamp"):
        if col not in df.columns:
            raise ValueError(f"Expected column {col!r} in {path.name}; got {list(df.columns)}")
    # Turbine-major, then time order within each turbine.
    df = df.sort_values(["TurbID", "Day", "Tmstamp"], kind="mergesort").reset_index(drop=True)
    return df


def _per_turbine_blocks(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Description:
        Split the sorted frame into one contiguous block per turbine and check
        that every turbine has the same number of rows (the window format
        assumes a rectangular layout).
    Args:
        df: the sorted 245-day frame
    Returns:
        A list of 134 per-turbine frames, ordered by TurbID.
    """
    blocks = [g.reset_index(drop=True) for _, g in df.groupby("TurbID", sort=True)]
    if len(blocks) != CAPACITY:
        raise ValueError(f"Expected {CAPACITY} turbines, found {len(blocks)}")
    lengths = {len(b) for b in blocks}
    if len(lengths) != 1:
        raise ValueError(f"Turbines have unequal row counts: {sorted(lengths)}")
    return blocks


def _write_window(blocks: list[pd.DataFrame], origin: int, wid: int) -> None:
    """
    Description:
        Write one validation window: the INPUT_LEN steps ending at `origin` as
        the infile, and the following OUTPUT_LEN steps as the outfile. Rows are
        stacked turbine-major so test_data.TestData slices them correctly.

        NaNs are preserved. The scorer's masking rules depend on them, so
        filling them here would silently change the metric.
    Args:
        blocks: per-turbine frames
        origin: index of the first forecast step (exclusive end of history)
        wid: 1-based window id, used for the NNNNin.csv / NNNNout.csv names
    Returns:
        None
    """
    in_parts, out_parts = [], []
    for b in blocks:
        in_parts.append(b.iloc[origin - INPUT_LEN:origin])
        out_parts.append(b.iloc[origin:origin + OUTPUT_LEN])

    # metrics.regressor_detailed_scores parses the id as int(name[:-6]) - 1,
    # so the stem must be exactly NNNN + "in"/"out".
    name = f"{wid:04d}"
    pd.concat(in_parts, ignore_index=True).to_csv(
        VAL_WINDOWS_DIR / "infile" / f"{name}in.csv", index=False
    )
    pd.concat(out_parts, ignore_index=True).to_csv(
        VAL_WINDOWS_DIR / "outfile" / f"{name}out.csv", index=False
    )


def build(val_days: int, n_windows: int, seed: int) -> None:
    """
    Description:
        Build the validation window set and the truncated training CSV.

        Forecast targets are drawn only from the final `val_days` of the record.
        History may reach back before that boundary, which is correct: at
        inference time recent history is legitimately available. What must not
        overlap is the FITTED model's training data and the validation targets,
        which is what the truncated CSV enforces.
    Args:
        val_days: number of trailing days reserved for validation targets
        n_windows: how many windows to sample
        seed: RNG seed for the window origins (fixed, so the split is stable)
    Returns:
        None
    """
    df = _load_sorted(TRAIN_FILE)
    blocks = _per_turbine_blocks(df)
    total_steps = len(blocks[0])

    val_steps = val_days * STEPS_PER_DAY
    split = total_steps - val_steps            # first step of the validation period

    if split < INPUT_LEN:
        raise ValueError("Not enough history before the validation split.")

    # Legal origins: forecast fully inside the validation period, and enough
    # history behind it to fill an infile.
    lo = max(split, INPUT_LEN)
    hi = total_steps - OUTPUT_LEN
    if hi <= lo:
        raise ValueError(f"No room for windows (lo={lo}, hi={hi}).")

    rng = np.random.default_rng(seed)
    origins = np.sort(rng.choice(np.arange(lo, hi + 1), size=min(n_windows, hi - lo + 1),
                                 replace=False))

    if VAL_WINDOWS_DIR.exists():
        shutil.rmtree(VAL_WINDOWS_DIR)
    (VAL_WINDOWS_DIR / "infile").mkdir(parents=True)
    (VAL_WINDOWS_DIR / "outfile").mkdir(parents=True)

    print(f"total steps/turbine : {total_steps}  ({total_steps / STEPS_PER_DAY:.0f} days)")
    print(f"validation period   : step {split} onward  (last {val_days} days)")
    print(f"writing {len(origins)} windows ...", flush=True)
    for wid, origin in enumerate(origins, start=1):
        _write_window(blocks, int(origin), wid)
        if wid % 10 == 0:
            print(f"  {wid}/{len(origins)}", flush=True)

    # Truncated training CSV: everything strictly before the validation period.
    print(f"writing {VAL_TRAIN_FILE.name} ...", flush=True)
    pd.concat([b.iloc[:split] for b in blocks], ignore_index=True).to_csv(
        VAL_TRAIN_FILE, index=False
    )

    print("\ndone. point the search at the validation split with:")
    print(f'  export SDWPF_TEST_DIR="{VAL_WINDOWS_DIR}"')
    print(f'  export SDWPF_TRAIN_FILE="{VAL_TRAIN_FILE}"')
    print("\nleave final_phase_test untouched until the end of the project.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3].strip())
    ap.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    ap.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build(args.val_days, args.n_windows, args.seed)
