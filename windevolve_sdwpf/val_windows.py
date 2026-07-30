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
    python val_windows.py
    python val_windows.py --val-days 30 --n-windows 40 --seed 1
    python val_windows.py --train-file /path/to/sdwpf_245days_v1.csv
"""
import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed protocol constants. These mirror the official settings and are NOT
# part of any search space.
# ---------------------------------------------------------------------------
CAPACITY = 134                     # turbines in the farm
STEPS_PER_DAY = 144                # 10-min sampling -> 144 steps/day
INPUT_LEN = 2016                   # official max_input_len: 14 days of history
OUTPUT_LEN = 288                   # official horizon: 48 h

DEFAULT_VAL_DAYS = 30              # last N days of the record held out
DEFAULT_N_WINDOWS = 40             # official test set uses 142; 40 keeps evals quick
DEFAULT_SEED = 1                   # fixed: the validation set must never change

_HERE = Path(__file__).resolve().parent

# Env vars checked for the 245-day CSV, in order. Kept flat rather than as
# nested os.environ.get() calls -- the nested form is what breaks when this
# file is copy-pasted and a line wraps.
_TRAIN_ENV_VARS = ("SDWPF_TRAIN_FILE_FULL", "SDWPF_TRAIN_FILE")

# Conventional locations tried if no env var is set.
_TRAIN_FALLBACKS = (
    _HERE.parent / "wind_forecasting_data" / "sdwpf_kddcup" / "sdwpf_245days_v1.csv",
    _HERE.parent / "wind_forecasting_data" / "SDWPF_dataset" / "sdwpf_kddcup"
    / "sdwpf_245days_v1.csv",
    Path.home() / "srip" / "wind_forecasting_data" / "sdwpf_kddcup"
    / "sdwpf_245days_v1.csv",
)


def resolve_train_file(explicit=None) -> Path:
    """
    Description:
        Locate the 245-day training CSV. Precedence: an explicit --train-file
        argument, then SDWPF_TRAIN_FILE_FULL, then SDWPF_TRAIN_FILE, then a
        short list of conventional locations relative to the repo and home dir.
    Args:
        explicit: path passed on the command line, or None
    Returns:
        A Path to the CSV. Existence is checked by the caller so the error can
        list everything that was tried.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    for var in _TRAIN_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return Path(value).expanduser().resolve()
    for candidate in _TRAIN_FALLBACKS:
        if candidate.exists():
            return candidate.resolve()
    # Nothing found. Return the first fallback so the error names a real path.
    return _TRAIN_FALLBACKS[0]


def resolve_out_dir() -> Path:
    """
    Description:
        Where everything this script produces lands.
    Args:
        None
    Returns:
        A Path, from SDWPF_VAL_DIR if set, else <this dir>/val_split.
    """
    value = os.environ.get("SDWPF_VAL_DIR")
    if value:
        return Path(value).expanduser().resolve()
    return _HERE / "val_split"


def _load_sorted(path) -> pd.DataFrame:
    """
    Description:
        Read the 245-day CSV and sort it turbine-major, then chronologically,
        matching the row order the official window files use.
    Args:
        path: path to sdwpf_245days_v1.csv (str or Path; coerced either way)
    Returns:
        The sorted DataFrame with a stable RangeIndex.
    """
    path = Path(path)              # accept str or Path -- never assume
    if not path.exists():
        tried = "\n  ".join(str(p) for p in _TRAIN_FALLBACKS)
        raise FileNotFoundError(
            f"Training CSV not found: {path}\n\n"
            f"Set it explicitly with either:\n"
            f"  export SDWPF_TRAIN_FILE_FULL=/path/to/sdwpf_245days_v1.csv\n"
            f"  python val_windows.py --train-file /path/to/sdwpf_245days_v1.csv\n\n"
            f"Conventional locations checked:\n  {tried}"
        )
    print(f"reading {path} ...", flush=True)
    df = pd.read_csv(path)
    for col in ("TurbID", "Day", "Tmstamp"):
        if col not in df.columns:
            raise ValueError(
                f"Expected column {col!r} in {path.name}; got {list(df.columns)}"
            )
    # Turbine-major, then time order within each turbine.
    df = df.sort_values(["TurbID", "Day", "Tmstamp"], kind="mergesort")
    return df.reset_index(drop=True)


def _per_turbine_blocks(df: pd.DataFrame) -> list:
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
        raise ValueError(
            f"Turbines have unequal row counts: {sorted(lengths)}. "
            "The window format requires a rectangular layout."
        )
    return blocks


def _write_window(blocks: list, origin: int, wid: int, windows_dir: Path) -> None:
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
        windows_dir: directory holding infile/ and outfile/
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
        windows_dir / "infile" / f"{name}in.csv", index=False
    )
    pd.concat(out_parts, ignore_index=True).to_csv(
        windows_dir / "outfile" / f"{name}out.csv", index=False
    )


def build(train_file, out_dir, val_days: int, n_windows: int, seed: int) -> None:
    """
    Description:
        Build the validation window set and the truncated training CSV.

        Forecast targets are drawn only from the final `val_days` of the record.
        History may reach back before that boundary, which is correct: at
        inference time recent history is legitimately available. What must not
        overlap is the FITTED model's training data and the validation targets,
        which is what the truncated CSV enforces.
    Args:
        train_file: path to the 245-day CSV
        out_dir: where to write val_windows/ and val_train.csv
        val_days: number of trailing days reserved for validation targets
        n_windows: how many windows to sample
        seed: RNG seed for the window origins (fixed, so the split is stable)
    Returns:
        None
    """
    out_dir = Path(out_dir)
    windows_dir = out_dir / "val_windows"
    val_train_file = out_dir / "val_train.csv"

    df = _load_sorted(train_file)
    blocks = _per_turbine_blocks(df)
    total_steps = len(blocks[0])

    val_steps = val_days * STEPS_PER_DAY
    split = total_steps - val_steps            # first step of the validation period

    if split < INPUT_LEN:
        raise ValueError(
            f"Not enough history before the split (split={split}, "
            f"need >= {INPUT_LEN}). Reduce --val-days."
        )

    # Legal origins: forecast fully inside the validation period, and enough
    # history behind it to fill an infile.
    lo = max(split, INPUT_LEN)
    hi = total_steps - OUTPUT_LEN
    if hi <= lo:
        raise ValueError(f"No room for windows (lo={lo}, hi={hi}).")

    rng = np.random.default_rng(seed)
    n_avail = hi - lo + 1
    if n_windows > n_avail:
        print(f"note: only {n_avail} origins available; using all of them")
    origins = np.sort(
        rng.choice(np.arange(lo, hi + 1), size=min(n_windows, n_avail), replace=False)
    )

    if windows_dir.exists():
        shutil.rmtree(windows_dir)
    (windows_dir / "infile").mkdir(parents=True)
    (windows_dir / "outfile").mkdir(parents=True)

    first_val_day = split // STEPS_PER_DAY + 1
    last_day = total_steps // STEPS_PER_DAY
    print(f"total steps/turbine : {total_steps:,}  "
          f"({total_steps / STEPS_PER_DAY:.0f} days)")
    print(f"validation period   : step {split:,} onward  (last {val_days} days)")
    print(f"train = days 1..{split // STEPS_PER_DAY}, "
          f"val targets = days {first_val_day}..{last_day}")
    print(f"writing {len(origins)} windows to {windows_dir} ...", flush=True)
    for wid, origin in enumerate(origins, start=1):
        _write_window(blocks, int(origin), wid, windows_dir)
        if wid % 10 == 0:
            print(f"  {wid}/{len(origins)}", flush=True)

    # Truncated training CSV: everything strictly before the validation period.
    print(f"writing {val_train_file.name} ...", flush=True)
    pd.concat([b.iloc[:split] for b in blocks], ignore_index=True).to_csv(
        val_train_file, index=False
    )

    print("\ndone. point the search at the validation split with:")
    print(f'  export SDWPF_TEST_DIR="{windows_dir}"')
    print(f'  export SDWPF_TRAIN_FILE="{val_train_file}"')
    print("\nleave final_phase_test untouched until the end of the project.")


def main() -> None:
    """
    Description:
        Parse arguments, resolve paths, and build the split.
    Args:
        None
    Returns:
        None
    """
    ap = argparse.ArgumentParser(
        description="Build a held-out validation window set from the 245-day CSV."
    )
    ap.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS)
    ap.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train-file", default=None,
                    help="path to sdwpf_245days_v1.csv (overrides env vars)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write val_windows/ and val_train.csv")
    args = ap.parse_args()

    train_file = resolve_train_file(args.train_file)
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = resolve_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"train file : {train_file}")
    print(f"out dir    : {out_dir}\n")
    build(train_file, out_dir, args.val_days, args.n_windows, args.seed)


if __name__ == "__main__":
    main()
