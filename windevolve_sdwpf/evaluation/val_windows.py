# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/val_windows.py

Description:
    Builds the held-out VALIDATION split that the evolve loop scores against, so
    that final_phase_test stays locked for a single end-of-project measurement.

    Given the 245-day SDWPF training CSV, this writes:

        val_split/val_train.csv            training rows STRICTLY BEFORE the
                                           validation period -- the only file a
                                           candidate may fit on
        val_split/val_windows/infile/NNNNin.csv    14 days of history per window
        val_split/val_windows/outfile/NNNNout.csv  the next 48 hours, ground truth
        val_split/manifest.json            exact step indices of every window

    The on-disk format matches the official competition layout exactly: all 134
    turbines stacked TURBINE-MAJOR in one CSV per window, the original 13 columns
    in the original order, and NaNs PRESERVED. The last point is not cosmetic --
    official_metrics.turbine_scores masks a timestep if ANY column of that row is
    NaN, so filling missing values here would silently change which timesteps get
    scored and make our numbers incomparable to the competition's.

    Anti-leakage rule, enforced by construction: val_train.csv is truncated at
    the FIRST target step of the EARLIEST window. No validation target ever
    appears in the training file. Window input history may reach back into the
    training period -- that is legitimate, it is data the forecaster would have
    at prediction time -- but no target does.

Citation:
    On-disk format and masking follow the official Baidu KDD Cup 2022 SDWPF
    evaluator (PaddlePaddle/PaddleSpatial,
    apps/wpf_baseline_gru/kddcup22-sdwpf-evaluation).
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # inspect the plan without writing anything (fast, do this first):
    python val_windows.py --train-file /data/dorsa/srip/wind_forecasting_data/\\
sdwpf_kddcup/wtbdata_245days.csv --dry-run

    # build the split:
    python val_windows.py --train-file <245-day csv> --n-windows 48

    # then point the evaluator at it (these are its defaults already):
    python evaluator.py initial_program.py
"""
import os                          # paths, dirs, free-space check
import sys                         # exit codes
import json                        # manifest
import shutil                      # disk usage
import argparse                    # CLI

import numpy as np                 # index maths
import pandas as pd                # CSV read/write

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths import SPLITS_DIR, TRAIN_FILE as DEFAULT_TRAIN_FILE

CAPACITY = 134                     # turbines
STEPS_PER_DAY = 144                # 10-minute steps per day
INPUT_LEN = 2016                   # 14 days of history per window (official)
OUTPUT_LEN = 288                   # 48 hour horizon (official)


# The official column order. The first three are labels (start_col=3 in
# test_data.py); everything from Wspd on is modeled, and Patv is the target.
COLUMNS = ["TurbID", "Day", "Tmstamp", "Wspd", "Wdir", "Etmp",
           "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv", "Patv"]
LABEL_COLUMNS = ["TurbID", "Day", "Tmstamp"]


def _tmstamp_to_slot(series):
    """Map an 'HH:MM' string column to its 0-based 10-minute slot in the day."""
    parts = series.astype(str).str.split(":", n=1, expand=True)
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    return (hours * 6 + minutes // 10).astype("Int64")


def load_train_csv(path):
    """
    Description:
        Read the 245-day CSV and return it sorted turbine-major, with each
        turbine's rows in chronological order. Validates the schema and that
        every turbine has the same number of steps.
    Args:
        path: path to the 245-day training CSV
    Returns:
        (df, n_turbines, steps_per_turbine)
    """
    sys.stderr.write("reading {} ...\n".format(path))
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "input CSV is missing expected column(s) {}. Found: {}".format(
                missing, list(df.columns)))
    df = df[COLUMNS]                              # drop extras, fix the order

    # Labels must be clean; a NaN here would corrupt the turbine-major layout.
    for col in ("TurbID", "Day"):
        if df[col].isna().any():
            raise ValueError("column {!r} contains NaN; cannot build the split".format(col))
        df[col] = df[col].astype(np.int64)

    slot = _tmstamp_to_slot(df["Tmstamp"])
    if slot.isna().any():
        raise ValueError("Tmstamp values that do not parse as HH:MM are present")

    order = np.lexsort((slot.to_numpy(dtype=np.int64),
                        df["Day"].to_numpy(),
                        df["TurbID"].to_numpy()))
    df = df.iloc[order].reset_index(drop=True)

    counts = df["TurbID"].value_counts()
    n_turbines = int(counts.size)
    if counts.nunique() != 1:
        raise ValueError(
            "turbines have differing row counts (min {}, max {}); the official "
            "turbine-major layout requires equal blocks".format(counts.min(), counts.max()))
    steps = int(counts.iloc[0])

    sys.stderr.write("  {} turbines x {} steps ({:.1f} days) = {} rows\n".format(
        n_turbines, steps, steps / STEPS_PER_DAY, len(df)))
    return df, n_turbines, steps


def plan_windows(steps, val_days, n_windows, input_len, output_len):
    """
    Description:
        Choose window start steps, evenly spaced (deterministic, no RNG) across
        the eligible range. A window's TARGET must fall entirely inside the last
        `val_days` days; its input history may reach back before that.
    Args:
        steps: steps per turbine in the full series
        val_days: size of the validation period, in days
        n_windows: how many windows to cut
        input_len / output_len: window geometry, in steps
    Returns:
        (starts, val_start_step) -- starts is a sorted list of target start steps
    """
    val_start = steps - val_days * STEPS_PER_DAY
    if val_start < input_len:
        raise ValueError(
            "series too short: need at least {} steps of history before the "
            "validation period, have {}".format(input_len, val_start))

    first, last = val_start, steps - output_len
    if last < first:
        raise ValueError("validation period shorter than one forecast horizon")

    span = last - first
    if n_windows == 1:
        starts = [first]
    else:
        starts = [int(round(first + span * i / (n_windows - 1)))
                  for i in range(n_windows)]
    starts = sorted(set(starts))
    if len(starts) < n_windows:
        sys.stderr.write(
            "  NOTE: only {} distinct starts fit in the validation period "
            "(asked for {})\n".format(len(starts), n_windows))
    return starts, val_start


def _slice_turbine_major(df, n_turbines, steps, lo, hi):
    """Concatenate rows [lo, hi) of every turbine, preserving turbine-major order."""
    idx = np.concatenate([np.arange(t * steps + lo, t * steps + hi)
                          for t in range(n_turbines)])
    return df.iloc[idx]


def _write_csv(frame, path):
    """Write one window/train CSV with ints kept as ints and NaNs preserved."""
    out = frame.copy()
    out["TurbID"] = out["TurbID"].astype(np.int64)
    out["Day"] = out["Day"].astype(np.int64)
    # na_rep="" is pandas' default and round-trips back to NaN on read; stated
    # explicitly because the masking rules depend on NaNs surviving the trip.
    out.to_csv(path, index=False, na_rep="")


def build(args):
    """Cut the split and write it to disk."""
    df, n_turbines, steps = load_train_csv(args.train_file)
    if n_turbines != args.capacity:
        sys.stderr.write("  WARNING: found {} turbines, expected {}\n".format(
            n_turbines, args.capacity))

    starts, val_start = plan_windows(
        steps, args.val_days, args.n_windows, args.input_len, args.output_len)

    est_gb = (len(starts) * (args.input_len + args.output_len) * n_turbines
              + val_start * n_turbines) * 150 / 1e9      # ~150 bytes per row
    free_gb = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out_dir)) or ".").free / 1e9

    print("")
    print("PLAN")
    print("  series             : {} steps ({:.1f} days) x {} turbines".format(
        steps, steps / STEPS_PER_DAY, n_turbines))
    print("  validation period  : last {} days -> starts at step {} (day {})".format(
        args.val_days, val_start, val_start // STEPS_PER_DAY + 1))
    print("  windows            : {}  (input {} steps / output {} steps)".format(
        len(starts), args.input_len, args.output_len))
    print("  first / last start : {} / {}".format(starts[0], starts[-1]))
    print("  val_train.csv      : steps [0, {}) -> {} rows/turbine ({:.1f} days)".format(
        val_start, val_start, val_start / STEPS_PER_DAY))
    print("  earliest history   : step {} (window 1 reads back into train, which is fine)".format(
        starts[0] - args.input_len))
    print("  est. disk needed   : ~{:.1f} GB   (free: {:.1f} GB)".format(est_gb, free_gb))
    print("")

    if est_gb > free_gb * 0.9:
        sys.stderr.write("REFUSING: not enough free disk. Reduce --n-windows.\n")
        return 1
    if args.dry_run:
        print("--dry-run: nothing written.")
        return 0

    win_dir = os.path.join(args.out_dir, "val_windows")
    in_dir = os.path.join(win_dir, "infile")
    out_dir = os.path.join(win_dir, "outfile")
    for d in (in_dir, out_dir):
        os.makedirs(d, exist_ok=True)

    train_path = os.path.join(args.out_dir, "val_train.csv")
    sys.stderr.write("writing {} ...\n".format(train_path))
    _write_csv(_slice_turbine_major(df, n_turbines, steps, 0, val_start), train_path)

    manifest = {
        "source_csv": os.path.abspath(args.train_file),
        "n_turbines": n_turbines,
        "steps_per_turbine": steps,
        "steps_per_day": STEPS_PER_DAY,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "val_days": args.val_days,
        "val_start_step": val_start,
        "train_rows_per_turbine": val_start,
        "windows": [],
    }

    for i, s in enumerate(starts, start=1):
        in_path = os.path.join(in_dir, "{:04d}in.csv".format(i))
        out_path = os.path.join(out_dir, "{:04d}out.csv".format(i))

        _write_csv(_slice_turbine_major(
            df, n_turbines, steps, s - args.input_len, s), in_path)
        _write_csv(_slice_turbine_major(
            df, n_turbines, steps, s, s + args.output_len), out_path)

        manifest["windows"].append({
            "index": i,
            "infile": os.path.basename(in_path),
            "outfile": os.path.basename(out_path),
            "history_steps": [s - args.input_len, s],
            "target_steps": [s, s + args.output_len],
            "target_day": s // STEPS_PER_DAY + 1,
        })
        sys.stderr.write("  window {:>4}/{}  target step {}\n".format(i, len(starts), s))

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\nwrote {} windows to {}".format(len(starts), win_dir))
    print("verify with: python val_windows.py --verify --out-dir {}".format(args.out_dir))
    return 0


def verify(args):
    """
    Description:
        Re-read the written split and check the properties the evaluator and the
        official grader depend on: shape, turbine-major layout, NaN survival,
        and -- the important one -- that no validation target step appears in
        val_train.csv.
    Args:
        args: parsed CLI namespace
    Returns:
        0 on success, 1 on any failure
    """
    man_path = os.path.join(args.out_dir, "manifest.json")
    if not os.path.isfile(man_path):
        sys.stderr.write("no manifest at {}\n".format(man_path))
        return 1
    with open(man_path) as fh:
        man = json.load(fh)

    win_dir = os.path.join(args.out_dir, "val_windows")
    fails = []

    train = pd.read_csv(os.path.join(args.out_dir, "val_train.csv"), low_memory=False)
    exp_train = man["train_rows_per_turbine"] * man["n_turbines"]
    if len(train) != exp_train:
        fails.append("val_train.csv has {} rows, expected {}".format(len(train), exp_train))
    if list(train.columns) != COLUMNS:
        fails.append("val_train.csv column order differs from the official order")

    # The anti-leakage check: the last training step must precede every target.
    train_last_step = man["train_rows_per_turbine"] - 1
    earliest_target = min(w["target_steps"][0] for w in man["windows"])
    if train_last_step >= earliest_target:
        fails.append("LEAKAGE: val_train reaches step {} but the earliest target "
                     "starts at {}".format(train_last_step, earliest_target))
    else:
        print("leakage check     : OK (train ends at step {}, first target at {})".format(
            train_last_step, earliest_target))

    n_check = len(man["windows"]) if args.verify_all else min(3, len(man["windows"]))
    for w in man["windows"][:n_check]:
        ip = os.path.join(win_dir, "infile", w["infile"])
        op = os.path.join(win_dir, "outfile", w["outfile"])
        din = pd.read_csv(ip, low_memory=False)
        dout = pd.read_csv(op, low_memory=False)

        if len(din) != man["input_len"] * man["n_turbines"]:
            fails.append("{}: {} rows, expected {}".format(
                w["infile"], len(din), man["input_len"] * man["n_turbines"]))
        if len(dout) != man["output_len"] * man["n_turbines"]:
            fails.append("{}: {} rows, expected {}".format(
                w["outfile"], len(dout), man["output_len"] * man["n_turbines"]))

        # Turbine-major: each block must hold exactly one TurbID.
        blk = man["input_len"]
        ids = din["TurbID"].to_numpy()[: blk * 3].reshape(3, blk)
        if not all(len(np.unique(r)) == 1 for r in ids):
            fails.append("{}: not turbine-major".format(w["infile"]))

        if not dout.isna().any().any():
            fails.append("{}: contains no NaNs at all -- suspicious, the masking "
                         "rules expect missing readings to survive".format(w["outfile"]))

    if fails:
        print("\nFAILED:")
        for f in fails:
            print("  - {}".format(f))
        return 1

    print("windows checked   : {}/{}".format(n_check, len(man["windows"])))
    print("shapes, layout    : OK")
    print("NaNs preserved    : OK")
    print("\nsplit looks good.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Build the SDWPF validation split.")
    ap.add_argument("--train-file", default=DEFAULT_TRAIN_FILE,
                    help="the 245-day SDWPF training CSV (default: paths.TRAIN_FILE)")
    ap.add_argument("--out-dir", default=SPLITS_DIR)    
    ap.add_argument("--n-windows", type=int, default=24,
                    help="number of validation windows (default 24)")
    ap.add_argument("--val-days", type=int, default=48,
                    help="size of the validation period in days (default 48)")
    ap.add_argument("--input-len", type=int, default=INPUT_LEN)
    ap.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    ap.add_argument("--capacity", type=int, default=CAPACITY)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and disk estimate, write nothing")
    ap.add_argument("--verify", action="store_true",
                    help="check an existing split instead of building one")
    ap.add_argument("--verify-all", action="store_true",
                    help="with --verify, read every window rather than the first 3")
    args = ap.parse_args()

    if args.verify:
        return verify(args)
    if not args.train_file:
        ap.error("--train-file is required unless --verify is given")
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
