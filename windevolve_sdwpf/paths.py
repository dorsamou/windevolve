# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/paths.py

Description:
    Single source of truth for every filesystem location in the project.
    No other file should hardcode a path, walk the directory tree looking
    for a marker file, or build a path from "..".

    Repo-internal locations are derived from this file's own position, so
    they are correct on any machine without configuration. The only thing
    that varies per machine is where the raw SDWPF data lives, and that is
    a single environment variable.

Machine setup (once per machine, in ~/.bashrc):

    export SDWPF_DATA_DIR=/data/dorsa/srip/wind_forecasting_data/sdwpf_kddcup

    If unset, DATA_DIR falls back to a sibling of the repo:
        <parent of repo>/wind_forecasting_data/sdwpf_kddcup
    which is correct for the current server layout.

Usage:
    # from a file at the repo root, or already on sys.path:
    from paths import VAL_DIR, VAL_TRAIN_FILE, RUNS_DIR

    # from a file in a subdirectory (baselines/, arms/, analysis/):
    import os, sys
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from paths import DATA_DIR

Every constant below is overridable by the environment variable named
beside it, so existing scripts and shell exports keep working unchanged.
"""

import os

# ---------------------------------------------------------------------------
# Repo layout. Derived from this file's location -- never configured.
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

EVAL_DIR      = os.path.join(REPO_ROOT, "evaluation")
ARMS_DIR      = os.path.join(REPO_ROOT, "arms")
BASELINES_DIR = os.path.join(REPO_ROOT, "baselines")
ANALYSIS_DIR  = os.path.join(REPO_ROOT, "analysis")
ARCHIVE_DIR   = os.path.join(REPO_ROOT, "archive")

# ---------------------------------------------------------------------------
# Raw SDWPF data. Lives OUTSIDE the repo (large, gitignored, sha256-verified).
# This is the one location that differs between machines.
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get(
    "SDWPF_DATA_DIR",
    os.path.join(os.path.dirname(REPO_ROOT), "wind_forecasting_data", "sdwpf_kddcup"))

# The 245-day training CSV -- the full dataset, including the validation tail.
# Only val_windows.py should read this. Anything that fits a model during
# search must use VAL_TRAIN_FILE instead, or it trains on its own targets.
TRAIN_FILE = os.environ.get(
    "SDWPF_TRAIN_FILE", os.path.join(DATA_DIR, "sdwpf_245days_v1.csv"))

LOCATION_FILE = os.environ.get(
    "SDWPF_LOCATION_FILE",
    os.path.join(DATA_DIR, "sdwpf_baidukddcup2022_turb_location.csv"))

# The competition's held-out test windows. LOCKED: read only by
# competition_score.py, for a single end-of-project measurement.
# evaluator.py must have no path to this.
TEST_DIR = os.environ.get(
    "SDWPF_TEST_DIR", os.path.join(DATA_DIR, "final_phase_test"))

# ---------------------------------------------------------------------------
# Generated validation split (written by evaluation/val_windows.py).
# Derived, not raw: reproducible from TRAIN_FILE plus the split arguments.
# Gitignored. Lives beside the scorer because the split is part of what
# defines a score -- if it changes, every number becomes incomparable.
# ---------------------------------------------------------------------------
SPLITS_DIR = os.environ.get(
    "SDWPF_SPLITS_DIR", os.path.join(EVAL_DIR, "splits"))

VAL_DIR = os.environ.get(
    "SDWPF_VAL_WINDOWS", os.path.join(SPLITS_DIR, "val_windows"))

VAL_TRAIN_FILE = os.environ.get(
    "SDWPF_VAL_TRAIN", os.path.join(SPLITS_DIR, "val_train.csv"))

SPLIT_MANIFEST = os.path.join(SPLITS_DIR, "manifest.json")

# ---------------------------------------------------------------------------
# Run outputs. One subdirectory per execution, any arm or baseline.
# ---------------------------------------------------------------------------
RUNS_DIR = os.environ.get("SDWPF_RUNS_DIR", os.path.join(REPO_ROOT, "runs"))


# ---------------------------------------------------------------------------
# Diagnostics: `python paths.py` prints every location and whether it exists.
# ---------------------------------------------------------------------------
def _report():
    groups = [
        ("repo layout", [
            ("REPO_ROOT", REPO_ROOT), ("EVAL_DIR", EVAL_DIR),
            ("ARMS_DIR", ARMS_DIR), ("BASELINES_DIR", BASELINES_DIR),
            ("ANALYSIS_DIR", ANALYSIS_DIR), ("ARCHIVE_DIR", ARCHIVE_DIR),
        ]),
        ("raw data (machine-specific)", [
            ("DATA_DIR", DATA_DIR), ("TRAIN_FILE", TRAIN_FILE),
            ("LOCATION_FILE", LOCATION_FILE), ("TEST_DIR", TEST_DIR),
        ]),
        ("generated split", [
            ("SPLITS_DIR", SPLITS_DIR), ("VAL_DIR", VAL_DIR),
            ("VAL_TRAIN_FILE", VAL_TRAIN_FILE),
            ("SPLIT_MANIFEST", SPLIT_MANIFEST),
        ]),
        ("outputs", [("RUNS_DIR", RUNS_DIR)]),
    ]
    missing = 0
    for title, items in groups:
        print("\n{}".format(title))
        print("-" * len(title))
        for name, value in items:
            ok = os.path.exists(value)
            if not ok:
                missing += 1
            print("  [{}] {:<16} {}".format("ok" if ok else "MISSING", name, value))

    print("\nSDWPF_DATA_DIR is {}".format(
        "set: " + os.environ["SDWPF_DATA_DIR"] if "SDWPF_DATA_DIR" in os.environ
        else "not set (using the derived default above)"))
    if missing:
        print("\n{} path(s) missing. That is expected before the split is "
              "generated; it is a problem if the raw data paths are missing."
              .format(missing))


if __name__ == "__main__":
    _report()