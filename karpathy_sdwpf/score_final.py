# -*- coding: utf-8 -*-
"""
path: karpathy_sdwpf/score_final.py

Description:
    Scores a champion program ONCE on the locked official test set
    (final_phase_test, 142 windows) and reports the generalisation gap against
    its validation score.

    This is not part of the search loop. The agent never runs it. It exists so
    the final measurement is a deliberate, logged, one-time act rather than
    something that quietly becomes another selection signal.

    Requires --confirm, so an accidental invocation cannot happen.

    After running this, stop. Tuning further and re-measuring means the number is
    no longer held out, and the comparison to the KDD Cup leaderboard is lost.

Example usage:
    python score_final.py --confirm
    python score_final.py --confirm --program forecaster.py --val-score 52.31
"""
import os                          # env vars must be set before importing evaluator
import sys                         # exit codes and import path
import time                        # timestamp the measurement
import json                        # append a machine-readable record
import argparse                    # CLI options
import subprocess                  # capture the git commit of the champion
from pathlib import Path           # path arithmetic

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_HARNESS = _REPO / "windevolve_sdwpf"

# The locked test set. Deliberately read from its own env var so it can never be
# confused with SDWPF_TEST_DIR, which the search loop points at the val split.
FINAL_TEST_DIR = os.environ.get(
    "SDWPF_FINAL_TEST_DIR",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/final_phase_test",
)

# The FULL 245-day training file. A final-measurement champion may legitimately
# refit on all available history, matching what a competition entrant would do.
FULL_TRAIN_FILE = os.environ.get(
    "SDWPF_TRAIN_FILE_FULL",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/sdwpf_245days_v1.csv",
)

os.environ["SDWPF_TEST_DIR"] = FINAL_TEST_DIR
os.environ["SDWPF_TRAIN_FILE"] = FULL_TRAIN_FILE

if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

import evaluator                   # noqa: E402  -- shared scorer, do not modify

RECORD_FILE = _HERE / "final_measurements.jsonl"


def _git_commit() -> str:
    """
    Description:
        Capture the current git commit so the measurement is tied to an exact
        version of the champion program.
    Args:
        None
    Returns:
        The short commit hash, or "unknown" if git is unavailable.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_HERE, capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 -- a missing commit must not block the measurement
        return "unknown"


def main() -> int:
    """
    Description:
        Score the champion on the locked test set, print the result and the
        generalisation gap, and append a record to final_measurements.jsonl.
    Args:
        None (reads flags from argv)
    Returns:
        Process exit code.
    """
    ap = argparse.ArgumentParser(description="One-shot locked-test measurement.")
    ap.add_argument("--confirm", action="store_true",
                    help="required; acknowledges this is the one-time measurement")
    ap.add_argument("--program", default="forecaster.py",
                    help="champion program to score")
    ap.add_argument("--val-score", type=float, default=None,
                    help="its validation score, to report the generalisation gap")
    ap.add_argument("--note", default="",
                    help="free-text note recorded alongside the result")
    args = ap.parse_args()

    if not args.confirm:
        print("Refusing to run without --confirm.\n")
        print("This scores the LOCKED test set (final_phase_test). It is intended")
        print("to happen once per arm, on a champion already selected using the")
        print("validation split. Re-running it after further tuning means the")
        print("number is no longer held out.\n")
        print("If that is what you intend:  python score_final.py --confirm")
        return 2

    program = (_HERE / args.program).resolve()
    if not program.exists():
        print(f"error: no such program: {program}", file=sys.stderr)
        return 1

    if not Path(FINAL_TEST_DIR).exists():
        print(f"error: locked test set not found: {FINAL_TEST_DIR}", file=sys.stderr)
        print("set SDWPF_FINAL_TEST_DIR", file=sys.stderr)
        return 1

    if RECORD_FILE.exists():
        n_prior = sum(1 for _ in RECORD_FILE.open())
        print(f"NOTE: {n_prior} prior measurement(s) already recorded in "
              f"{RECORD_FILE.name}.")
        print("      Each additional run erodes the held-out status of this set.\n")

    print(f"program   : {program.name}")
    print(f"test set  : {FINAL_TEST_DIR}")
    print(f"train file: {FULL_TRAIN_FILE}")
    print("scoring ...\n", flush=True)

    t0 = time.perf_counter()
    result = evaluator.evaluate(str(program))
    elapsed = time.perf_counter() - t0

    if not result.get("valid", False):
        print(f"FAILED: {result.get('reason', 'unknown')}")
        return 1

    final = result["competition_score"]
    print(f"final_score : {final:.4f}   (official metric, lower is better)")
    print(f"rmse        : {result['rmse']:.4f}")
    print(f"mae         : {result['mae']:.4f}")
    print(f"n_windows   : {result['n_windows']}")
    print(f"elapsed     : {elapsed:.1f}s")

    gap = None
    if args.val_score is not None:
        gap = args.val_score - final
        print(f"\nval_score   : {args.val_score:.4f}")
        print(f"gap         : {gap:+.4f}   (validation minus final)")
        print("\nA non-zero gap is expected. Report it as a result rather than")
        print("hiding it; comparing the gap across harness arms is informative.")

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arm": "karpathy_sdwpf",
        "program": program.name,
        "git_commit": _git_commit(),
        "final_score": final,
        "rmse": result["rmse"],
        "mae": result["mae"],
        "n_windows": result["n_windows"],
        "val_score": args.val_score,
        "generalisation_gap": gap,
        "note": args.note,
    }
    with RECORD_FILE.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"\nrecorded in {RECORD_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
