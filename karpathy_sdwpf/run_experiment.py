# -*- coding: utf-8 -*-
"""
path: karpathy_sdwpf/run_experiment.py

Description:
    Runs ONE experiment: scores forecaster.py over the VALIDATION windows and
    prints a single structured RESULT line for the agent to parse.

    Deliberately thin. All scoring is delegated to
    ../windevolve_sdwpf/evaluator.py so both harness arms share one scoring path
    byte-for-byte. If this file grew its own metric the two arms would stop being
    comparable, which is the whole point of the study.

    The validation split is used, never final_phase_test, so the search signal
    stays held out. The locked test set is scored once at the end by
    score_final.py, which the agent does not run.

Example usage:
    python run_experiment.py                        # all validation windows
    python run_experiment.py --n-windows 3          # quick smoke test
"""
import os                          # env vars must be set before importing evaluator
import sys                         # exit codes and import path
import time                        # wall-clock budget accounting
import argparse                    # CLI options
from pathlib import Path           # path arithmetic

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_HARNESS = _REPO / "windevolve_sdwpf"        # shared harness package

# ---------------------------------------------------------------------------
# Point the shared evaluator at the VALIDATION split BEFORE importing it.
# evaluator.py resolves these at import time, so the ordering here matters.
# ---------------------------------------------------------------------------
_VAL = Path(os.environ.get("SDWPF_VAL_DIR", _HARNESS / "val_split"))
os.environ["SDWPF_TEST_DIR"] = str(_VAL / "val_windows")
os.environ["SDWPF_TRAIN_FILE"] = str(_VAL / "val_train.csv")

if str(_HARNESS) not in sys.path:
    sys.path.insert(0, str(_HARNESS))

import evaluator                   # noqa: E402  -- shared scorer, do not modify

# Total wall-clock budget for one experiment, in seconds. Every experiment gets
# the same budget; this is the fairness constraint that makes runs comparable.
# TODO(dors): set from how long the GRU baseline costs end to end.
TIME_BUDGET_S = int(os.environ.get("TIME_BUDGET_S", 600))

FORECASTER = _HERE / "forecaster.py"


def main() -> int:
    """
    Description:
        Score forecaster.py on the validation windows and print the RESULT line.
    Args:
        None (reads --n-windows from argv)
    Returns:
        Process exit code: 0 on a valid score, 1 otherwise.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=None,
                    help="cap windows scored; default is every validation window")
    args = ap.parse_args()

    if not FORECASTER.exists():
        print(f"RESULT valid=0 reason=\"missing forecaster.py at {FORECASTER}\"")
        return 1

    val_dir = Path(os.environ["SDWPF_TEST_DIR"])
    if not val_dir.exists():
        print("RESULT valid=0 reason=\"validation split not built\"")
        print(f"  expected: {val_dir}", file=sys.stderr)
        print("  build it: cd ../windevolve_sdwpf && python val_windows.py",
              file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    result = evaluator.evaluate(str(FORECASTER), n_windows=args.n_windows)
    elapsed = time.perf_counter() - t0

    if not result.get("valid", False):
        # A failed experiment is still a result. Record it, revert, move on.
        print(f"RESULT valid=0 reason=\"{result.get('reason', 'unknown')}\" "
              f"elapsed_seconds={elapsed:.1f}")
        return 1

    budget_status = "OVER_BUDGET" if elapsed > TIME_BUDGET_S else "ok"

    # The agent parses this line. Keep the format stable.
    print(
        f"RESULT valid=1 val_score={result['competition_score']:.4f} "
        f"rmse={result['rmse']:.4f} mae={result['mae']:.4f} "
        f"n_windows={result['n_windows']} elapsed_seconds={elapsed:.1f} "
        f"budget={TIME_BUDGET_S} budget_status={budget_status}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
