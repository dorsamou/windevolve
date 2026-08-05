# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/ablation/collect_results.py

Description:
    Walks the ablation run directories, pulls the best validation score from each
    run, and prints a per-arm table with mean, standard deviation and range over
    repeats. Also reports the seed score so improvement-over-seed is visible, and
    a per-arm best-of-k curve (k = 1..n_repeats), which is the top-k diagnostic
    from Toledo et al. (2025) Sec. 5.3 applied within validation only.

    Deliberately makes NO claim that one arm beats the other. It prints the
    numbers and the spread; with 3 repeats the spread is usually wide enough that
    a difference in means is not a result. Read the overlap, not the ordering.

Citation:
    Top-k / best-of-k framing follows Toledo et al., "AI Research Agents for
    Machine Learning", arXiv:2507.02554v2 (2025), Sec. 5.3.

Example usage:
    python ablation/collect_results.py
    python ablation/collect_results.py --runs ablation/runs --csv ablation/summary.csv
"""
import os                          # path walking
import re                          # checkpoint dir parsing
import csv                         # optional csv export
import json                        # read OpenEvolve metadata
import glob                        # find checkpoints
import argparse                    # CLI
import statistics                  # mean / stdev

BEST_KEYS = ("combined_score", "score", "fitness")   # tried in order


def _load_json(path):
    """Read a JSON file, returning None on any failure."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _score_from_obj(obj):
    """Pull a combined_score out of whatever shape OpenEvolve wrote."""
    if not isinstance(obj, dict):
        return None
    for key in ("metrics", "best_program_metrics", "best_metrics"):
        sub = obj.get(key)
        if isinstance(sub, dict):
            for k in BEST_KEYS:
                if isinstance(sub.get(k), (int, float)):
                    return float(sub[k])
    for k in BEST_KEYS:
        if isinstance(obj.get(k), (int, float)):
            return float(obj[k])
    return None


def best_score_for_run(run_dir):
    """
    Description:
        Find the best combined_score recorded anywhere in one run directory.
        Searches the run root and every checkpoint, since OpenEvolve's on-disk
        layout has moved between versions.
    Args:
        run_dir: path to one run's output directory
    Returns:
        (best_combined_score, n_checkpoints) -- score is None if nothing found.
    """
    candidates = []
    patterns = [
        os.path.join(run_dir, "*.json"),
        os.path.join(run_dir, "**", "*.json"),
    ]
    seen = set()
    for pat in patterns:
        for path in glob.glob(pat, recursive=True):
            if path in seen:
                continue
            seen.add(path)
            obj = _load_json(path)
            if obj is None:
                continue
            score = _score_from_obj(obj)
            if score is not None and score > -1e5:      # exclude penalty rows
                candidates.append(score)
            if isinstance(obj, list):
                for item in obj:
                    s = _score_from_obj(item)
                    if s is not None and s > -1e5:
                        candidates.append(s)

    ckpts = len([d for d in glob.glob(os.path.join(run_dir, "*"))
                 if os.path.isdir(d) and re.search(r"checkpoint", os.path.basename(d), re.I)])
    return (max(candidates) if candidates else None), ckpts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--csv", default=None, help="optional path to write a summary csv")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs, "*_seed*"))):
        name = os.path.basename(run_dir)
        arm, seed = name.rsplit("_seed", 1)
        done = os.path.isfile(os.path.join(run_dir, "COMPLETED"))
        best, ckpts = best_score_for_run(run_dir)
        rows.append({
            "arm": arm,
            "seed": seed,
            "completed": done,
            "checkpoints": ckpts,
            "best_combined_score": best,
            "competition_score": (-best) if best is not None else None,
        })

    if not rows:
        print("No runs found under {!r}.".format(args.runs))
        return

    print("\nPER-RUN")
    print("{:<8} {:>5} {:>10} {:>7} {:>20}".format(
        "arm", "seed", "done", "ckpts", "competition_score"))
    for r in rows:
        cs = "n/a" if r["competition_score"] is None else "{:.4f}".format(r["competition_score"])
        print("{:<8} {:>5} {:>10} {:>7} {:>20}".format(
            r["arm"], r["seed"], str(r["completed"]), r["checkpoints"], cs))

    print("\nPER-ARM  (competition_score, MW -- lower is better)")
    print("{:<8} {:>4} {:>10} {:>10} {:>10} {:>10}".format(
        "arm", "n", "mean", "stdev", "best", "worst"))
    per_arm = {}
    for r in rows:
        if r["competition_score"] is not None:
            per_arm.setdefault(r["arm"], []).append(r["competition_score"])
    for arm, vals in sorted(per_arm.items()):
        sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
        print("{:<8} {:>4} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            arm, len(vals), statistics.mean(vals), sd, min(vals), max(vals)))

    print("\nBEST-OF-k WITHIN ARM  (validation only; the Sec. 5.3 top-k curve)")
    for arm, vals in sorted(per_arm.items()):
        ordered = sorted(vals)
        curve = ["k={}: {:.4f}".format(k, ordered[0]) for k in range(1, len(ordered) + 1)]
        print("  {:<8} {}".format(arm, "   ".join(
            "k={}: {:.4f}".format(k, min(ordered[:k])) for k in range(1, len(ordered) + 1))))

    print("\nREAD THIS BEFORE QUOTING ANY OF THE ABOVE")
    print("  - 3 repeats is a floor, not a sufficient sample. If the per-arm")
    print("    ranges overlap, the difference in means is not a result.")
    print("  - Both arms ran on Claude; prior Gemini runs are NOT comparable.")
    print("  - These are VALIDATION scores. final_phase_test stays locked.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\nwrote {}".format(args.csv))


if __name__ == "__main__":
    main()
