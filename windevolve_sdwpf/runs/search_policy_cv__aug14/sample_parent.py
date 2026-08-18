# -*- coding: utf-8 -*-
"""
path: arms/karpathy/sample_parent.py

Description:
    Parent selection for the archive-policy arm. The agent is required to run
    this at the top of every iteration; it chooses the starting program so the
    agent does not. Read-only from the agent's point of view -- it must not be
    edited, and its output must not be overruled.

    The policy is a scaled-down version of the OpenEvolve/ThetaEvolve database
    sampler: one island, no eviction, no MAP-Elites bins. A single roll picks
    between exploiting the elite set, sampling uniformly at random, and taking
    the current best. Programs that scored worse are NEVER removed, which is
    the whole point -- a lineage can re-enter from a program greedy search
    would have discarded.

    Run from inside a worktree. Reads ./results.tsv, copies the chosen program
    over ./forecaster.py, and prints its id.

Usage:
    python ../../sample_parent.py            # normal use
    python ../../sample_parent.py --dry-run  # print the choice, touch nothing
"""

import argparse                    # --dry-run / --seed flags
import csv                         # results.tsv is tab-separated
import os                          # path joins and existence checks
import random                      # the draw itself
import shutil                      # copy archive entry over forecaster.py
import sys                         # exit codes

ELITE_N = 5                        # size of the elite set sampled from
P_ELITE = 0.60                     # exploit: draw from the top ELITE_N
P_RANDOM = 0.30                    # explore: draw uniformly from all valid
P_BEST = 0.10                      # take the current best outright
INVALID_SCORE = 999.0              # rows at or above this did not produce a score

RESULTS = "results.tsv"
ARCHIVE = "archive"
TARGET = "forecaster.py"


def load_valid_rows(path):
    """
    Description:
        Read results.tsv and return the rows that named a real, scoring
        program. Invalid, crashed and retrain rows are excluded: they are not
        usable as starting points.
    Args:
        path: path to results.tsv.
    Returns:
        List of (id, score) tuples, ordered as they appear in the file.
    """
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                score = float(row["competition_score"])
            except (KeyError, ValueError):
                continue                       # malformed line, skip it
            if score >= INVALID_SCORE:
                continue                       # crash / invalid / retrain
            pid = row["id"].strip()
            if os.path.exists(os.path.join(ARCHIVE, pid + ".py")):
                rows.append((pid, score))
    return rows


def choose(rows, rng):
    """
    Description:
        Apply the sampling policy to the valid rows.
    Args:
        rows: list of (id, score); rng: a random.Random instance.
    Returns:
        (chosen_id, mode) where mode is one of "best", "elite", "random".
    """
    ranked = sorted(rows, key=lambda r: r[1])          # lower score is better
    if len(ranked) == 1:
        return ranked[0][0], "best"                    # only the seed exists

    roll = rng.random()
    if roll < P_BEST:
        return ranked[0][0], "best"
    if roll < P_BEST + P_ELITE:
        elite = ranked[:min(ELITE_N, len(ranked))]
        return rng.choice(elite)[0], "elite"
    return rng.choice(ranked)[0], "random"


def main():
    parser = argparse.ArgumentParser(description="Sample a parent program.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the choice without overwriting forecaster.py")
    parser.add_argument("--seed", type=int, default=None,
                        help="fix the draw (for reproducing a run)")
    args = parser.parse_args()

    if not os.path.exists(RESULTS):
        sys.exit("no results.tsv in this directory -- run from inside a worktree")
    if not os.path.isdir(ARCHIVE):
        sys.exit("no archive/ directory -- create it before starting the loop")

    rows = load_valid_rows(RESULTS)
    if not rows:
        # Nothing scored yet. The seed is still in place; score it first.
        print("parent=000 mode=seed  (archive empty -- score the seed as id 001)")
        return

    rng = random.Random(args.seed)
    pid, mode = choose(rows, rng)
    score = dict(rows)[pid]
    src = os.path.join(ARCHIVE, pid + ".py")

    print("parent={} mode={} score={:.6f} pool={}".format(pid, mode, score, len(rows)))
    if args.dry_run:
        print("dry run -- {} left untouched.".format(TARGET))
    else:
        shutil.copyfile(src, TARGET)
        print("{} overwritten with {}. Edit from there.".format(TARGET, src))


if __name__ == "__main__":
    main()
