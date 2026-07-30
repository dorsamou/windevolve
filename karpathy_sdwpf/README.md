# karpathy_sdwpf

A **single-level agentic autoresearch loop** on the SDWPF / Baidu KDD Cup 2022 wind power forecasting task, following the structure of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

This is the second of two harness arms in this project. The first, `../windevolve_sdwpf/`, is a population-based evolutionary search (OpenEvolve: MAP-Elites over islands). This one is the minimal alternative: a coding agent edits one file, runs it, keeps the change if the score improved and reverts it otherwise, and repeats.

Both arms optimise the same contract and are scored by the **same evaluator**, so the difference between them is the search mechanism rather than the plumbing.

---

## Why this arm exists

Two reasons, and the second is the point.

1. **A comparison.** Karpathy's loop is greedy hill-climbing with a single evaluation per candidate and no population. OpenEvolve maintains a MAP-Elites archive across islands. Running both on the same benchmark, through the same scorer, at a matched budget is a controlled comparison of inner-loop mechanism.

2. **To document how it fails.** A greedy single-seed loop on a noisy regression metric should, in principle, commit changes that are really noise. Building the loop by hand and watching that happen produces a measured failure rate on this benchmark rather than a claim borrowed from a paper. `FAILURE_MODES.md` is where that evidence accumulates, and it is a research output, not a scratch file.

**Single seed per experiment is deliberate.** Averaging over seeds would be the statistically sound choice and it is the wrong choice here — it would suppress the exact behaviour this arm is meant to measure. The multi-seed version is the documented fix, applied afterwards.

---

## Layout

```
karpathy_sdwpf/
├── program.md              # the instruction file the agent reads  (human-written)
├── forecaster.py           # the ONLY file the agent edits
├── run_experiment.py       # scores forecaster.py on the validation split
├── score_final.py          # locked-test measurement; run once, manually
├── results.tsv             # one row per experiment  (append-only)
├── FAILURE_MODES.md        # observed failure modes  (research output)
└── logs/                   # per-experiment stdout
```

Nothing here duplicates the scorer. `run_experiment.py` imports `evaluator.py` from `../windevolve_sdwpf/` and calls it unchanged.

## What the human writes vs. what the agent writes

| File | Author |
|---|---|
| `program.md` | **you** — objective, constraints, research directions |
| `forecaster.py` | the agent (seeded by you from `initial_program.py`) |
| `results.tsv` | the agent |
| `FAILURE_MODES.md` | the agent during runs; you annotate afterwards |

The premise of this design is that you stop writing forecasting code and start writing research directions. `program.md` is the real artefact.

---

## Requirements

**An agentic coding tool** — not a chat window. The loop needs something that can read files, edit them, run shell commands, and use git *unattended* for hours: Claude Code, Codex CLI, or Cursor in agent mode.

Pasting code back and forth in a chat caps you at the handful of cycles per day this design exists to escape.

Record which agent and which model you used. Reliability varies between them, and it is an experimental variable — Karpathy notes that some agents terminate early because they ignore the instruction to keep going.

---

## Setup

**1. Build the validation split** (shared with the other arm; run once):

```bash
cd ../windevolve_sdwpf
python val_windows.py
cd ../karpathy_sdwpf
```

**2. Seed the forecaster.** `forecaster.py` ships as a copy of `../windevolve_sdwpf/initial_program.py` (last-288-step persistence). Both arms must start from the identical seed or the comparison means nothing.

**3. Fill in the baseline table in `program.md`.** Score your baselines on the validation split and record them, including the seed-to-seed standard deviation of the GRU. The agent needs to know what counts as beating the baseline, and what counts as noise.

**4. Set the wall-clock budget.** `TIME_BUDGET_S` in `run_experiment.py` (default 600s). Set it from what your GRU costs end to end. Every experiment gets the same budget — this is the fairness constraint that makes runs comparable.

**5. Smoke-test before handing it to an agent:**

```bash
python run_experiment.py --n-windows 3
```

Expect a `RESULT valid=1 val_score=...` line. If the score is wildly different from the same program's `final_phase_test` score, the validation split is not representative — fix that before starting.

---

## Running

```bash
git checkout -b autoresearch/<tag>
```

Then point your agent at `program.md` and let it run. It will create `results.tsv`, record a baseline, and begin iterating.

Each experiment:

```bash
python run_experiment.py 2>&1 | tee logs/exp_<n>.log
```

Improvements get committed; failures get `git checkout -- forecaster.py`. The git history on the branch becomes the record of what actually worked.

---

## The metric

Identical to the other arm — the official competition score, `0.5 * (RMSE + MAE)`, per-turbine in MW, summed across the 134 turbines, averaged across windows. **Lower is better.** Full description in `../windevolve_sdwpf/README.md`.

Two things to keep in mind while reading `results.tsv`:

- Errors are **summed** over turbines, so a handful of bad turbines dominate the total.
- The entire top ten of the original competition spanned about **0.4 MW**. Improvements below the measured seed noise are not results.

## Search vs. final evaluation

`run_experiment.py` scores the **validation split** only. `final_phase_test` is never touched during the loop — `program.md` forbids it explicitly, and the agent has no reason to reach for it.

The locked set is scored once, at the end, by:

```bash
python score_final.py --confirm
```

The `--confirm` flag exists to make an accidental run impossible. Record the result, the date, and the git commit of the champion. Then stop; a second measurement after further tuning is no longer held out.

Report the **generalisation gap** (validation minus final) alongside the score. A non-zero gap is expected and informative, and comparing the gap between this arm and the OpenEvolve arm is one of the more interesting numbers this project will produce.

---

## Analysis to run afterwards

The end-of-session questions in `FAILURE_MODES.md`, in particular:

- how many changes were committed as improvements
- how many of those survive re-running at several fresh seeds
- the ratio between them — the **noise-chasing rate**

That ratio is the empirical case for the population-based, archive-preserving design in the other arm, measured on this benchmark rather than assumed.

---

## Related

- `../windevolve_sdwpf/` — the OpenEvolve arm and the shared scoring harness.

## Citation

Loop structure follows:

- Karpathy, A. *autoresearch*. <https://github.com/karpathy/autoresearch>

Methodology for comparing search mechanisms under a fixed budget, and for treating the validation-to-test gap as a first-class result:

- Toledo et al. "AI Research Agents for Machine Learning: Search, Exploration, and Generalization in MLE-bench." arXiv:2507.02554, 2025.

Task, scoring, and program contract follow the official Baidu KDD Cup 2022 SDWPF evaluator (PaddlePaddle/PaddleSpatial). Dataset: Zhou et al., arXiv:2208.04360 (2022); *Scientific Data* 11:649 (2024), doi:10.1038/s41597-024-03427-5.
