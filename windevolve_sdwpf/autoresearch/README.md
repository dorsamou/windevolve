# arms/karpathy — greedy agentic search

One of the search arms in this project. A coding agent edits a single
forecasting program, scores it on the held-out validation split, and keeps or
discards the change. Greedy hill-climbing, with git as the search state.

In the vocabulary of Toledo et al. (2025), this arm fixes the search policy to
greedy and varies the operator: instead of an LLM emitting a mutated program,
the operator is a full coding agent with shell access, able to run the scorer,
read the traceback, and revise before committing.

## Attribution

Adapted from Andrej Karpathy's `autoresearch`
(https://github.com/karpathy/autoresearch, March 2026), which applied this loop
structure to LLM pretraining. The loop mechanics — one editable file, an
immutable scorer, append-only `results.tsv`, branch-advance on improvement and
`git reset` on regression, and the "never stop" instruction — are his.

**This is an adaptation, not a reproduction.** The task, the metric, and
several mechanics differ. The deviations are listed below so that no result
from this arm is mistaken for a replication of his.

## Deviations from Karpathy's `program.md`

| Deviation | Why |
|---|---|
| `FAILURE_MODES.md` as cross-run memory | Project requirement. Karpathy has no cross-run memory; his agent's memory is the context window plus `results.tsv`. |
| Four statuses (`keep` / `discard` / `invalid` / `crash`) | Karpathy has three. This evaluator never raises — it returns `valid: False` with a reason — so "rejected" and "crashed" are distinct outcomes carrying different information. |
| `eval_seconds` column instead of `memory_gb` | VRAM is not the binding constraint here; wall-clock per evaluation is. |
| Explicit "do not work around a check" instruction | The official guards have a documented hole (see the repo README): pinning 89% of turbines to a constant passes both official guards and scores ~19% better than honest persistence. `evaluator.py` closes it, but the agent is told not to go looking. |
| Leakage warning about `VAL_TRAIN_FILE` | Karpathy's task has no train/validation leakage risk. Here, fitting on the full 245-day file would train on the validation targets. |
| "No additional files; fit lazily and cache" | The `forecast(envs)` contract is called per window. Karpathy's `train.py` is self-contained and leaves no artifact. |
| `paths.py` verification during setup | Replaces his check that the data cache and tokenizer exist. |
| Budget is evaluations, not wall-clock | Karpathy fixes a 5-minute training budget so runs are comparable across hardware. Here every candidate sees the same 24 validation windows, and the scarce resource is evaluations — which is also the common currency against the OpenEvolve arm. |

## Layout

```
arms/karpathy/
├── README.md          this file
├── program.md         the agent's brief — the operator definition
├── environment.txt    package list the agent may rely on
└── workspace/         the agent's cwd; a standalone git repo, gitignored
                       by the parent
    ├── forecaster.py  the one file the agent edits
    ├── results.tsv    untracked, append-only
    ├── FAILURE_MODES.md
    └── run.log        scratch
```

`workspace/` is a separate git repository on purpose. The loop uses
`git reset --hard` to discard failed experiments; confining git to a directory
containing only `forecaster.py` means no reset can reach the evaluation stack,
the baselines, or anything else in the parent repo.

## Running it

Requires the validation split to exist. Generate it first if `python paths.py`
reports `VAL_DIR` as MISSING.

```bash
cd arms/karpathy/workspace
claude
```

Then:

```
Read ../program.md and let's kick off a new experiment. Let's do the setup first.
```

Use tmux for overnight runs.

## Reading the results

`results.tsv` is the record. It is untracked by design — a `git reset` would
otherwise erase the log of the experiment that caused it.

Budget accounting: every row is one evaluation, including `invalid` and
`crash`. Comparisons against other arms are made at a matched evaluation count,
so rows beyond that count are excluded from the comparison — though they remain
data about what the operator does with a larger budget.

## References

- Karpathy, A. `autoresearch`. https://github.com/karpathy/autoresearch, 2026.
- Toledo, E. et al. "AI Research Agents for Machine Learning: Search,
  Exploration, and Generalization in MLE-bench." arXiv:2507.02554, 2025.