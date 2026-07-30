# Failure-mode log

Observed failure modes of a **minimal single-level agentic autoresearch loop** on
the SDWPF benchmark. Written by the agent during runs; annotated by the human
afterwards.

This is a research output. It is the evidence base for the design choices in the
OpenEvolve arm, and it is the part of the project that only someone who built the
loop by hand can write. Prior work reports these failure modes in the abstract;
this file records them as measurements on a specific benchmark.

---

## How to write an entry

One entry per notable event. Terse is fine; specific is essential.

Tag with one or more:

| Tag | Meaning |
|---|---|
| `edit-failure` | the edit did not apply, or applied somewhere unintended |
| `context-loss` | lost track of what had already been tried |
| `duplicate-proposal` | proposed something already tested |
| `noise-chasing` | committed a change later shown to be within seed noise |
| `plateau` | many consecutive experiments with no improvement |
| `reward-hacking` | improved the metric without improving the forecast |
| `degeneracy` | prediction rejected by the zero / near-constant guards |
| `budget-violation` | exceeded the wall-clock budget |
| `caching-error` | expensive work left inside `forecast()` instead of module level |
| `unexplained` | score moved for a reason nobody could identify |
| `crash-loop` | repeated failures on the same idea |

---

## Entries

<!-- Template — copy for each entry:

### exp 042 — blended persistence with diurnal profile
- **tag**: noise-chasing
- **what happened**: committed a 0.08 MW improvement as the new baseline.
- **how it was noticed**: re-running at 5 seeds gave a spread of 0.31 MW,
  wider than the improvement itself.
- **cost**: 6 subsequent experiments built on a baseline that was not real.
- **human note**: this is the core argument for the archive-based arm.

-->

_(empty — populated during the first session)_

---

## Post-run analysis

Fill in after each full session.

### Session metadata

| Field | Value |
|---|---|
| Date | |
| Agent / model | |
| Branch | |
| Wall-clock duration | |
| Experiments run | |
| Wall-clock budget per experiment | |
| Validation windows | |
| Approx. LLM calls / tokens | |

### Headline numbers

| Metric | Value |
|---|---|
| Seed baseline score | |
| Best score reached | |
| Total improvement | |
| Changes committed as improvements | |
| **Of those, how many survived re-running at 5 fresh seeds** | |
| **Noise-chasing rate** (committed but not reproducible) | ___ % |
| Experiment at which improvement effectively stopped | |
| Failed / invalid experiments | |
| Distinct ideas vs. near-duplicate proposals | |

### The number that matters

**Noise-chasing rate.** Take every change the loop committed as an improvement,
re-run each at several fresh seeds, and count how many survive. If a meaningful
share do not, that is the empirical case — measured on this benchmark, not
borrowed from a paper — for the multi-seed evaluation and diversity-preserving
archive used in the OpenEvolve arm.

Report it alongside the generalisation gap from `score_final.py`. Together they
describe two different ways a search can fool itself: chasing seed noise within
the validation signal, and overfitting the validation signal itself.

### Qualitative notes

- Where did the loop get stuck, and what did being stuck look like?
- Did it explore or exploit? Did it abandon promising directions too early?
- Which of the `program.md` research directions did it actually pursue, and
  which did it ignore?
- What would you change in `program.md` before the next session?
