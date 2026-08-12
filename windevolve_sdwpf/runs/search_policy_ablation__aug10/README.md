# Search policy ablation — greedy vs. archive

**Run date:** 2026-08-10 · **Seed:** GRU baseline · **Budget:** 120 evaluations per arm

---

## Headline

Two runs of the agentic loop, identical in every respect except how the next
starting program is chosen. The archive policy beat greedy on validation by
0.87 MW at matched budget, and the mechanism behind that advantage is clearly
visible in the lineage data.

**On the held-out test set both champions are worse than the seed they started
from.**

| | validation | test | vs. seed on test |
|---|---:|---:|---:|
| GRU seed | 44.537 | **46.809** | — |
| greedy champion 115 | 43.728 | 47.738 | +0.93 worse |
| archive champion 117 | **42.863** | 48.209 | +1.40 worse |
| published Baidu baseline | — | 47.850 | +1.04 worse |

240 evaluations of automated search produced no program that generalises better
than its own starting point. The seed also beats the published competition
baseline on test, and neither champion does.

The rest of this document reports the policy comparison as measured, then treats
the validation-to-test gap as the finding it is.

---

## 1. Design

| | |
|---|---|
| **Varied** | selection policy — which program the agent starts from each iteration |
| Seed program | GRU baseline, 134 per-turbine models, `44.53736578814683` MW |
| Operator | Claude Code CLI, one session per arm, context accumulated across the run |
| Evaluator | `evaluation/evaluator.py`, 24 non-overlapping validation windows (days 198–245) |
| Metric | ½(RMSE + MAE) in MW, summed over 134 turbines, lower is better |
| Budget | 120 evaluations per arm |
| Retrain cap | 15 per arm |
| Arms | `ws/greedy-gru`, `ws/archive-gru` |

Both worktrees were verified at `44.53736578814683` before launch, and re-scoring
the same program returns a bit-identical value, so scoring is deterministic and
observed differences are real rather than re-rolls.

### The two loops

**greedy-gru — population of 1**

```
edit forecaster.py  →  git commit  →  score
                                        ├─ improved → keep the commit   (status: keep)
                                        └─ else     → git reset --hard  (status: discard)
```

`parent_id` is always the current best. A program that scores worse than its
parent no longer exists after the reset.

**archive-gru — population grows monotonically**

```
python ../../sample_parent.py     (overwrites forecaster.py, prints parent id)
        ↓
read it  →  edit  →  git commit  →  score  →  cp forecaster.py archive/<id>.py
        ↓
back to the sampler
```

No keep/discard decision and no `git reset` anywhere. Git records history rather
than storing state.

### The sampler

`arms/karpathy/sample_parent.py` is the policy, and it is external to the agent:
unwritable via the settings deny list, required at the top of every iteration,
and it **overwrites `forecaster.py`** before the agent edits anything. The agent
does not choose its own parent and cannot decline the one it is given.

One roll per iteration:

| probability | draw |
|---|---|
| 10% | the current best program |
| 60% | uniform over the top 5 by score |
| 30% | uniform over *every* valid program scored so far |

Nothing is ever evicted. The 30% uniform draw is the mechanism under test — the
only route back to a program that regressed.

### What was held fixed

`program_greedy.md` and `program_archive.md` are byte-identical except for 56
lines, all inside the loop section and the status vocabulary. The task contract,
all ten prohibitions, the training-data rule, the checkpoint rules, the
*No surrogate scoring* section, the simplicity criterion, the evaluator
invocation and output format, the `results.tsv` spec, and the entire budget
section are shared verbatim.

**One data row in `results.tsv` is one evaluation**, whatever its status.
Counting evaluations rather than wall-clock is what makes the arms comparable:
scoring costs ~8.3 s, retraining all 134 turbines ~1 hour, and the arms retrained
7 and 0 times respectively.

---

## 2. Budget composition

| status | greedy | archive |
|---|---:|---:|
| discard | 88 | 0 |
| keep | 26 | 0 |
| scored | 2 | 113 |
| invalid | 4 | 0 |
| **retrain** | **0** | **7** |

### Why greedy never retrained

Greedy deleted the neural network at **evaluation 002** — its first experiment,
"replace GRU with fitted linear anomaly-decay (mean anchor)". `archive/002.py`
contains zero torch references. Rows 002 through 011 are all hand-built
level-and-deviation models, and every one scores worse than the seed's 44.537.

Having removed the network on move one, there was no GRU checkpoint left to
retrain. It did build a neural network later: at evaluation 041 it wrote a
feedforward net in torch — 288 inputs (one day of the turbine's own power history
plus one day of the farm average), two hidden layers of 256 units, 288 outputs,
one per step of the horizon — trained on GPU inside the evaluation rather than as
a separate checkpoint job, which is why the row is `discard` rather than
`retrain`. It scored 46.864 against 44.262 and the agent wrote: *"a nonlinear
response loses 2.6 MW, and that closes the neural avenue."* It never revisited
the direction in the remaining 79 evaluations.

The archive arm retrained 7 times: mix loss (0.5·MSE + 0.5·L1) at 004, an
asymmetry correction at 007, and GRU hidden size 128 at 050, whose weights the
champion loads.

---

## 3. Validation trajectory

| evaluations | greedy | archive | gap |
|---:|---:|---:|---:|
| 20 | 44.515 | 43.769 | 0.746 |
| 40 | 44.262 | 43.521 | 0.741 |
| 60 | 43.886 | 43.009 | 0.877 |
| 80 | 43.858 | 42.935 | 0.923 |
| 100 | 43.733 | 42.900 | 0.833 |
| 120 | 43.728 | 42.863 | 0.865 |

Archive led at all six checkpoints, and archive at 40 evaluations already beat
greedy at 120. Over the final 20 evaluations greedy improved 0.005 MW and archive
0.037 MW.

Both curves flatten. The difference in end-of-run rate comes from two numbers in
two single runs and should not be over-read — and in light of section 6, greedy's
earlier convergence is not obviously a defect.

![trajectory](figs/trajectory.png)

---

## 4. Valley crossing

The structural result, and the part that does not depend on n=1.

Under greedy a program scoring worse than its own parent is destroyed by
`git reset --hard` and cannot be edited again. So for each program P in the
archive arm, ask whether P was itself a regression on its own parent; if so,
greedy would no longer have P. Then ask whether any child of P improved.

**18 children improved on a parent greedy would have deleted.** Largest:

| child | score | parent | parent score | gain |
|---|---:|---|---:|---:|
| 083 | 42.930 | 003 | 44.297 | 1.367 |
| 116 | 42.907 | 042 | 44.259 | 1.352 |
| 048 | 43.220 | 008 | 44.450 | 1.230 |
| 041 | 43.515 | 015 | 44.721 | 1.206 |
| 101 | 42.896 | 027 | 43.890 | 0.994 |

Program 003 scored 44.297 against its parent 002 at 44.233 — a regression. Eighty
evaluations later the sampler handed it back, and the edit made from it scored
42.930, beating the greedy arm's entire final validation result.

![valley crossing](figs/valley_crossing.png)

---

## 5. The size of the moves

| | best single edit | second | third |
|---|---:|---:|---:|
| greedy | 0.360 | 0.097 | 0.082 |
| archive | 1.367 | 1.352 | 1.281 |

### Greedy's best edit: one number

042 → 054, worth 0.360 MW. The entire diff:

```python
CLIM_LEN = 10 * DAY_LEN   # was 14
```

**In plain terms: use the last 10 days of the input file instead of the last 14
to work out each turbine's typical power level.**

That level matters because skill is essentially gone by 48 hours ahead
(autocorrelation 0.20 at 24 h, 0.02 at 48 h). The far end of the forecast is not
really a prediction of the weather — it is a prediction of the turbine's normal
output, and the metric is decided mostly by whether that number is right.

Why the shorter window helped: days 198–245 were a period of declining wind. A
14-day window still contains part of the earlier, windier stretch, so it averages
out too high and every long-horizon forecast is biased upward. Ten days weights
recent conditions more heavily and lands closer to the truth. But that only works
while the decline continues — it is a bet on the regime, not a better estimator,
which is exactly the sort of edit not expected to transfer.

### Archive's best edit: three changes at once

003 → 083, worth 1.367 MW, +81 lines. In plain terms:

1. **A bigger network, trained differently.** The seed's GRU has a hidden size of
   48; this swaps in weights with hidden size 128, trained with a loss that is
   half squared error and half absolute error — matching the competition metric,
   which is half RMSE and half MAE. Those weights came from retrain 050.
2. **Less blanket shrinking of the output.** Predictions were being multiplied by
   0.80 before being submitted; this raises it to 0.95. The shrink exists because
   training targets the average while half the metric rewards the median, and
   power output is right-skewed. A larger model fits that distribution better and
   needs less of a blunt correction.
3. **New post-processing on top of the network.** Estimate each turbine's recent
   level from the input file counting only rows the competition would score;
   smooth predictions across physically nearby turbines with a distance-weighted
   kernel; blend in short-run persistence.

**The 1.367 MW figure overstates a single edit**, because the retrain that
produced those weights was its own budget row. The comparison with greedy's
0.360 MW is still informative about the *kind* of move each policy affords —
greedy always edits the current best, so its moves are cautious tweaks to a
working program; archive is handed older programs and has to change them
substantially — but it is not a like-for-like measure of edit size.

## 6. Test set — the result reverses

Scored once per arm with `evaluation/competition_score.py`, the official Baidu
grader over 142 locked `final_phase_test` windows.

| | validation | test | val→test gap | error % of nameplate |
|---|---:|---:|---:|---:|
| GRU seed | 44.537 | 46.809 | 2.272 | 34.9% |
| greedy 115 | 43.728 | 47.738 | 4.010 | 35.6% |
| archive 117 | 42.863 | 48.209 | 5.346 | 36.0% |

*Error % of nameplate = competition_score / 134 MW. The official `accuracy` field
is a different quantity: 1 − (summed RMSE over the **last 24 h** only) / 134,
which discards half the horizon. For interpretability prefer either the error %
above or normalised MAE (`total_mae` / 134): greedy 32.0%, archive 32.1%.*

Two things follow.

**The validation ranking inverted.** Archive won validation by 0.87 MW and lost
test by 0.47 MW. Every conclusion in sections 3–5 is a statement about the
validation split, and the policy comparison does not survive to held-out data.

**Selection intensity predicts the gap.** Greedy made 26 keeps and has a 4.010 MW
gap; archive made 113 scored selections and has a 5.346 MW gap; the seed, selected
zero times, has a 2.272 MW gap. The arm that searched harder generalised worse,
in the order the selection-bias account predicts. This is a testable claim, not a
narrative, and it is the most useful thing this run produced.

### Is this the agent's doing, or the validation design?

The agent never read `val_windows/outfile/`; the deny list blocks it and
`evaluator.py` enforces masking, shape, and no-future-timestep independently.
No rule was broken.

But the greedy agent's notes state plainly that *"days 198–245 are a genuinely
lower-wind regime."* It did not need the outfiles to know that. Two legitimate
channels carry the information:

1. **The infiles.** Each validation window's infile contains the 14 days of
   history preceding it — that is the model's input and cannot be withheld.
   Across 24 windows the infiles collectively describe the conditions of the
   validation period.
2. **The scores.** One hundred and twenty returned numbers are themselves
   information about the period. Adapting to them is what optimisation *is*.

So this is not leakage; it is regime-specific tuning of a fixed validation
period, and it cannot be prohibited — only made harder to exploit. Both
champions carry constants selected against one contiguous 48-day low-wind
stretch: `CLIM_LEN`, `PRED_SCALE`, `LEVEL_DAYS`, `LEVEL_QUANTILE`, `PULL_TAU`,
shade bounds. On 142 test windows drawn from elsewhere those bets do not pay.

**The implied fix is a validation-design change, not a search change:** more
windows, spread across the year rather than contiguous, and a selection split
held out from the tuning split so that champion choice is not made on the same
data the constants were fitted to.

---

## 7. Limitations

**n = 1 per arm.** The measured re-roll spread for this loop is 0.50–0.96 MW and
the validation gap is 0.865 MW, so as a single number the validation difference
is not separable from run-to-run variance. Archive led at all six checkpoints,
which is harder to attribute to luck, but a replicate of each arm remains the
right next experiment.

**Different work, not just different scores.** Archive retrained 7 times and
greedy 0, so the arms spent identical evaluation budgets on very different
wall-clock and very different kinds of work.

**Checkpoint asymmetry on test.** The archive champion runs on `ckpt_050`,
trained on `VAL_TRAIN_FILE` (days 1–197) — correct and leak-free, but it never
saw days 198–245, whereas the greedy champion has no learned weights at all and
estimates its level from each infile at inference time. This asymmetry favours
greedy on test and is a candidate partial explanation for the reversal.

**Scaled-down policy.** Islands, MAP-Elites niching, eviction, batch sampling and
test-time RL were all dropped. At 120 programs those components are inert
(ten islands would hold twelve programs each; a 10,000-slot population never
evicts), and bundling inert mechanisms makes results unattributable. ThetaEvolve
Appendix E.4 / Table 10 ablates roughly this simplification and reports it as
their weakest configuration — but that ablation collapsed diversity to a pure
priority queue, whereas the 30% uniform draw here is a diversity mechanism, and
section 4 is evidence it fired. MAP-Elites niching is the one dropped component
genuinely active at this budget and is the natural third arm.

**Known deviation in the sampler.** `invalid` and `crash` rows are excluded from
the parent pool. ThetaEvolve instead admits them with penalty scores
(−0.4 / −0.3 / −0.2 / −0.1) so they rank low but stay reachable. The archive arm
logged zero invalid and zero crash rows, so the exclusion never fired in this
run, but it should be changed to a penalty tier next time.

---

## 8. What to do next

1. **Replicate both arms.** The validation gap sits inside the re-roll spread.
2. **Rebuild the validation split** — more windows, non-contiguous, and a
   selection split separate from the tuning split. The val→test gap is currently
   larger than any effect the search produces, which makes it the binding
   constraint on the whole project.
3. **Report test scores per arm as standard**, not only at the end.
4. **MAP-Elites niching as a third policy arm**, once 1 and 2 are in place.

---

## 9. Files

```
arms/karpathy/
├── program_greedy.md      brief for the greedy arm
├── program_archive.md     brief for the archive arm (56 lines differ)
├── sample_parent.py       the policy; read-only to the agent
└── ws/
    ├── greedy-gru/        branch autoresearch/aug10-greedy-gru
    └── archive-gru/       branch autoresearch/aug10-archive-gru
        ├── forecaster.py      the only tracked file — the search state
        ├── checkpoints/       134 seed GRU models, immutable
        ├── archive/<id>.py    every program scored
        ├── logs/<id>.log      raw evaluator output
        ├── .cache/ckpt_<id>/  retrained weights
        ├── results.tsv        id, parent_id, competition_score, eval_seconds, status, description
        └── FAILURE_MODES.md   the agent's running notes
```

Both arms are git worktrees sharing one repository at
`arms/karpathy/workspace/.git` — files live in `ws/`, commit history in
`workspace/.git`. Everything except `forecaster.py` is gitignored by design:
greedy's `git reset --hard` reverts tracked files, so an untracked `results.tsv`
survives a discard where a tracked one would be rewound.

Analysis: `notebooks/search_policy_analysis.ipynb`.
Scoring a champion on test requires copying it to the worktree root first — the
archive champion resolves `.cache/ckpt_050` relative to its own file location.
