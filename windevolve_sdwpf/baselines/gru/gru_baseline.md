# GRU Baseline

Reference for `baselines/gru/gru_baseline.py` — **G**ated **R**ecurrent **U**nit.

> **CHAMPION — `scope = global`, `hidden = 96`, 43.591 MW (5-fold CV).**

---

## The whole model

```
walk the 144-step window one step at a time, carrying a 96-number summary
  → take the final hidden state
  → one Linear layer → all 288 horizon values at once
```

~115k parameters. Emitting the horizon in **one projection** rather than decoding
step by step.

At every step, each of the 96 units reads:

- the **10** current channel values, **and**
- all **96** values of the previous hidden state

That is 106 numbers in, one number out, per unit. Which is why the weights come
in pairs — `W_input` is 96 × 10 and `W_hidden` is 96 × 96, for each gate.

**The hidden state is the cell's own running notebook, not data arriving from
outside.** Layer 2 is identical except its 10-wide input is replaced by layer 1's
96 outputs.

### The two gates

| component | activation | job |
|---|---|---|
| update | sigmoid | how much of the old summary to keep vs. overwrite |
| reset | sigmoid | how much of the old summary to use when forming the candidate |
| candidate | tanh | the proposed new content |

### Why the activations are fixed

They do two different jobs and neither is a tuning preference.

**A gate must output a fraction** — keep 30% of the old state, overwrite 70%.
Sigmoid guarantees that by mapping anything into (0, 1). Put ReLU there and a
gate could output 5.7, meaning "keep 570% of the old state", and the state
explodes within a few steps.

**The candidate is signed content, not a fraction.** The summary has to be able
to move *down* as well as up, so the candidate must be able to be negative.
Sigmoid cannot do that — its output is always positive, so the state could only
ever grow. tanh is zero-centred and still bounded, to (−1, 1), so no single step
can shove the state arbitrarily far.

---

## The sweep

| axis | value | score (MW) | Δ vs ref | `fold_std` |
|---|---|---|---|---|
| reference | global, hidden 48, MSE | 43.786 | — | 5.30 |
| scope | 134 per turbine | 45.035 | +1.249 | 5.16 |
| hidden | 24 | 43.965 | +0.179 | 5.39 |
| **hidden** | **96** | **43.591** | **−0.195** | **5.11** |
| loss | half MSE + half L1 | 43.811 | +0.025 | 5.48 |

## Parameters

| parameter | default | what it is |
|---|---|---|
| `LOOKBACK` | 144 | 24 h of history; the official baseline's input length |
| `SCOPE` | **`global`** | One model for all 134 turbines — **champion** |
| `HIDDEN` | **96** | Size of the summary carried between steps — **champion** |
| `LAYERS` | 2 | Stacked GRUs; layer 2 reads layer 1 |
| `DROPOUT` | 0.05 | Between layers during training |
| `LOSS` | `mse` | What the official baseline uses |
| `CLIP_OUTPUT` | `True` | Clamp predictions to [0, 1500] kW |
| activation | sigmoid + tanh | **Fixed by the architecture** — not a sweep axis |


## Cost

As shipped (`global`): a few hours for the 4-point grid across 5 folds. RNN wall
clock is roughly linear in `LOOKBACK`, since the steps are sequential and cannot
be parallelised over time.

Under folds each fold needs its own checkpoint:

```bash
export SDWPF_GRU_CKPT="baselines/gru/checkpoints/gru_{fold}.pt"
```

A path without `{fold}` scored across folds would load one model for all five —
four of which trained on their own validation targets. The code raises rather
than let that happen.

