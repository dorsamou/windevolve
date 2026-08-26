# LSTM Baseline

Reference for `baselines/lstm/lstm.py` — **L**ong **S**hort-**T**erm **M**emory.

> **CHAMPION — the GRU-matched defaults, 45.083 MW (5-fold CV).**

---

## The whole model

```
run an LSTM over the 144-step window, one step at a time
  → take the final hidden state (48 numbers)
  → one Linear layer → all 288 horizon values at once
```

Same idea as the GRU — walk the series carrying a summary — but with **two**
summaries instead of one.

**"48 units" does not mean 48 numbers in total.** It means the state is 48 slots
wide, and in an LSTM **each slot carries two numbers**: a cell value and a hidden
value.

So 48 units gives one 48-vector of cell state *and* one 48-vector of hidden
state. 

| | GRU | LSTM |
|---|---|---|
| state vectors | 1 (hidden) | 2 (hidden + cell) |
| gates | 2 | 3 |
| weights per model | 36,864 | 44,448 |

### What the two states are for

**The cell state is the long-term ledger.** At each step the forget gate
multiplies it elementwise by a value in (0, 1), then the input gate adds a
bounded contribution. Note it is **edited, never rewritten** — that is the entire
design intent, and the reason it can hold something steady for hundreds of steps.

**The hidden state is what everything else sees.** It is the cell state passed
through tanh and filtered by the output gate. The next layer and the next
timestep only ever see this one.

| gate | activation | job |
|---|---|---|
| forget | sigmoid | what to erase from the cell |
| input | sigmoid | what to write into it |
| output | sigmoid | how much of the cell to expose as hidden |

Plus tanh on the cell content, for the same reason as the GRU: gates need a
**fraction** (sigmoid, 0 to 1), content needs to be **signed and bounded** (tanh,
−1 to 1). Neither is a tuning choice.

### How the pieces connect

The **same weights are reused at every timestep** — that is what recurrence
means. There is one set of gate matrices, applied 144 times.

Layer 1 produces one 48-vector per step, and that vector goes **two places at
once**: up to layer 2 at that same step, and forward to layer 1 at the next step.
Same numbers, two destinations.

The final output is layer 2's hidden state at the last step, projected to 288.

### Flat head, not autoregressive

The forecast is produced in **one shot** from the final hidden state — a single
linear layer emitting all 288 values.

The alternative is decoding step by step, feeding each prediction back. Over 288
steps that is bad twice over: it is 288 sequential operations at inference, and
errors compound — a small mistake at step 5 becomes an input to steps 6 onward
and the trajectory drifts.

---


**Every setting is pinned to the GRU's**, so the *only* difference between
`lstm.py` and `gru_baseline.py` is the cell:

---

## Parameters

| parameter | default | what it is |
|---|---|---|
| `LOOKBACK` | 144 | History read; RNN cost is linear in this |
| `SCOPE` | `global` | One model for all 134 turbines |
| `HIDDEN` | 48 | Width of **each** of the two state vectors |
| `LAYERS` | 2 | Stacked LSTMs |
| `DROPOUT` | 0.05 | Between layers and before the head |
| `LOSS` | `mse` | Matches the GRU and the competition solutions |
| `CLIP_OUTPUT` | `True` | Clamp predictions to [0, 1500] kW |
| activation | sigmoid + tanh | **Fixed by the architecture** — not a sweep axis |



## Cost

Roughly an hour for the single configuration across 5 folds — the cheapest neural
model in the suite. RNN wall clock is linear in `LOOKBACK`, since the steps are
sequential and cannot be parallelised over time.

Under folds each fold needs its own checkpoint:

```bash
export SDWPF_LSTM_CKPT="baselines/lstm/checkpoints/lstm_{fold}.pt"
```

A path without `{fold}` scored across folds would load one model for all five —
four of which trained on their own validation targets. The code raises rather
than let that happen.

