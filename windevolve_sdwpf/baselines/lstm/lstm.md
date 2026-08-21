# LSTM Baseline

Reference for `baselines/lstm/lstm.py` — **L**ong **S**hort-**T**erm **M**emory.

## The whole model

```
run an LSTM over the 144-step window, one step at a time
  → take the final hidden state (48 numbers)
  → one Linear layer → all 288 horizon values at once
```

Same idea as the GRU — walk the series carrying a summary — but with **two**
summaries instead of one.

## The 48-unit confusion, cleared up

**"48 units" does not mean 48 numbers in total.** It means the state is 48 slots
wide, and in an LSTM **each slot carries two numbers**: a cell value and a
hidden value.

So 48 units gives you one 48-vector of cell state *and* one 48-vector of hidden
state. The GRU has the same 48 slots with one number each.

| | GRU | LSTM |
|---|---|---|
| state vectors | 1 (hidden) | 2 (hidden + cell) |
| gates | 2 | 3 |
| weights per turbine | ~23k | ~30k |

## What the two states are for

**The cell state is the long-term ledger.** At each step the forget gate
multiplies it elementwise by a value in (0, 1), then the input gate adds a
bounded contribution. Note that it is **edited, never rewritten** — that is the
entire design intent, and the reason it can hold something steady for hundreds
of steps without being overwritten.

**The hidden state is what everything else sees.** It is the cell state passed
through tanh and filtered by the output gate. The next layer and the next
timestep only ever see this one.

### The three gates

| gate | activation | job |
|---|---|---|
| forget | sigmoid | what to erase from the cell |
| input | sigmoid | what to write into it |
| output | sigmoid | how much of the cell to expose as hidden |

Plus tanh on the cell content, for the same reason as the GRU: gates need a
fraction (sigmoid, 0 to 1), content needs to be signed and bounded (tanh, −1 to
1). Neither is a tuning choice.

## How the pieces connect

The **same weights are reused at every timestep** — that is what recurrence
means. There is one set of gate matrices, applied 144 times.

Layer 1 produces one 48-vector per step, and that vector goes **two places at
once**:

- **up** to layer 2 at that same step
- **forward** to layer 1 at the next step

Same numbers, two destinations. The final output is layer 2's hidden state at
the last step, projected to 288.

## Flat head, not autoregressive

The forecast is produced in **one shot** from the final hidden state — a single
linear layer emitting all 288 values.

The alternative is decoding step by step, feeding each prediction back in. Over
288 steps that is bad twice over: it is 288 sequential operations at inference,
and errors compound — a small mistake at step 5 becomes an input to steps 6
onward and the trajectory drifts.

## Parameters

| parameter | default | what it is |
|---|---|---|
| `lookback` | 144 | History steps; RNN cost is linear in this |
| `scope` | global | One model, or 134 per-turbine models |
| `hidden` | 48 | Width of **each** of the two state vectors |
| `layers` | 2 | Stacked LSTMs |
| `dropout` | 0.05 | Between layers |
| `loss` | mse | Matches the GRU and the competition solutions |

**Every one of these is copied from the GRU.** That is the entire point: with
lookback, width, depth, dropout and loss all pinned, the **only** difference
between `lstm.py` and `gru_global.py` is the cell itself. It is a controlled
comparison rather than two loosely related models.

An earlier draft used lookback 576 and hidden 128. Neither had evidence behind
it, and both were changed once the competition settings were confirmed.

## Why it gets no sweep

Once `scope` is an axis on the GRU, this model's original justification —
providing the global-vs-per-turbine comparison — disappears, because the GRU
provides it directly and at a fixed architecture.

What is left is "three gates instead of two," which the literature says is
noise on forecasting tasks. So this runs as **one configuration**, purely as the
controlled comparison, and it is the first thing to cut if GPU time is short.

It is also useful insurance: if the GRU's per-turbine arm cannot be run, this
becomes the substitute.

## What to expect

Most likely: within noise of the GRU, having used ~30% more parameters and more
wall clock to get there. If so, that is the finding — the extra gate is not
buying anything on this data, and reporting it plainly is more useful than
tuning until it wins.

## Cost

Roughly an hour for the single configuration across 5 folds. Wall clock is
linear in `lookback` because the steps are sequential and cannot be
parallelised over time.

## Fold awareness

```bash
export SDWPF_LSTM_CKPT="baselines/lstm/checkpoints/lstm_{fold}.pt"
```

Each fold trains on its own prefix and needs its own checkpoint. A path without
`{fold}` scored across folds would load one model for all five — four of which
trained on their own validation targets. The code raises rather than let that
happen.

## Self-contained by design

Full pipeline inlined, no imports from sibling baselines, so this file works as
a **seed program** for an evolve arm with nothing else on disk. The section
marked `SHARED PIPELINE` is duplicated across `mlp.py`, `transformer.py`,
`gru_global.py` and this file — **change it in one, change it in all four.**