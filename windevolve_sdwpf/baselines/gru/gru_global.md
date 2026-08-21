# GRU Baseline (sweepable)

Reference for `baselines/gru/gru_global.py` — **G**ated **R**ecurrent **U**nit.

## Why this file exists alongside `gru_baseline.py`

`baselines/gru/gru_baseline.py` is a faithful port of the published Baidu
baseline. It is **per-turbine by construction** — 134 independent models, no
global mode — and it is the reference the whole suite is measured against.
Sweeping `scope` on it would mean rewriting a published baseline, which we do
not want to do.

This file runs the **same GRU cell at either scope**, so `global` against
`per_turbine` is a clean one-variable comparison and the port stays untouched.

| file | scope | role |
|---|---|---|
| `gru_baseline.py` | per-turbine only | published reference, **not swept** |
| `gru_global.py` | either | the scope axis |

## How the cell works

The GRU walks the window one step at a time, carrying a **48-number summary**
called the hidden state.

### Where the 48 comes from

This is the part that confuses people. The 10 input channels are **not** the
only thing entering the cell. At every step, each of the 48 units reads:

- the **10** current channel values, **and**
- all **48** values of the previous hidden state

That is 58 numbers in, one number out, per unit. Which is why the weights come
in pairs — `W_input` is 48 × 10 and `W_hidden` is 48 × 48, for each gate.

The hidden state is the cell's own running notebook, not data arriving from
outside. Layer 2 is identical except its 10-wide input is replaced by layer 1's
48 outputs. About **23k weights** in total.

### The two gates

| gate | activation | job |
|---|---|---|
| update | sigmoid | how much of the old summary to keep vs. overwrite |
| reset | sigmoid | how much of the old summary to use when forming the candidate |
| candidate | tanh | the proposed new content |

### Why the activations are fixed

They do two different jobs and neither is a tuning preference.

**A gate must output a fraction** — keep 30% of the old state, overwrite 70%.
Sigmoid guarantees that by mapping anything into (0, 1). Put ReLU there and a
gate could output 5.7, meaning "keep 570% of the old state," and the state
explodes within a few steps.

**The candidate is signed content, not a fraction.** The summary has to be able
to move *down* as well as up, so the candidate must be able to be negative.
Sigmoid cannot do that — its output is always positive, so the state could only
ever grow. tanh is zero-centred and still bounded, to (−1, 1), so no single step
can shove the state arbitrarily far.

Change either and the thing you have built is a different cell that does not
have a name.

## One difference from the official port

The published baseline unrolls **432 steps** — 144 real inputs, then 288
zero-padded steps — reads one output per step, and discards the first 144.

This file uses a **flat head** instead: the whole horizon in one projection from
the final hidden state. That is the same choice `lstm.py` and `transformer.py`
make, so the neural models stay comparable to each other. It does mean this is
**not a byte-for-byte reproduction** of the published GRU, and the writeup
should say so.

## Parameters

| parameter | default | what it is |
|---|---|---|
| `lookback` | 144 | 24 h — the official baseline's input length |
| `scope` | global | One model, or 134 per-turbine models |
| `hidden` | 48 | Size of the summary carried between steps |
| `layers` | 2 | Stacked GRUs — layer 2 reads layer 1 |
| `dropout` | 0.05 | Between layers during training |
| `loss` | mse | The official baseline uses `nn.MSELoss` |
| lr | 1e-4 | Adam |

**These defaults are left alone.** They come from the official Baidu baseline
*and* were independently published by Team 88VIP (arXiv:2208.08952, Table 3) —
2 layers, 48 hidden units, dropout 0.05, learning rate 1e-4, Adam. Two
independent sources converging is about as much validation as a default gets, so
`scope` is the only axis here with real headroom.

## On the loss axis, and a correction

**The GRU is already trained on MSE.** So the observed smoothness — it captures
the daily shape but not local ramps — **cannot** be blamed on L1, and switching
*to* L1 would make it smoother, not less: L1 targets the conditional median, MSE
the conditional mean, and both are smooth when the future is genuinely uncertain.

The honest explanation is that minimising **any** average-error objective at a
48-hour horizon produces the conditional mean, and at that horizon the
conditional mean really is close to flat.

So why sweep loss at all? Because the competition score is **½(MAE + RMSE)**.
MSE is the objective matched to RMSE; L1 is matched to MAE; **neither alone
matches the score**. A half-and-half loss does — which is the one alternative
worth the compute, and notably what the archive agent independently arrived at.
Huber was dropped: it is a robustness tool, not something the scoring function
asks for.

Expect it to move the score slightly and the smoothness barely at all. That
negative result is worth more than a marginal tuning win.

## The scope axis

The comparison the whole suite exists for.

- **global** — one parameter set sees 134× the data
- **per_turbine** — each model gets a specialist fit on roughly 200 days

Cost is very asymmetric: `per_turbine` trains 134 models per fold, so **670
trainings for one grid point**. It gets its own run via `--only scope`.

Both outcomes are publishable. If global wins, the turbines share structure and
one program can serve the farm. If per-turbine wins, the turbines are
heterogeneous, and the prescribed response is a portfolio or per-cluster
approach — which is a finding, not a null result.

## Cost

Global: a few hours for the 4-point grid across 5 folds. Per-turbine: 6–12 h on
its own. RNN wall-clock is roughly linear in `lookback`, since the steps are
sequential and cannot be parallelised over time.

## Fold awareness

```bash
export SDWPF_GRUG_CKPT="baselines/gru/checkpoints/gru_global_{fold}.pt"
```

Each fold trains on its own prefix and needs its own checkpoint. A path without
`{fold}`, scored across folds, would load one model for all five — four of which
trained on their own validation targets. The code raises rather than let that
happen.

## Self-contained by design

Full pipeline inlined, no imports from sibling baselines, so this file works as
a **seed program** for an evolve arm with nothing else on disk. The section
marked `SHARED PIPELINE` is duplicated across `mlp.py`, `lstm.py`,
`transformer.py` and this file — **change it in one, change it in all four.**