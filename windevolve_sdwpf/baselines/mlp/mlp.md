# MLP Baseline

Reference for `baselines/mlp/mlp.py` — **M**ulti-**L**ayer **P**erceptron.

## The whole model

```
flatten the 144-step window into one long vector   (1,440 numbers)
  → Linear → GELU → dropout        (512 units)
  → Linear → GELU → dropout        (512 units)
  → Linear                          (288 outputs)
```

The simplest neural network there is. No recurrence, no attention, and — the
thing that matters — **no notion that the inputs are ordered in time**.

## How one layer works

**Step 1 — one unit.** A unit holds one weight for every input, multiplies each
input by its weight, sums them, and adds a bias:

```
sum = w₁·x₁ + w₂·x₂ + ... + w₁₄₄₀·x₁₄₄₀ + b
```

One number out. Think of it as a judge with an opinion about which inputs
matter — a large weight on input 500, near-zero on input 12.

**Step 2 — bend it.** Push that sum through GELU, which passes positives through
roughly unchanged and squashes negatives toward zero.

**Step 3 — a layer is 512 units doing this in parallel.** All 512 see the same
1,440 inputs, but each has its own weights, so each learns to look for something
different. Out come 512 numbers.

**Step 4 — stack another.** The second layer's units take the *first* layer's 512
outputs as their inputs, building on features rather than raw data.

**Step 5 — the output layer.** 288 units, each a weighted sum of those 512, with
no activation. Those 288 numbers *are* the 48-hour forecast.

### Why the activation is load-bearing

Without it, a weighted sum of weighted sums is **still just a weighted sum** —
two stacked linear layers collapse into a single linear map, and depth buys
nothing at all. The activation is the only reason stacking helps, which is why
it is a sweep axis rather than a detail.

## Weights

| connection | shape | weights |
|---|---|---|
| input → hidden 1 | 1,440 × 512 | 0.74M |
| hidden 1 → hidden 2 | 512 × 512 | 0.26M |
| hidden 2 → output | 512 × 288 | 0.15M |
| | **total** | **1.15M** |

## Parameters

| parameter | default | what it is |
|---|---|---|
| `lookback` | 144 | History steps flattened into the input vector |
| `scope` | global | One model, or 134 per-turbine models |
| `hidden` | 512 | Units in each hidden layer |
| `depth` | 2 | Number of hidden layers |
| `activation` | gelu | `relu` / `gelu` / `tanh` |
| `dropout` | 0.1 | Units zeroed at random each training step; off at prediction |
| `loss` | mse | Matches the official GRU baseline |

**`lookback` is pinned at 144** — the same window the GRU, LSTM and Transformer
get. That is deliberate: it means a difference between any two of them is a
difference in *architecture* and nothing else. An earlier draft used 288 here,
which would have confounded architecture with input length.

**`dropout`** switches off 10% of hidden units at random on each training step,
so no single unit can become indispensable and the network is forced to spread
its representation. It is turned off entirely at prediction time.

**`hidden` and `depth`** are the capacity axes. Width matters more than depth
here — the hard part of this task is the input mapping, not deep composition.
1024 units was deliberately excluded: 5.9M weights on roughly 200 days of data
is not a serious candidate, only a way to confirm overfitting slowly.

## Why it is in the suite

The MLP is a **control**. It isolates *nonlinearity* from *architecture*:

| model | nonlinear? | time structure? |
|---|---|---|
| Climatology | no | none |
| **MLP** | **yes** | **none** |
| GRU / LSTM | yes | sequential |
| Transformer | yes | attention |

So **GRU − MLP** is what the sequential inductive bias buys, and **Transformer −
MLP** is what attention buys. Without the MLP in between, both effects are
confounded with the nonlinearity and neither number means anything on its own.

It is not expected to win. It is here to make the other results interpretable.

## The weakness, stated plainly

Every hidden unit connects to all 1,440 inputs, and **nothing tells the network
which of them is recent**. Input 1,439 is ten minutes old and input 4 is a day
old, and the MLP has to discover that from data — spending weights on structure
a GRU gets for free from its shape. That is exactly the gap the suite is
measuring.

## Where it has no precedent

This is the only model in the suite with **no competition-validated reference
configuration**. The GRU's settings come from the official baseline and were
independently published by Team 88VIP; the MLP's width and depth are generic
literature defaults. That is why its capacity is genuinely unknown and why its
grid keeps a capacity axis when most others do not.

## Cost

Minutes per fold. The `scope=per_turbine` point is the expensive one — 134
models per fold, 670 trainings across the sweep — and is held back for a
separate run with `--skip scope` / `--only scope`.

## Fold awareness

Each fold trains on its own prefix, so **each fold needs its own checkpoint**.
The path carries a `{fold}` placeholder filled from `envs["fold"]`:

```bash
export SDWPF_MLP_CKPT="baselines/mlp/checkpoints/mlp_{fold}.pt"
```

A path *without* the placeholder, scored across folds, would load one model for
all five — four of which trained on their own validation targets. The code
raises rather than let that happen silently.

## Self-contained by design

The full pipeline — data loading, scaler, windowing, training loop, inference —
is inlined in this one file, with no imports from sibling baselines. That is
deliberate: this file is a **seed program** for the evolve arms, and an arm
rewrites one file. If the pipeline lived in a shared module, evolution could
only edit the architecture wrapper while the windowing and normalisation sat
outside the search space.

The cost is that the section marked `SHARED PIPELINE` is duplicated across
`mlp.py`, `lstm.py`, `transformer.py` and `gru_global.py`. **Change it in one,
change it in all four** — a divergence there silently makes their scores
incomparable.