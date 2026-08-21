# Transformer Baseline

Reference for `baselines/transformer/transformer.py` — a small patch-based
transformer encoder, ~123k parameters.

## Every step, with its dimensions

Take one turbine, one forecast origin. `B` = batch.

| step | shape after | weights |
|---|---|---|
| 0 · input | `(B, 144, 10)` | — |
| 1 · patch | `(B, 12, 120)` | 0 — reshape only |
| 2 · embed | `(B, 12, 32)` | 120 × 32 = 3,840 |
| 3 · position | `(B, 12, 32)` | 0 — fixed sinusoid |
| 4 · attention | `(B, 12, 32)` | ~4,096 |
| 5 · feedforward | `(B, 12, 32)` | ~4,096 |
| 6 · flatten | `(B, 384)` | 0 — reshape only |
| 7 · head | `(B, 288)` | 384 × 288 = 110,592 |

**About 85% of the parameters sit in that final projection.** The attention
machinery — steps 4 and 5 together — is only about 8k weights. Worth knowing
when interpreting the result: most of this model's capacity is an ordinary
linear layer.

### Step 1 — patching, in detail

Reshape the 144 steps into 12 blocks of 12 consecutive steps: `(B, 12, 12, 10)`,
then flatten the last two axes to `(B, 12, 120)`. Twelve rows of 120 numbers,
each row being **two hours of all ten channels**.

### Step 2 — embedding

**One shared** linear layer, 120 → 32, applied to each of the 12 rows
independently. "Shared" matters: the same matrix embeds every patch, which is
why the model still works if `patch_len` changes the token count.

### Step 3 — position

Attention is permutation-invariant on its own. Without a position signal,
shuffling the 12 tokens would change nothing, and the model could not tell
yesterday's patch from this morning's. The signal is a fixed sinusoid — zero
parameters.

### Step 4 — how the weighted average is actually built

Each token produces three vectors from its own 32 numbers, using three small
learned matrices:

- **query** — what am I looking for
- **key** — what do I contain
- **value** — what I would contribute

To update token 5: take token 5's query, dot-product it with **every** token's
key, giving 12 raw scores — high where the key matches what the query wants.
Push those through softmax so they are positive and sum to 1; those are the
weights. Token 5's new value is the sum of all 12 **value** vectors, each
multiplied by its weight. Repeat for all 12 tokens.

With 2 heads this happens twice in parallel with different matrices, and the
results are concatenated. The score matrix is `(B, 2, 12, 12)` — that 12 × 12
grid is the thing that grows quadratically.

### Why that is "attention" and not just a weighted sum

**The weights are computed from the data at run time.** Feed a different window
and the weights differ.

Contrast the MLP: its weights are fixed numbers learned once, so input position
400 always gets the same weight no matter what the value is. Attention decides
*per input* which positions to listen to — so it can learn "when the current
patch looks like a ramp starting, weight the patches from three days ago that
also looked like that." An MLP cannot express that; it only has "position 400
matters this much, always."

### Steps 6 and 7 — why they are shown separately

The encoder block (steps 4 + 5, with residual connections and layer norms) is
where **all** the learning about token relationships happens. Flatten is pure
reshaping — no parameters, no computation. Collapsing them into one box would
suggest the model does nothing between tokens and output, which is precisely
backwards.

## Why patch at all

Attention is **O(L²)**. One token per timestep would give 144 tokens and 20,736
attention pairs instead of 144.

But the cost argument is secondary. The real reason is **content**: in the
original transformer the tokens were words, and a word carries meaning. A single
10-minute power reading is one noisy number — asking attention to find
content-based relationships between individual readings is asking it to find
structure in noise. A patch carries two hours of all ten channels, which is
enough to have something worth matching on.

## Parameters

| parameter | default | what it is |
|---|---|---|
| `lookback` | 144 | Same window every other model gets |
| `patch_len` | 12 | Steps per token — 12 steps = 2 h, so 12 tokens |
| `d_model` | 32 | Width of each token's vector |
| `n_heads` | 2 | Parallel attention patterns |
| `n_layers` | 1 | Encoder blocks |
| `d_ff` | 64 | Feedforward width inside the block |
| `dropout` | 0.1 | |
| `scope` | global | One model, or 134 per-turbine models |

**`lookback` must be divisible by `patch_len`** — the model raises rather than
silently truncating.

**Deliberately sized down.** PatchTST uses 3 blocks at width 128 or more; this
runs 1 block at width 32, on the advisor's suggestion. Roughly 200 days per fold
cannot support a large transformer, and a small model that loses cleanly is a
more useful result than a large one that overfits.

## The one axis: `patch_len` = 1 / 6 / 12 / 24

Giving 144 / 24 / 12 / 6 tokens.

**`patch_len = 1` is the vanilla transformer** — one token per timestep, no
patching. It is the configuration the original 2017 paper describes, minus the
autoregressive decoder.

Including it turns a tuning axis into a **test of PatchTST's central claim**:
that patching, not attention, does most of the work. If patch 12 clearly beats
patch 1, that claim reproduces on this data. If they are level, the patching
story does not hold here, and the gain — if any — is coming from somewhere else.

## Is this a standard forecasting architecture?

Yes for the shape, no for the domain.

Patch → embed → encoder → flat head is the **PatchTST** recipe (Nie et al.,
ICLR 2023) and the standard way transformers are applied to long-horizon
forecasting. What it is *not* is wind-specific.

**No SDWPF top team used a plain transformer.** HIK (1st) used graphs plus GRUs;
88VIP used GBDT plus GRU. Combined with Zeng et al. showing simple linear models
beating transformers on long-horizon benchmarks, the prior here is genuinely
low.

## The budget caveat, to state when reporting

Every other model in the suite gets a real sweep and this one gets four points
on one axis. So **a loss here is evidence at this budget, not evidence that
attention cannot work on wind data.** Say that plainly rather than letting the
number stand alone.

If it does underperform badly, there are three explanations to separate before
concluding anything:

1. attention genuinely does not help at this data scale
2. the model is mis-sized
3. the joint-channel patch embedding is the limit — PatchTST is
   channel-independent, embedding each channel separately and sharing the
   backbone, and this file embeds all 10 channels jointly for simplicity

Only the first is a finding about transformers.

## Cost

The cheapest neural sweep in the suite despite being the most complex model —
4 points × 5 folds = 20 evaluations, and the model is tiny.

## Fold awareness

```bash
export SDWPF_TST_CKPT="baselines/transformer/checkpoints/transformer_{fold}.pt"
```

Each fold trains on its own prefix and needs its own checkpoint. A path without
`{fold}` scored across folds would load one model for all five — four of which
trained on their own validation targets. The code raises rather than let that
happen.

## Self-contained by design

Full pipeline inlined, no imports from sibling baselines, so this file works as
a **seed program** for an evolve arm with nothing else on disk. The section
marked `SHARED PIPELINE` is duplicated across `mlp.py`, `lstm.py`,
`gru_global.py` and this file — **change it in one, change it in all four.**