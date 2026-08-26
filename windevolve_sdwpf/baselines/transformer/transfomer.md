# Transformer Baseline

Reference for `baselines/transformer/transformer.py`

> **CHAMPION — `patch_len = 24`, 46.171 MW (5-fold CV).**

---

## What the model does, in one picture

You have **144 readings** — 24 hours of data, one every 10 minutes.

A transformer cannot work on a raw list of numbers. It needs **tokens**: chunks
it can compare against each other. `patch_len` decides how big a chunk is.

```
patch_len = 1    each reading is its own token      → 144 tokens
patch_len = 6    glue 6 together (1 hour each)      →  24 tokens
patch_len = 12   glue 12 together (2 hours each)    →  12 tokens
patch_len = 24   glue 24 together (4 hours each)    →   6 tokens   ← DEFAULT
```

Same 144 numbers every time. Only the chunk size changes.

Then every token is compared against every other token, and each one is replaced
by a weighted average of all of them(attention). Finally the
tokens are flattened and one linear layer produces all 288 forecast values.


## Every step, with its dimensions

`B` = batch. At the default `patch_len = 24`:

| step | shape after | weights |
|---|---|---|
| 0 · input | `(B, 144, 10)` | — |
| 1 · patch | `(B, 6, 240)` | 0 — reshape only |
| 2 · embed | `(B, 6, 32)` | 240 × 32 = 7,680 |
| 3 · position | `(B, 6, 32)` | 0 — fixed sinusoid |
| 4 · attention | `(B, 6, 32)` | ~4,200 |
| 5 · feedforward | `(B, 6, 32)` | ~4,200 |
| 6 · flatten | `(B, 192)` | 0 — reshape only |
| 7 · head | `(B, 288)` | 192 × 288 = 55,584 |
| | **total** | **71,840** |


### How the weighted average is built

Each token produces three vectors from its own 32 numbers, using three small
learned matrices:

- **query** — what am I looking for
- **key** — what do I contain
- **value** — what I would contribute

To update token 5: take its query, dot-product against every token's key, giving
6 raw scores. Push those through softmax so they are positive and sum to 1 —
those are the weights. Token 5's new value is the sum of all 6 **value** vectors,
each multiplied by its weight. Repeat for every token.

---

## The sweep

**One axis, four points.** Everything else — `d_model`, `n_heads`, `n_layers`,
`lookback` — was never touched.

| `patch_len` | tokens | params | score (MW) | Δ vs ref | `fold_std` |
|---|---|---|---|---|---|
| 1 (vanilla) | 144 | 1,336,288 | 48.742 | −1.693 | 4.59 |
| 6 | 24 | 231,968 | 50.436 | — (ref) | 6.84 |
| 12 (old default) | 12 | 123,296 | 48.225 | −2.211 | 4.69 |
| **24** | **6** | **71,840** | **46.171** | **−4.265** | 5.63 |


## Parameters

| parameter | default | what it is |
|---|---|---|
| `LOOKBACK` | 144 | Same window every other neural model gets |
| `PATCH_LEN` | **24** | Steps per token — 24 → 6 tokens. **The only swept axis** |
| `D_MODEL` | 32 | Width of each token's vector |
| `N_HEADS` | 2 | Parallel attention patterns over the same tokens |
| `N_LAYERS` | 1 | Encoder blocks |
| `D_FF` | 64 | Feedforward width inside the block |
| `DROPOUT` | 0.1 | |
| `SCOPE` | `global` | One model for all 134 turbines |
| `CLIP_OUTPUT` | `True` | Clamp predictions to [0, 1500] kW |


## Cost

The cheapest neural sweep in the suite — 4 points × 5 folds = 20 evaluations, and
the model itself is tiny at 72k parameters.

Under folds each fold needs its own checkpoint:

```bash
export SDWPF_TST_CKPT="baselines/transformer/checkpoints/transformer_{fold}.pt"
```

A path without `{fold}` scored across folds would load one model for all five —
four of which trained on their own validation targets. The code raises rather
than let that happen.
