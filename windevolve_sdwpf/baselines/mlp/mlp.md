# MLP Baseline

Reference for `baselines/mlp/mlp.py` — **M**ulti-**L**ayer **P**erceptron.

> **CHAMPION — `scope = per_turbine`, 45.409 MW (5-fold CV).**
>
> **SHIPPED DEFAULT — `scope = global`, 46.598 MW.** The default deliberately
> differs from the champion, on cost.

---

## The whole model

```
flatten the 144-step window into one vector    (1,440 numbers)
  → Linear → GELU → dropout       (512 units)
  → Linear → GELU → dropout       (512 units)
  → Linear                        (288 outputs)
```
 **no notion that the inputs are ordered in time**.

### How one layer works

**One unit** holds a weight for every input, multiplies, sums, adds a bias:

```
sum = w₁·x₁ + w₂·x₂ + ... + w₁₄₄₀·x₁₄₄₀ + b
```

Then GELU bends it. **One layer is 512 such units** seeing the same inputs with
different weights. **Stack two**, then a final layer of 288 with no activation —
those 288 numbers *are* the forecast.

**Without the activation, stacked layers collapse into a single linear map.** It
is the only reason depth buys anything.

| connection | shape | weights |
|---|---|---|
| input → hidden 1 | 1,440 × 512 | 0.74M |
| hidden 1 → hidden 2 | 512 × 512 | 0.26M |
| hidden 2 → output | 512 × 288 | 0.15M |
| | **total** | **1.15M** |

---

## Why it is in the suite

The MLP is a **control**. It isolates *nonlinearity* from *architecture*:

| model | nonlinear? | time structure? |
|---|---|---|
| climatology | no | none |
| **MLP** | **yes** | **none** |
| GRU / LSTM | yes | sequential |
| Transformer | yes | attention |


## The sweep

| axis | value | score (MW) | Δ vs ref | `fold_std` |
|---|---|---|---|---|
| reference | global, hidden 512, dropout 0.1 | 46.598 | — | 5.53 |
| **scope** | **per turbine** | **45.409** | **−1.189** | **4.84** |
| hidden | 128 | 47.032 | +0.434 | 5.89 |
| dropout | 0.3 | 46.604 | +0.006 | 5.89 |

### `scope` — and it disagrees with the other two families

| model | per-turbine penalty |
|---|---|
| gru_global | +1.249 |
| gbdt | +1.162 |
| **mlp** | **−1.189** |



### Why the default is not the champion

`SCOPE = "global"` ships, even though `per_turbine` won by 1.189 MW.

**Cost.** Per-turbine trains **134 models per fold** — 670 trainings for a 5-fold
run, against 5 for global.

```python
SCOPE = "per_turbine"
```


## Parameters

| parameter | default | what it is |
|---|---|---|
| `LOOKBACK` | 144 | History steps flattened into the input vector |
| `SCOPE` | **`global`** | One model, or 134. Champion was `per_turbine`; global ships on cost |
| `HIDDEN` | 512 | Units in each hidden layer |
| `DEPTH` | 2 | Hidden layers stacked |
| `ACTIVATION` | `gelu` | `relu` / `gelu` / `tanh` |
| `DROPOUT` | 0.1 | Units zeroed at random each step; off at prediction |
| `LOSS` | `mse` | Matches the GRU and the competition solutions |
| `CLIP_OUTPUT` | `True` | Clamp predictions to [0, 1500] kW |


## Cost

As shipped (`global`): a few minutes per fold, under an hour for 5 folds.

The champion configuration is far more expensive — `scope=per_turbine` trains
**134 models per fold**, so 670 trainings for one grid point. It ran on its own
overnight alongside the other scope points, and that cost is why it is not the
default.

Under folds each fold needs its own checkpoint:

```bash
export SDWPF_MLP_CKPT="baselines/mlp/checkpoints/mlp_{fold}.pt"
```
