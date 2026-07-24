# SDWPF GRU Baseline 

Reference for `gru_baseline.py` (PyTorch port of the official Baidu KDD Cup 2022 GRU baseline).

---

## 1. Summary

The baseline trains **134 completely independent GRU models — one per turbine**. Each model sees only its own turbine's history, gets its own normalization statistics, its own early stopping, and its own saved file on disk. At forecast time, all 134 models are loaded once into memory; for each test window, every model reads the last 24 hours of its own turbine's data and emits that turbine's next 48 hours of power. Nothing is shared between turbines at any point.

---

## 2. Hyperparameters, and what each one means

### Architecture

| Parameter | Value | What it means |
|---|---|---|
| `in_var` | 10 | Input variables per timestep: `Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, Patv`. (`TurbID/Day/Tmstamp` are labels, not inputs — `start_col=3` skips them.) |
| `hidden size` | 48 | Size of the GRU's memory vector. The model compresses "everything relevant so far" into 48 numbers and updates them at every timestep. Larger = more capacity to remember, more parameters, more overfitting risk. |
| `lstm_layer` | 2 | Two stacked GRU layers. Layer 1 reads the raw inputs; layer 2 reads layer 1's output. Stacking lets the model build higher-level temporal features. |
| `dropout` | 0.05 | During training, randomly zero 5% of activations before the projection. Forces the model not to depend on any single pathway → regularization. **Switched off at inference** via `model.eval()`. |
| `out_var` | 1 | One predicted variable: `Patv` (active power). |
| `input_len` | 144 | Steps of history fed in = 24 hours at 10-min resolution. |
| `output_len` | 288 | Steps predicted = 48 hours. |

Total: roughly 20k parameters per turbine — a small model by modern standards.

### Training

| Parameter | Value | What it means |
|---|---|---|
| loss | MSE | Mean Squared Error. Squares each error, so large misses are punished disproportionately. (Distinct from the *evaluation* metric, ½(RMSE+MAE).) |
| optimizer | Adam | Adaptive optimizer; maintains a running memory of gradients and gives each weight its own effective step size. Standard default. |
| `lr` | 1e-4 | Learning rate = step size for weight updates. |
| lr schedule | `lr × 0.5^(epoch−1)` | Halves every epoch. Big steps early to find the right region, fine steps later to settle. By epoch 10 the rate is ~1/500th of the original. |
| `batch_size` | 32 | Windows processed together per update. Faster than one-at-a-time and gives more stable gradients. |
| `train_epochs` | 10 | Maximum passes over the training data. |
| `patience` | 3 | Stop early after 3 consecutive epochs with no validation improvement. |
| splits | 214 / 31 | Days of training / validation, out of the 245-day file. |

---

## 3. Training algorithm, step by step

### Phase A — once, before any turbine

**A1. Load the training CSV.**
Read `sdwpf_245days_v1.csv` → keep columns 3–12 (the 10 modeled variables) → replace NaN with 0 → a `(4,727,520 × 10)` float array. Rows are turbine-major: turbine `t` occupies rows `t × 35280` to `(t+1) × 35280` (245 days × 144 steps).

Loaded **once** and reused by all 134 turbines.

### Phase B — repeated for each turbine `tid`

**B1. Seed the RNG.**
`torch.manual_seed(seed + tid)` so each turbine's weight initialization is reproducible and distinct.

**B2. Slice this turbine's rows and split by time.**
- Train = first **214 days** (30,816 steps)
- Validation = next **31 days** (4,464 steps), starting `input_len` steps early so its first window has full history

The split is strictly chronological — never shuffled — so no future data leaks into training.

**B3. Fit the scaler on the training split only.**

```
mean = np.mean(train_rows)    # ONE scalar over ALL elements
std  = np.std(train_rows)
```

> **Note the quirk:** `np.mean()` on a 2-D array returns a *single* number, so all 10 columns share one mean and one std — not per-column statistics. This is what the official code does, and it is reproduced faithfully. It is arguably a weakness of the baseline (power ~hundreds and wind speed ~single digits get lumped into one distribution).

**Why normalize at all:** the raw columns span wildly different scales (power 0–1500, wind 0–25, direction ±720). Un-normalized, the large columns dominate the gradients purely because they are numerically bigger. Neural nets train best on inputs centered near 0 with spread near 1.

**B4. Normalize both splits** with that scaler: `(x − mean) / std`.

**B5. Build rolling windows.**
Slide across the split: each sample is 144 steps of input followed by the next 288 steps as target. From ~30,816 training steps this yields ~30,385 overlapping windows.

**B6. Wrap in DataLoaders.**
- Train: `batch_size=32`, `shuffle=True` (decorrelates updates), `drop_last=True`
- Validation: `shuffle=False` (no need to randomize when only measuring)

**B7. Initialize** a fresh `BaselineGruModel`, an Adam optimizer at lr=1e-4, and MSE loss.

**B8. Train, up to 10 epochs.** Each epoch is two passes:

*Training pass* (weights update). For each batch:
1. **Forward:** pad the 144-step input with 288 zero steps → 432 steps total. The GRU reads all 432 in order, carrying its 48-number hidden state. Project each step through `Linear(48 → 1)`. Keep the last 288 outputs.
2. **Loss:** MSE between those 288 predictions and the true 288 `Patv` values.
3. **Backward:** `loss.backward()` computes, for every weight, which direction reduces the loss.
4. **Update:** `optimizer.step()` nudges the weights.

*Validation pass* (no updates). Same forward + loss with gradients disabled — measurement only.

> **Why the zero-padding?** There is no separate decoder. The GRU builds memory over the 144 real steps, then coasts through 288 steps of zeros — no new information — and the projection at each of those steps is read off as that timestep's prediction. This is a crude approach to multi-step forecasting and is part of why the baseline is beatable.

**B9. Early stopping check** after each epoch:
- **Validation improved** → record the new best, reset the patience counter, and **snapshot the weights** (`.detach().cpu().clone()` — a true copy, not a live reference).
- **Did not improve** → increment the counter; at 3, `break`.

Tracking the best weights matters because **the final epoch is not necessarily the best**. If the model peaks at epoch 4 and degrades through 7, you want epoch 4.

Training loss falling while validation loss rises is the classic overfitting signature — exactly what early stopping catches.

**B10. Decay the learning rate**: `lr × 0.5^(epoch−1)`.

**B11. Save the checkpoint** to `checkpoints/model_{tid}.pt`:

```python
{
  "state_dict":    best_state,   # BEST weights, not final
  "scaler_mean":   mean,         # this turbine's normalization stats
  "scaler_std":    std,
  "settings":      settings,     # architecture + hyperparameters
  "best_val_loss": best_val,
}
```

> Storing the scaler *inside* the checkpoint is a deliberate improvement over the original (which re-reads the 4.7M-row CSV at inference just to recover two numbers). Mathematically identical, far faster.

### Phase C — orchestration

`main()` loops the selected turbines and **skips any whose checkpoint already exists** (unless `--overwrite`). So an interrupted run resumes where it left off, and you can shard across machines and merge the folders.

---

## 4. How models are stored and called

**On disk:** `checkpoints/model_0.pt` … `model_133.pt` — 134 files, each ~100 KB. All 134 together are only ~10–15 MB, so they are trivial to copy back from a cluster.

**Loading (`_load_all`)** — happens **once per process**:

1. Guard: `if _CACHE["models"] is not None: return` — the module-level cache means models load once, not 142 times.
2. Pick device (CUDA → MPS → CPU).
3. For each turbine: load the blob → rebuild the architecture from the *saved* settings → `load_state_dict()` to pour in the weights → `.to(device)` → **`model.eval()`** (disables dropout; forgetting this makes predictions randomly noisy).
4. Store models + scaler stats in `_CACHE`.

`map_location=device` lets you train on a GPU box and score on CPU without error.

---

## 5. Forecasting algorithm, step by step

Called once per test window; **142 windows per evaluation**.

**F1.** `_load_all(capacity)` — no-op after the first window (this is what makes 142 windows affordable: ~2.2s for the first, ~2.05s thereafter).

**F2.** Load the test window with `TestData`, which returns per-turbine arrays of the 10 modeled columns with NaN → 0.

**F3.** For each turbine `tid`, inside `torch.no_grad()` (no gradient graph → faster, less memory):

1. **Get history:** `seq, _ = test_x.get_turbine(tid)` → `(2016, 10)`, i.e. 14 days.
2. **Normalize** with *this turbine's* stored scaler: `(seq − mean) / std`.
3. **Truncate:** `seq[-input_len:]` → keep only the most recent **144** steps.
   > The model was trained on 144-step inputs, so it discards ~13 of the 14 available days. A place an evolved program could do better.
4. **Add batch axis:** `unsqueeze(0)` → `(1, 144, 10)`. PyTorch models always expect a batch dimension.
5. **Run the model** → `(1, 288, 1)` in normalized units.
6. **Inverse-transform:** `pred × std + mean` → back to kW. Essential, since the evaluator expects kW.

**F4.** Assemble into `(134, 288, 1)` and return.

Predictions are **not clipped at zero**, matching the official `predict.py`. Set `CLIP_NEGATIVE = True` to test that variant (a deviation from the baseline).

**F5.** `evaluator.py` then applies the official masking, converts kW → MW, sums MAE/RMSE across the 134 turbines, averages over the 142 windows, and reports ½(RMSE+MAE).

---
