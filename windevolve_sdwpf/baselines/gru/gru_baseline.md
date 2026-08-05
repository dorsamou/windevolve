# SDWPF GRU Baseline

Reference for `gru_baseline.py` (PyTorch port of the official Baidu KDD Cup 2022 GRU baseline).

> **This port reproduces the official code's behaviour, including several of its
> defects.** That is deliberate: the point of a baseline is comparability with
> the published number, not to be the best loop we could write. Section 6 lists
> every deviation from a "normal" PyTorch implementation, why it is there, and
> how each was verified against the official Paddle code. If something below
> looks wrong to you, check Section 6 before changing it — it is probably wrong
> on purpose.

---

## 1. Summary

The baseline trains **134 completely independent GRU models — one per turbine**. Each model sees only its own turbine's history, gets its own normalization statistics, its own early stopping, and its own saved file on disk. At forecast time, all 134 models are loaded once into memory; for each test window, every model reads the last 24 hours of its own turbine's data and emits that turbine's next 48 hours of power. Nothing is shared between turbines at any point.

Two consequences of matching the official code that will surprise you if you don't expect them:

- **Forecasting is random.** Dropout stays switched on at prediction time, because the official `predict.py` never turns it off. Calling `forecast()` twice on the same input gives two different answers. Score the model several times and report mean ± std.
- **All 134 turbines must be trained in one process.** The official seeds the random number generator *once*, before the turbine loop, so turbine 100's starting weights depend on turbines 0–99 having drawn from the generator first. Training a subset, or resuming an interrupted run, produces different models. The script allows it for smoke tests but stamps those checkpoints as partial and warns whenever they are loaded.

---

## 2. Hyperparameters, and what each one means

### Architecture

| Parameter | Value | What it means |
|---|---|---|
| `in_var` | 10 | Input variables per timestep: `Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Pab2, Pab3, Prtv, Patv`. (`TurbID/Day/Tmstamp` are labels, not inputs — `start_col=3` skips them.) |
| `hidden size` | 48 | Size of the GRU's memory vector. The model compresses "everything relevant so far" into 48 numbers and updates them at every timestep. Larger = more capacity to remember, more parameters, more overfitting risk. |
| `lstm_layer` | 2 | Two stacked GRU layers. Layer 1 reads the raw inputs; layer 2 reads layer 1's output. Stacking lets the model build higher-level temporal features. |
| `dropout` | 0.05 | Randomly zero 5% of activations before the projection. Normally a training-only trick, **but the official code leaves it on during validation and prediction too** — see Section 6, deviations 1 and 2. |
| `out_var` | 1 | One predicted variable: `Patv` (active power). |
| `input_len` | 144 | Steps of history fed in = 24 hours at 10-min resolution. |
| `output_len` | 288 | Steps predicted = 48 hours. |

**22,801 parameters per turbine** — tiny by modern standards:

| tensor | shape | count |
|---|---|---|
| `lstm.weight_ih_l0` | (144, 10) | 1,440 |
| `lstm.weight_hh_l0` | (144, 48) | 6,912 |
| `lstm.weight_ih_l1` | (144, 48) | 6,912 |
| `lstm.weight_hh_l1` | (144, 48) | 6,912 |
| biases (4 × 144) | | 576 |
| `projection` | (1, 48) + 1 | 49 |

(The 144 = 3 × 48: a GRU keeps three gates per hidden unit — reset, update, and candidate.)

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
| gradient clipping | 50.0, **per parameter** | Caps how big any single update can be, so one freak batch can't wreck the weights. "Per parameter" is unusual — see Section 6, deviation 5. |
| seed | 3407, **set once** | Not per turbine. See Section 6, deviation 7. |
| splits | 214 / 31 | Days of training / validation, out of the 245-day file. |

---

## 3. Training algorithm, step by step

### Phase A — once, before any turbine

**A1. Seed the random number generator.**
`random`, `numpy` and `torch` are all seeded with **3407**, once, before the turbine loop begins — matching the official `fix_seed = 3407`.

Why this matters: a random number generator is a stream, not a dice roll. Each draw advances it. Because the official seeds once and then trains all 134 turbines in order, turbine 50's initial weights are whatever the stream had reached after 50 turbines' worth of draws. Train turbines 50–60 on their own and they get *different* starting weights — so the run is only reproducible end-to-end.

**A2. Load the training CSV.**
Read `sdwpf_245days_v1.csv` → keep columns 3–12 (the 10 modeled variables) → replace NaN with 0 → a `(4,727,520 × 10)` float array. Rows are turbine-major: turbine `t` occupies rows `t × 35280` to `(t+1) × 35280` (245 days × 144 steps).

Loaded **once** and reused by all 134 turbines. (Takes ~2 s.)

### Phase B — repeated for each turbine `tid`

**B1. Slice this turbine's rows and split by time.**
- Train = first **214 days** (30,816 steps)
- Validation = next **31 days** (4,464 steps), starting `input_len` steps early so its first window has full history

The split is strictly chronological — never shuffled — so no future data leaks into training.

No re-seeding happens here. See A1.

**B2. Fit the scaler on the training split only.**

```
mean = np.mean(train_rows)    # ONE scalar over ALL elements
std  = np.std(train_rows)
```

> **Note the quirk:** `np.mean()` on a 2-D array returns a *single* number, so all 10 columns share one mean and one std — not per-column statistics. This is what the official code does, and it is reproduced faithfully. It is arguably a weakness of the baseline (power ~hundreds and wind speed ~single digits get lumped into one distribution).

**Why normalize at all:** the raw columns span wildly different scales (power 0–1500, wind 0–25, direction ±720). Un-normalized, the large columns dominate the gradients purely because they are numerically bigger. Neural nets train best on inputs centered near 0 with spread near 1.

**B3. Normalize both splits** with that scaler: `(x − mean) / std`.

**B4. Build rolling windows.**
Slide across the split: each sample is 144 steps of input followed by the next 288 steps as target.
- Train: 30,816 − 144 − 288 + 1 = **30,385** overlapping windows
- Validation: (4,464 + 144) − 144 − 288 + 1 = **4,177** windows

**B5. Wrap in DataLoaders.**
- Train: `batch_size=32`, `shuffle=True` (decorrelates updates), `drop_last=True`
- Validation: `shuffle=True`, `drop_last=True` — **also shuffled**, matching the official. Since `drop_last` throws away the leftover partial batch, shuffling means a *different* random slice of validation windows is discarded each epoch. See Section 6, deviation 8.

**B6. Initialize** a fresh `BaselineGruModel`, an Adam optimizer at lr=1e-4, and MSE loss.

**B7. Train, up to 10 epochs.** Each epoch is two passes:

*Training pass* (weights update). For each batch:
1. **Forward:** pad the 144-step input with 288 zero steps → 432 steps total. The GRU reads all 432 in order, carrying its 48-number hidden state. Project each step through `Linear(48 → 1)`. Keep the last 288 outputs.
2. **Loss:** MSE between those 288 predictions and the true 288 `Patv` values.
3. **Backward:** `loss.backward()` computes, for every weight, which direction reduces the loss — and **adds** it to whatever gradient was already stored (deviation 3).
4. **Clip, step, clip, step:** gradients are clipped per parameter and the optimizer is applied — twice (deviations 4 and 5).

*Validation pass* (no weight updates). Same forward + loss with gradients disabled — but the model stays in training mode, so dropout is still firing (deviation 2). This makes the validation number noisy, which in turn makes early stopping noisy.

> **Why the zero-padding?** There is no separate decoder. The GRU builds memory over the 144 real steps, then coasts through 288 steps of zeros — no new information — and the projection at each of those steps is read off as that timestep's prediction. This is a crude approach to multi-step forecasting and is part of why the baseline is beatable.

**B8. Early stopping check** after each epoch:
- **Validation improved** → record the new best, reset the patience counter, and **snapshot the weights** (`.detach().cpu().clone()` — a true copy, not a live reference).
- **Did not improve** → increment the counter; at 3, `break`.

Tracking the best weights matters because **the final epoch is not necessarily the best**. If the model peaks at epoch 4 and degrades through 7, you want epoch 4.

Training loss falling while validation loss rises is the classic overfitting signature — exactly what early stopping catches. Bear in mind the validation signal here is noisy for two independent reasons (dropout is live, and a random slice of windows is dropped each epoch), so a single non-improving epoch means less than it normally would.

**B9. Decay the learning rate**: `lr × 0.5^(epoch−1)`.

**B10. Save the checkpoint** to `checkpoints/model_{tid}.pt` (~95 KB each):

```python
{
  "state_dict":    best_state,   # BEST weights, not final
  "scaler_mean":   mean,         # this turbine's normalization stats
  "scaler_std":    std,
  "settings":      settings,     # architecture + hyperparameters
  "full_run":      True,         # was this a complete 134-turbine pass?
  "best_val_loss": best_val,
}
```

> Storing the scaler *inside* the checkpoint is a deliberate improvement over the original (which re-reads the 4.7M-row CSV at inference just to recover two numbers). Mathematically identical, far faster — and the only change made purely for convenience rather than fidelity.

`full_run` records whether this checkpoint came from a single pass over all 134 turbines with an empty starting folder. It is `False` if `--turbines` was used, or if the run resumed an existing folder. `forecast()` warns when it loads a `False`.

### Phase C — orchestration

`main()` seeds once, loads the CSV once, then loops the turbines. It skips any whose checkpoint already exists (unless `--overwrite`) — but note that a resumed run no longer reproduces the official RNG stream, so it is stamped partial.

**For a real run: start from an empty folder and pass no `--turbines`.**

```bash
rm -rf baseline_models/gru/checkpoints
python baseline_models/gru/gru_baseline.py --data $DATA --device cuda
```

Check the header says `turbines : 0..133 (n=134)` and shows **no** `*** PARTIAL RUN ***` banner. Roughly 5 s per turbine-epoch on a modern GPU → under two hours for the full run.

For a quick pipeline check, `--turbines 0-0 --epochs 1 --checkpoints /tmp/gru_smoke` is fine; just don't report a score from it.

---

## 4. How models are stored and called

**On disk:** `checkpoints/model_0.pt` … `model_133.pt` — 134 files, ~95 KB each, ~13 MB total, trivial to copy back from a cluster.

**Loading (`_load_all`)** — happens **once per process**:

1. Guard: `if _CACHE["models"] is not None: return` — the module-level cache means models load once, not 142 times.
2. Pick device (CUDA → MPS → CPU).
3. For each turbine: load the blob → rebuild the architecture from the *saved* settings → `load_state_dict()` to pour in the weights → `.to(device)` → **`model.train()`**.
4. Store models + scaler stats in `_CACHE`, print the stochastic-inference notice, and warn about any partial checkpoints.

> **`model.train()` at inference is not a bug.** It is what makes dropout stay active, reproducing the official `predict.py`, which never calls `.eval()`. See Section 6, deviation 1. A normal PyTorch model would call `.eval()` here.

`map_location=device` lets you train on a GPU box and score on CPU without error.

---

## 5. Forecasting algorithm, step by step

Called once per test window; **142 windows per evaluation**.

**F1.** `_load_all(capacity)` — no-op after the first window (this is what makes 142 windows affordable).

**F2.** Optionally seed: if `SDWPF_GRU_INFER_SEED` is set, `torch.manual_seed()` is called so a given scoring run is repeatable. The official seeds nothing here, so by default two runs differ.

**F3.** Load the test window with `TestData`, which returns per-turbine arrays of the 10 modeled columns with NaN → 0.

**F4.** For each turbine `tid`, inside `torch.no_grad()` (no gradient graph → faster, less memory):

1. **Get history:** `seq, _ = test_x.get_turbine(tid)` → `(2016, 10)`, i.e. 14 days.
2. **Normalize** with *this turbine's* stored scaler: `(seq − mean) / std`.
3. **Truncate:** `seq[-input_len:]` → keep only the most recent **144** steps.
   > The model was trained on 144-step inputs, so it discards ~13 of the 14 available days. A place an evolved program could do better.
4. **Add batch axis:** `unsqueeze(0)` → `(1, 144, 10)`. PyTorch models always expect a batch dimension.
5. **Run the model** → `(1, 288, 1)` in normalized units. Dropout fires here, so this step is random.
6. **Inverse-transform:** `pred × std + mean` → back to kW. Essential, since the evaluator expects kW.

**F5.** Assemble into `(134, 288, 1)` and return.

Predictions are **not clipped at zero**, matching the official `predict.py`. Set `CLIP_NEGATIVE = True` to test that variant (a deviation from the baseline). Negative forecasts are normal, not a bug: about 27% of real `Patv` readings are slightly negative (idling turbines drawing power), and the official evaluator never inspects a prediction's sign.

**F6.** `evaluator.py` then applies the official masking, converts kW → MW, sums MAE/RMSE across the 134 turbines, averages over the windows, and reports ½(RMSE+MAE).

### Scoring a stochastic model

Because dropout is live, one score is one draw. Take several:

```bash
for s in 1 2 3 4 5; do
  SDWPF_GRU_INFER_SEED=$s python evaluator.py baseline_models/gru/gru_baseline.py
done
```

Report mean ± std. This is not a workaround — it is a more honest measurement than the official's single number, and it tells you how close a reproduction can possibly get. Measured on a fixed input, two calls differed by up to ~40 kW at a single turbine-timestep; much of that averages out across 288 steps and 134 turbines, but not all of it.

---

## 6. Deviations from a normal PyTorch implementation

Every item here exists because the official code does it. Each was verified by running the official submission (`kddcup22-sdwpf-evaluation/paddlepaddle/tests/test-1.zip`) under paddle 3.3.1 and comparing.

| # | What the official does | What a normal loop does | Verified by |
|---|---|---|---|
| 1 | **Predicts with dropout on** — `predict.py` never calls `.eval()`, and Paddle layers default to training mode | `.eval()` before predicting | paddle `.training` is `True` after `set_state_dict`; no `eval()` anywhere in `predict.py` |
| 2 | **Validates with dropout on** — `train.py` sets `model.train()` per epoch and its `val()` never switches back | `.eval()` for validation | same source, read directly |
| 3 | **Never zeroes gradients** — no `clear_grad()` anywhere, so the gradient buffer accumulates across every batch *and* every epoch | `optimizer.zero_grad()` each step | gradient norms grew 1.00×, 2.00×, 3.00×, 4.00× over four batches with the learning rate forced to 0 |
| 4 | **Applies the optimizer twice per batch** — `minimize(loss)` already updates in Paddle's eager mode, and the loop then calls `step()` as well | one `step()` | 2.00× the weight movement of a single step |
| 5 | **Clips gradients per parameter** (`ClipGradByNorm`) | `clip_grad_norm_` clips the *global* norm | with norms 34.6 and 173.2 at clip 50, the official left the first untouched and clipped the second to 50.0; the global rule would give 9.8 and 49.0 |
| 6 | **XavierNormal projection weights, zero bias** — Paddle's `nn.Linear` default | `U(±1/√fan_in)` for weight *and* bias | port init std 0.20564 vs XavierNormal's 0.20203; PyTorch's default would be 0.08333 |
| 7 | **One global seed (3407)**, set before the turbine loop | seed per turbine | official `train.py` |
| 8 | **Shuffles the validation loader** with `drop_last=True` | no shuffling when only measuring | official `get_data()` passes `shuffle=True` for both splits |

Two things that look like differences but are not, both checked rather than assumed:

- `batch_first=True` (here) vs `time_major=True` + explicit transposes (official) — **bit-identical output**, max difference `0.000e+00`.
- GRU weight and bias initialisation — both frameworks default to `U(±1/√hidden)`, measured std 0.0833 in each.

### Verification

The official Paddle code produced reference artifacts (weights, inputs, outputs, per-batch weight trajectories); Paddle was then removed, PyTorch installed, and those artifacts replayed through this file's own functions:

| check | result |
|---|---|
| forward pass, identical weights | max difference `4.470e-08` |
| `run_epoch` over 6 batches, identical init and data | mean loss differs by `9.9e-09`; worst weight difference `5.066e-07` |
| gradient buffer left behind (deviation 3) | port `1.159312`, official `1.159312` |
| inference noise magnitude (deviation 1) | per-element std 0.0395 vs official 0.0377 |
| projection init (deviation 6) | 0.20564 vs XavierNormal 0.20203 |
| split borders, scaler, window counts | identical to the digit |

**What this establishes:** the same model and the same procedure, to float32 precision, on every parameter.

**What it does not:** identical *weights*. PyTorch and Paddle have different RNG algorithms and different DataLoader shuffling, so the two runs diverge at initialisation and never re-converge. The procedure is reproducible; the exact arithmetic path through it is not.

---

## 7. Which published number to compare against

Two numbers circulate for "the official GRU baseline". They are not two estimates of one quantity.

| | **47.850** | **42.319760** |
|---|---|---|
| source | Zhou et al., *Scientific Data* 11:649 (2024), Fig. 3 | Zhou et al., arXiv:2208.04360, §5 |
| evaluated on | the official competition test set | the validation tail of the public 245-day file |
| windows | 142 | 195 |
| scored by | the submission grader (`kddcup22-sdwpf-evaluation/paddlepaddle/metrics.py`) | the development scorer (`wpf_baseline_gru/metrics.py`) |
| comparable to | the leaderboard | nothing published |

**Target 47.850.** It is printed alongside the top-ten team scores (HIK 44.917 … SlienceGTeam 45.327), which are leaderboard results on the 142-window test set — so it sits on the same ruler as `competition_score.py`.

42.32 is the baseline authors' own development number. It is lower not because the model is better there, but because the validation tail is easier *and* the development scorer is more lenient — it has no NaN condition in its mask and reads about 4.5% lower on identical inputs.

One caution when reading results: a score from `evaluator.py` is a **validation** number and is not comparable with 47.850, which is a **test** number. Only `competition_score.py` on `final_phase_test` produces a like-for-like comparison.
