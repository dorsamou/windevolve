# BUAA_BIGSCity — AGCRN + MTGNN ensemble (rank 11, Baidu KDD Cup 2022)

Team BUAA_BIGSCity placed 11th of 2490 with a six-model ensemble of two
spatio-temporal graph neural networks. They released their **actual submission
artefact** — `submit.zip`, with trained checkpoints, precomputed graphs,
normalisation statistics and the fusion rule — so this entry runs inference
only. No training is required to reproduce their score.

## Reproduction — confirmed

Scored through `evaluation/competition_score.py` against `final_phase_test`:

```
total_mae         : 41.150749
total_rmse        : 49.569786
competition_score : 45.360267
latest-24h mae    : 43.019587
latest-24h rmse   : 50.789902
accuracy          : 62.0971%
windows           : 142
```

Published leaderboard score: **-45.36026**. Reproduced: **45.360267**. Match to
six significant figures.

Because their prediction is entirely deterministic — frozen weights, frozen
fusion, no sampling — this doubles as an end-to-end validation of the scorer.
The exact match means `competition_score.py` is equivalent to Baidu's
`evaluation.py` on a real submission, so every other score in this repo is
leaderboard-comparable by construction.

## The two architectures

Both are spatio-temporal GNNs, but with almost opposite inductive biases, which
is why ensembling them pays.

### AGCRN — Adaptive Graph Convolutional Recurrent Network

*Bai et al., NeurIPS 2020.* A GRU in which every dense matrix multiply is
replaced by a graph convolution. Two ideas carry it:

- **Node-adaptive parameter learning.** Rather than one weight matrix shared by
  all 134 turbines (too rigid — turbines differ) or 134 independent ones (too
  many parameters), it learns a node embedding matrix `E ∈ R^(134×10)` and a
  shared low-rank parameter pool. Each turbine's weights are `E_i · W_pool`, so
  turbines with similar embeddings behave similarly and the parameter count
  stays flat in the number of turbines.
- **Data-adaptive graph generation.** The adjacency is learned as
  `softmax(relu(E · Eᵀ))` from those same embeddings, rather than fixed in
  advance.

BIGSCity keep the learned adjacency (`add_apt=True`) *and* add a predefined
DTW graph as a normalised Laplacian — the model gets both a learned and a
prescribed notion of which turbines resemble each other.

The encoder is two AGCRN cells stacked, run over the 144 input steps. Only the
final hidden state is kept; a `Conv2d` with kernel `(1, hidden_dim)` then
projects it to all 288 output steps at once. **The decoder is single-shot, not
autoregressive** — no error accumulation over the horizon, and no possibility
of conditioning later steps on earlier predictions.

### MTGNN — Multivariate Time Series Graph Neural Network

*Wu et al., KDD 2020.* Convolutional rather than recurrent. It interleaves
temporal convolution modules (dilated inception layers with kernels 2/3/6/7)
with graph convolution modules (mix-hop propagation), joined by residual and
skip connections. Dilation grows as `2^layer`, giving a receptive field of 43
steps over the 144-step input.

MTGNN normally learns its graph too, but here `add_apt=False`: it uses only the
**geographic** graph, built by Gaussian-kernelling the pairwise distance
matrix, thresholding at 0.8, and binarising.

### Why they complement each other

| | AGCRN | MTGNN |
|---|---|---|
| temporal | recurrent, unbounded context | dilated conv, 43-step receptive field |
| spatial | DTW similarity + learned adjacency | fixed geographic adjacency |
| structure | who *behaves* alike | who *is* nearby |

Wake effects make geography informative; correlated weather exposure makes
behavioural similarity informative. Neither graph dominates, so averaging the
two model families helps more than adding capacity to either.

## The ensemble

| # | model | graph | `var_len` | `add_apt` | weight |
|---|---|---|---|---|---|
| 1 | AGCRN | DTW top-5, fold 0 | 6 | yes | 0.321763283 |
| 2 | AGCRN | DTW top-5, fold 1 | 6 | yes | 0.495282746 |
| 3 | AGCRN | DTW top-5, fold 2 | 6 | yes | 0.356973800 |
| 4 | AGCRN | DTW top-5, fold 3 | 6 | yes | 0.298808416 |
| 5 | AGCRN | DTW top-5, fold 4 | 6 | yes | 0.408473526 |
| 6 | MTGNN | geo, thresholded, binarised | 5 | no | 1.0 |

The five AGCRNs come from 5-fold cross-validation, each with its own DTW graph
built from that fold's training data. They are averaged with weights
proportional to `1/w`, where `w` is the fold's validation loss — so a better
fold pulls harder. MTGNN was trained once on the 214/31-day split and stands
alone. The two are blended `0.4 · AGCRN + 0.6 · MTGNN`, with the majority
share on the single model rather than the five-model average.

## Features

Selected: `weekday, time, Wspd, Etmp, Itmp, Prtv, Patv`. Both models discard
`weekday` and `time` inside their own `forward` (`inputs[:, :, :, 2:]`), so
five channels actually reach the network. AGCRN appends a sixth — the first
difference of normalised `Patv` — which is why its `var_len` is 6.

`Wdir`, `Ndir` and `Pab1-3` are dropped entirely. Output passes through ReLU,
clipping negative power to zero.

## FilterHuberLoss — the trick that matters most

Training minimises Huber loss (`delta=5`) with a mask that zeroes out exactly
the records the **official grader** excludes:

- `Patv < 0`
- any of `Pab1/2/3 > 89`
- `Wdir` outside ±180, or `Ndir` outside ±720
- `Patv == 0` while `Wspd > 2.5` (turbine curtailed or stopped — the output is
  unknowable, not zero)

Training on the same mask you are scored on is shared by essentially every top
team, and is worth more than any architectural choice at this point on the
leaderboard. Predicting curtailed periods well earns nothing; letting them
distort the fit costs a great deal.

One detail: the reduction is `mean(loss * mask)`, dividing by the *total*
element count rather than the valid count. Masked points therefore shrink the
gradient magnitude instead of being cleanly excluded. Probably unintended,
certainly present in the run that scored -45.36.

## Two quirks reproduced verbatim

**Normalisation is per-turbine, not per-feature.** `data_mean_214.npy` and
`data_scale_214.npy` are shape `(1, 134, 1, 1)` — one statistic per turbine,
broadcast across all five channels. Wind speed and both temperatures are
standardised using *active power's* mean and standard deviation, putting them
on an arbitrary scale.

**The k-fold models reuse the full-data statistics.** `forecast()` loads
`data_mean_214.npy` for every model, including the five fold models whose own
`data_mean_{ind}_5.npy` files sit unused in `npy/`.

Neither is corrected here. Both are part of what produced the published score,
and both are informative: two normalisation defects cost them nothing
measurable, which says the metric is far less sensitive to preprocessing
detail than to what the ensemble averages over. Worth remembering when
choosing what a search loop is allowed to mutate.

## Setup

Vendored artefacts are ~41 MB and not committed. Populate from the released
zip:

```bash
cd leaderboard/bigscity
mkdir -p vendor && cd vendor
unzip /path/to/submit.zip
mv submit/* . && rmdir submit
rm -rf __MACOSX
```

Only `AGCRN.py`, `MTGNN.py`, `npy/` and `kfold_dtw_5_data_diff/` are used; the
rest of their repo (training loop, dataset, losses) is unused here. Add
`leaderboard/bigscity/vendor/` to `.gitignore`.

## Run

```bash
SDWPF_DEVICE=cuda:0 python evaluation/competition_score.py \
    leaderboard/bigscity/bigscity_ensemble.py
```

Roughly 19 s per window on CPU (~45 min for 142 windows), far less on GPU. The
cost is AGCRN's sequential RNN over 144 steps × 134 nodes, run five times.

## Using this as a seed program

The ensemble is a poor starting point for a search loop as-is: six frozen
checkpoints with no training code, so a mutation can only touch fusion weights
and postprocessing. To use it as an initial program you would want to port
`main.py` / `main_kfold.py` as well and let the loop train, or seed with a
single AGCRN and treat the ensemble as the target to beat.

Its more immediate value is as a **fixed reference point**: a deterministic,
leaderboard-anchored score that any candidate can be compared against without
re-running a competitor.

## Reference

Jiang, Han & Wang. *BUAA_BIGSCity: Spatial-Temporal Graph Neural Network for
Wind Power Forecasting in Baidu KDD CUP 2022.* arXiv:2302.11159.
Code and submission artefact: github.com/BUAABIGSCity/KDDCUP2022

Bai et al. *Adaptive Graph Convolutional Recurrent Network for Traffic
Forecasting.* NeurIPS 2020. (AGCRN)

Wu et al. *Connecting the Dots: Multivariate Time Series Forecasting with Graph
Neural Networks.* KDD 2020. (MTGNN)