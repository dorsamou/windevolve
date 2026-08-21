"""
path: windevolve_sdwpf/baselines/neural/lstm.py

Description:
    Long Short-Term Memory encoder with a flat projection head. Walks the
    window one step at a time carrying TWO summaries, then emits all 288
    horizon steps in one projection from the final hidden state.

    THE TWO STATES. "48 units" does not mean 48 numbers in total -- it means
    the state is 48 slots wide, and in an LSTM each slot carries two numbers.
    The CELL state is the long-term ledger: the forget gate multiplies it
    elementwise by a value in (0,1), then the input gate adds a bounded
    contribution, so it is EDITED rather than rewritten. The HIDDEN state is
    the cell state passed through tanh and filtered by the output gate, and it
    is the only one the next layer and the next timestep see.

    At each step every unit reads the 10 current channels AND all 48 previous
    hidden values -- 58 numbers in. Weights come in pairs accordingly:
    48 x 10 for the input and 48 x 48 for the recurrence, times four gates.

    WHY IT IS HERE. Every setting is pinned to the GRU's -- 144 lookback,
    48 hidden, 2 layers, dropout 0.05, MSE -- so the ONLY difference between
    the two models is the cell. That makes it a controlled comparison rather
    than two loosely related models, which is why it gets no sweep of its own.

Citation:
    Hochreiter and Schmidhuber, "Long Short-Term Memory", Neural Computation
    9(8), 1997. Settings mirror the official SDWPF GRU baseline
    (PaddlePaddle/PaddleSpatial, apps/wpf_baseline_gru), independently
    confirmed by Team 88VIP (arXiv:2208.08952, Table 3).
    Dataset: Zhou et al., arXiv:2208.04360 (2022).

Example usage:
    # from the package root (windevolve_sdwpf/)
    python baselines/neural/lstm.py --epochs 2 --ckpt /tmp/lstm_smoke.pt

    for F in F1 F2 F3 F4 F5; do
      python baselines/neural/lstm.py \
          --train-csv evaluation/splits_rolling_5x30/$F/fold_train.csv \
          --ckpt baselines/neural/checkpoints/lstm_$F.pt
    done

    export SDWPF_LSTM_CKPT="baselines/neural/checkpoints/lstm_{fold}.pt"
    python evaluation/evaluator.py baselines/neural/lstm.py \
        --folds-root evaluation/splits_rolling_5x30
"""
import os                                    # paths, env vars
import sys                                   # package-root discovery
import time                                  # timing
import random                                # seeding
import argparse                              # training CLI

import numpy as np                           # array math
import pandas as pd                          # CSV reading
import torch                                 # tensors
import torch.nn as nn                        # layers and loss
from torch.utils.data import Dataset, DataLoader

# baselines/neural/ -> baselines/ -> repo root. Added so this file works when
# run directly, when imported by the sweep harness, when loaded by
# evaluator.py, and when handed to an evolve arm.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_data import TestData               # official per-window loader
from paths import VAL_TRAIN_FILE

# ===========================================================================
# SHARED PIPELINE -- data, scaler, windowing, training, inference.
#
# Duplicated verbatim across mlp.py / lstm.py / transformer.py so each file is
# a complete, independently-evolvable program. Keep the copies in sync by hand;
# a divergence here silently makes their scores incomparable.
# ===========================================================================

CAPACITY = 134          # turbines in the farm
DAY_LEN = 144           # 10-minute observations per day
OUTPUT_LEN = 288        # forecast horizon (48h) -- immutable
START_COL = 3           # first modeled column (Wspd)
N_FEATURES = 10         # Wspd Wdir Etmp Itmp Ndir Pab1 Pab2 Pab3 Prtv Patv
PATV_IDX = 9            # index of Patv within the modeled columns -- the target
MAX_LOOKBACK = 2016     # 14 days: what each window infile supplies


def resolve_fold_path(envs, template):
    """
    Description:
        Fill "{fold}" in a checkpoint path from envs. Under a rolling-origin
        layout each fold trains on its own prefix, so each fold needs its own
        weights. A path without the placeholder scored across folds would load
        one model for all five -- four of which trained on their own validation
        targets. That is a leak, so this raises instead.
    Args:
        envs: settings dict passed to forecast() (may be None)
        template: configured path, optionally containing "{fold}"
    Returns:
        The resolved filesystem path.
    """
    if "{fold}" not in template:
        return template
    fold = (envs or {}).get("fold")
    if not fold:
        raise ValueError(
            "path {!r} contains '{{fold}}' but the evaluator supplied no fold. "
            "Use --folds-root, or set a path without the placeholder."
            .format(template))
    return template.format(fold=fold)


def load_farm_array(path, capacity=CAPACITY, start_col=START_COL):
    """
    Description:
        Read a training CSV into a per-turbine array. NaN becomes 0, matching
        test_data.TestData, so training inputs and scoring inputs are treated
        identically -- a mismatch there is train/serve skew no hyperparameter
        can fix.
    Args:
        path: CSV path, turbine-major
        capacity: turbines stacked in the file
        start_col: index of the first modeled column
    Returns:
        float32 array (capacity, steps_per_turbine, n_features).
    """
    df = pd.read_csv(path)
    if df.shape[0] % capacity:
        raise ValueError("{} has {} rows, not divisible by capacity {}"
                         .format(path, df.shape[0], capacity))
    steps = df.shape[0] // capacity
    arr = df[df.columns[start_col:]].replace(np.nan, 0).values.astype(np.float32)
    return arr.reshape(capacity, steps, -1)


class Scaler(object):
    """
    Description:
        Per-feature standardiser. Fitted across all turbines for scope=global
        so the same physical wind speed means the same thing everywhere, and
        per turbine for scope=per_turbine.

        Always fitted on the TRAINING PREFIX ONLY -- fitting on the whole file
        would leak the validation period's distribution into the normaliser.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arr):
        """Fit on (turbines, steps, features) or (steps, features)."""
        flat = arr.reshape(-1, arr.shape[-1])
        self.mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        self.std = np.where(std < 1e-6, 1.0, std)   # guard constant columns
        return self

    def transform(self, arr):
        return (arr - self.mean) / self.std

    def inverse_target(self, y):
        """Undo normalisation for the Patv channel only, returning kW."""
        return y * self.std[PATV_IDX] + self.mean[PATV_IDX]

    def state(self):
        return {"mean": np.asarray(self.mean), "std": np.asarray(self.std)}

    @classmethod
    def from_state(cls, st):
        s = cls()
        s.mean, s.std = np.asarray(st["mean"]), np.asarray(st["std"])
        return s


class WindowDataset(Dataset):
    """
    Description:
        (lookback, horizon) pairs. With turbines=None it pools every turbine
        (scope=global); with a single turbine index it yields that turbine only
        (scope=per_turbine).

        STRIDE keeps every Nth forecast origin. Consecutive origins differ by
        one 10-minute step and are almost the same example, so this is an
        epoch-size knob rather than an accuracy knob.
    """

    def __init__(self, arr, lookback, stride=6, turbine=None):
        self.arr = arr
        self.lookback = lookback
        n_turb, steps, _ = arr.shape
        last_origin = steps - OUTPUT_LEN
        starts = np.arange(lookback, last_origin, stride)
        if starts.size == 0:
            raise ValueError(
                "lookback {} + horizon {} does not fit in {} steps"
                .format(lookback, OUTPUT_LEN, steps))
        tids = range(n_turb) if turbine is None else [turbine]
        self.index = [(t, int(s)) for t in tids for s in starts]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        tid, origin = self.index[i]
        x = self.arr[tid, origin - self.lookback:origin]           # (L, F)
        y = self.arr[tid, origin:origin + OUTPUT_LEN, PATV_IDX]    # (H,)
        return (torch.from_numpy(np.ascontiguousarray(x)).float(),
                torch.from_numpy(np.ascontiguousarray(y)).float())


def split_prefix(arr, val_days):
    """
    Description:
        Carve an INNER validation tail off the training prefix, used only for
        early stopping. This is NOT the project's validation split -- that one
        lives in the fold directories and is never touched here. Two levels of
        holdout is the point: early stopping on the scored fold would tune the
        model on the data it is about to be measured on.
    Args:
        arr: (capacity, steps, n_features)
        val_days: days reserved at the end for early stopping
    Returns:
        (train_arr, inner_val_arr)
    """
    steps = arr.shape[1]
    cut = steps - val_days * DAY_LEN
    if cut <= 0:
        raise ValueError("val_days {} exceeds the {} steps available"
                         .format(val_days, steps))
    return arr[:, :cut], arr[:, cut:]


def seed_everything(seed):
    """Seed python, numpy and torch from one integer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loss(name):
    """MSE by default -- it is what the official GRU baseline and the
    competition solutions use. 'half' is 0.5*MSE + 0.5*L1, the objective
    matched to the competition's own 0.5*(MAE + RMSE) score."""
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "half":
        mse, l1 = nn.MSELoss(), nn.L1Loss()
        return lambda p, t: 0.5 * mse(p, t) + 0.5 * l1(p, t)
    raise ValueError("unknown loss {!r}".format(name))


def _fit_one(cfg, train_arr, inner_arr, n_in, device, turbine, verbose):
    """
    Description:
        Train a single model -- either the one global model, or one turbine's
        model when scope=per_turbine. Returns (state_dict, best_inner_loss).
    """
    ds_tr = WindowDataset(train_arr, cfg["lookback"], cfg["stride"], turbine)
    ds_va = WindowDataset(inner_arr, cfg["lookback"],
                          max(1, cfg["stride"] // 2), turbine)
    # drop_last only when there is more than one full batch. With
    # scope=per_turbine a single turbine can yield fewer windows than the batch
    # size, and drop_last would then silently produce ZERO batches -- the model
    # would "train" without a single gradient step and score as noise.
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                       num_workers=cfg["num_workers"],
                       drop_last=len(ds_tr) > cfg["batch_size"])
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                       num_workers=cfg["num_workers"])

    model = build_model(cfg, n_in).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg["lr_decay"])
    loss_fn = make_loss(cfg["loss"])

    if len(ds_tr) == 0:
        raise ValueError(
            "no training windows: lookback {} + horizon {} does not fit the "
            "available history at stride {}".format(
                cfg["lookback"], OUTPUT_LEN, cfg["stride"]))

    best, best_state, waited = float("inf"), None, 0
    for epoch in range(cfg["epochs"]):
        t0 = time.perf_counter()
        model.train()
        run = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
            opt.step()
            run += loss.item()
        tr = run / max(1, len(dl_tr))

        model.eval()
        run = 0.0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                run += loss_fn(model(x), y).item()
        va = run / max(1, len(dl_va))
        if len(dl_tr):
            sched.step()

        if verbose:
            print("[epoch {:>3}] train {:.5f}  inner-val {:.5f}  ({:.1f}s)"
                  .format(epoch, tr, va, time.perf_counter() - t0))

        if va < best - 1e-6:
            best, waited = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                if verbose:
                    print("[train] early stop at epoch {}".format(epoch))
                break
    return best_state, best


def train_model(cfg, csv_path=None, ckpt_path=None, device=None, verbose=True):
    """
    Description:
        Fit this baseline on one training prefix and write a checkpoint. The
        config travels with the weights so forecast() rebuilds the identical
        architecture without being told -- otherwise a sweep that changed
        `hidden` would load old weights into a new shape.

        scope=global    -> one model, all turbines pooled, one scaler
        scope=per_turbine -> 134 models and 134 scalers in one checkpoint file
    Args:
        cfg: config dict (see default_config)
        csv_path: training CSV. Under folds this MUST be the fold's fold_train.csv
        ckpt_path: checkpoint destination
        device: "cuda" / "cpu"
        verbose: print per-epoch progress
    Returns:
        Best inner-validation loss (mean over turbines when per_turbine).
    """
    csv_path = csv_path or VAL_TRAIN_FILE
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])

    raw = load_farm_array(csv_path)
    train_raw, inner_raw = split_prefix(raw, cfg["inner_val_days"])
    n_in = N_FEATURES

    blob = {"cfg": cfg, "n_in": n_in, "scope": cfg["scope"]}

    if cfg["scope"] == "global":
        scaler = Scaler().fit(train_raw)
        train_arr = scaler.transform(train_raw)
        inner_arr = scaler.transform(
            np.concatenate([train_raw[:, -cfg["lookback"]:], inner_raw], axis=1))
        if verbose:
            n_par = sum(p.numel() for p in build_model(cfg, n_in).parameters())
            print("[train] scope=global  {} params  device={}".format(n_par, device))
        state, best = _fit_one(cfg, train_arr, inner_arr, n_in, device, None, verbose)
        blob["state_dict"] = state
        blob["scaler"] = scaler.state()

    else:
        states, scalers, losses = [], [], []
        for tid in range(raw.shape[0]):
            sc = Scaler().fit(train_raw[tid:tid + 1])
            tr = sc.transform(train_raw[tid:tid + 1])
            iv = sc.transform(np.concatenate(
                [train_raw[tid:tid + 1, -cfg["lookback"]:], inner_raw[tid:tid + 1]],
                axis=1))
            st, bl = _fit_one(cfg, tr, iv, n_in, device, 0, False)
            states.append(st); scalers.append(sc.state()); losses.append(bl)
            if verbose and (tid % 10 == 0 or tid == raw.shape[0] - 1):
                print("[train] scope=per_turbine  turbine {:>3}/{}  inner-val {:.5f}"
                      .format(tid, raw.shape[0] - 1, bl))
        blob["state_dicts"] = states
        blob["scalers"] = scalers
        best = float(np.mean(losses))

    blob["inner_val_loss"] = best
    if ckpt_path:
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        torch.save(blob, ckpt_path)
        if verbose:
            print("[train] wrote {}".format(ckpt_path))
    return best


_CACHE = {}


def load_checkpoint(path, device=None):
    """Load a checkpoint once and memoise it by path."""
    key = (path, device)
    if key in _CACHE:
        return _CACHE[key]
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no checkpoint at {}. Train this fold first.".format(path))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    if blob["scope"] == "global":
        m = build_model(cfg, blob["n_in"]).to(device)
        m.load_state_dict(blob["state_dict"])
        m.eval()
        out = (cfg, blob["scope"], [m], [Scaler.from_state(blob["scaler"])])
    else:
        models, scalers = [], []
        for st, sc in zip(blob["state_dicts"], blob["scalers"]):
            m = build_model(cfg, blob["n_in"]).to(device)
            m.load_state_dict(st)
            m.eval()
            models.append(m)
            scalers.append(Scaler.from_state(sc))
        out = (cfg, blob["scope"], models, scalers)
    _CACHE[key] = out
    return out


def predict_windows(envs, ckpt_template, device=None, clip_negative=True):
    """
    Description:
        The body of forecast(envs): resolve this fold's checkpoint, read the
        window infile through the official loader, take the most recent
        `lookback` steps per turbine, predict, and invert the normalisation
        back to kW.
    Args:
        envs: evaluator settings dict, needs "path_to_test_x" and, under folds, "fold"
        ckpt_template: checkpoint path, optionally containing "{fold}"
        device: torch device string
        clip_negative: clamp predictions at 0 (power cannot be negative)
    Returns:
        float array (capacity, OUTPUT_LEN, 1) of predicted Patv in kW.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_fold_path(envs, ckpt_template)
    cfg, scope, models, scalers = load_checkpoint(path, device)
    capacity = envs.get("capacity", CAPACITY)
    L = cfg["lookback"]

    test_x = TestData(path_to_data=envs["path_to_test_x"],
                      start_col=START_COL, farm_capacity=capacity)

    raw = np.zeros((capacity, L, N_FEATURES), dtype=np.float32)
    for tid in range(capacity):
        seq, _ = test_x.get_turbine(tid)
        seq = np.asarray(seq, dtype=np.float32)
        if seq.shape[0] < L:
            # Left-pad with the oldest observation rather than zeros: zero is a
            # legitimate power reading and would be read as a dead turbine.
            pad = np.repeat(seq[:1], L - seq.shape[0], axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        raw[tid] = seq[-L:]

    out = np.zeros((capacity, OUTPUT_LEN), dtype=np.float64)
    with torch.no_grad():
        if scope == "global":
            xs = scalers[0].transform(raw)
            pred = models[0](torch.from_numpy(xs).float().to(device)).cpu().numpy()
            out[:] = scalers[0].inverse_target(pred)
        else:
            for tid in range(capacity):
                sc, m = scalers[tid], models[tid]
                xs = sc.transform(raw[tid:tid + 1])
                p = m(torch.from_numpy(xs).float().to(device)).cpu().numpy()
                out[tid] = sc.inverse_target(p)[0]

    if clip_negative:
        out = np.clip(out, 0.0, None)
    return out.reshape(capacity, OUTPUT_LEN, 1).astype(float)


def base_config(**overrides):
    """
    Description:
        Hyperparameters shared by every neural baseline in this directory.
        Kept identical across the files ON PURPOSE: a knob that means the same
        thing everywhere is what makes their scores comparable. Change one here
        and change it in all of them.
    Args:
        **overrides: model-specific values
    Returns:
        A config dict.
    """
    cfg = {
        "lookback": 144,         # 24 h -- pinned to the official GRU baseline
        "scope": "global",       # "global" or "per_turbine"
        "stride": 6,             # keep every 6th forecast origin (hourly)
        "loss": "mse",           # "mse" (official) / "l1" / "half"
        "batch_size": 256,
        "lr": 1e-3,
        "lr_decay": 0.95,
        "epochs": 40,
        "patience": 8,
        "clip_norm": 5.0,
        "inner_val_days": 20,    # tail of the training prefix, early stop only
        "seed": 3407,            # matches the GRU baseline's base seed
        "num_workers": 4,
    }
    cfg.update(overrides)
    return cfg
# ===========================================================================
# MODEL -- everything below this line is specific to lstm.
# ===========================================================================
CKPT = os.environ.get(
    "SDWPF_LSTM_CKPT",
    os.path.join(_HERE, "checkpoints", "lstm_{fold}.pt"))

LOOKBACK = 144          # pinned to the GRU baseline
SCOPE = "global"        # "global" (1 model) or "per_turbine" (134 models)
HIDDEN = 48             # width of EACH of the two state vectors
LAYERS = 2              # stacked LSTM layers
DROPOUT = 0.05          # pinned to the GRU baseline
LOSS = "mse"            # pinned to the GRU baseline
CLIP_NEGATIVE = True


def default_config(**overrides):
    """
    Description:
        This model's config: the shared base plus its architecture keys, with
        the module-level constants as defaults so sweep overrides take effect.
    Args:
        **overrides: values from a CLI flag or a sweep point
    Returns:
        A config dict.
    """
    cfg = base_config(
        lookback=LOOKBACK, scope=SCOPE, loss=LOSS,
        hidden=HIDDEN, layers=LAYERS, dropout=DROPOUT,
        lr=1e-3, batch_size=256, epochs=40, patience=8,
    )
    cfg.update(overrides)
    return cfg


class LSTMForecaster(nn.Module):
    """
    Description:
        LSTM encoder over the lookback window, flat linear head to the horizon.

        The head reads the FINAL hidden state of the top layer -- the encoder's
        summary of the whole window. Emitting the horizon in one projection
        rather than decoding step by step means errors never compound, which
        matters over 288 steps.

        Activations are sigmoid on the three gates and tanh on the cell
        content. They are fixed by the architecture: a gate must output a
        fraction, which sigmoid guarantees, and the content must be signed and
        bounded, which tanh gives. Neither is a tuning choice.
    """

    def __init__(self, cfg, n_in):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_in,
            hidden_size=cfg["hidden"],
            num_layers=cfg["layers"],
            batch_first=True,
            dropout=cfg["dropout"] if cfg["layers"] > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = nn.Linear(cfg["hidden"], OUTPUT_LEN)

    def forward(self, x):
        """
        Args:
            x: (batch, lookback, channels) normalised history
        Returns:
            (batch, OUTPUT_LEN) normalised Patv forecast.
        """
        out, _ = self.lstm(x)            # (batch, steps, hidden)
        return self.head(self.drop(out[:, -1]))


def build_model(cfg, n_in):
    """Factory used by the shared pipeline."""
    return LSTMForecaster(cfg, n_in)


def train(train_csv=None, ckpt_path=None, device=None, verbose=True, **overrides):
    """
    Description:
        Fit this baseline and write a checkpoint. Under a rolling-origin
        layout, call once per fold with that fold's fold_train.csv -- each fold
        must have its own model or it trains on its own validation targets.
    Args:
        train_csv: training CSV (default VAL_TRAIN_FILE)
        ckpt_path: checkpoint destination
        device: "cuda" / "cpu"
        verbose: print progress
        **overrides: any config key
    Returns:
        Best inner-validation loss (float).
    """
    return train_model(default_config(**overrides), csv_path=train_csv,
                       ckpt_path=ckpt_path, device=device, verbose=verbose)


def forecast(envs):
    """
    Description:
        The program contract used by evaluator.py.
    Args:
        envs: settings dict with "path_to_test_x" and, under folds, "fold"
    Returns:
        float array (134, 288, 1) of predicted Patv in kW.
    """
    return predict_windows(envs, CKPT, clip_negative=CLIP_NEGATIVE)


def main():
    """CLI: train one model."""
    p = argparse.ArgumentParser(description="Train the lstm baseline")
    p.add_argument("--train-csv", type=str, default=None,
                   help="training CSV; under folds, that fold's fold_train.csv")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint path; must be fold-specific under folds")
    p.add_argument("--scope", type=str, default=None,
                   choices=["global", "per_turbine"])
    p.add_argument("--lookback", type=int, default=None)
    p.add_argument("--loss", type=str, default=None, choices=["mse", "l1", "half"])
    p.add_argument("--hidden", type=int, default=None, help="width of each state vector")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    a = p.parse_args()

    over = {k: v for k, v in [
        ("scope", a.scope), ("lookback", a.lookback), ("loss", a.loss),
        ("hidden", a.hidden), ("layers", a.layers), ("dropout", a.dropout),
        ("lr", a.lr), ("epochs", a.epochs), ("stride", a.stride),
        ("seed", a.seed)] if v is not None}

    ckpt = a.ckpt or CKPT
    if "{fold}" in ckpt:
        raise SystemExit(
            "--ckpt still contains '{fold}'. Training fits ONE fold at a time; "
            "pass the resolved path, e.g. .../lstm_F1.pt")
    train(train_csv=a.train_csv, ckpt_path=ckpt, device=a.device, **over)


if __name__ == "__main__":
    main()
