# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/baseline_models/gru/gru_baseline.py

Description:
    Single-file PyTorch port of the official Baidu KDD Cup 2022 SDWPF GRU
    baseline: ONE independent GRU per turbine (134 models). Contains everything
    needed to train the baseline and to score it through evaluator.py:
      * Scaler          -- the baseline's normalizer (global scalar mean/std)
      * TurbineWindowDataset / build_turbine_splits -- per-turbine 214/31 splits
      * BaselineGruModel -- GRU(10->48) x2 + dropout + Linear(48->1)
      * train_one_turbine / main() -- the official training recipe + CLI
      * forecast(envs)  -- the program contract, so evaluator.py can score it

Citation:
    Port of PaddlePaddle/PaddleSpatial, apps/wpf_baseline_gru/
    ({model,train,common,wind_turbine_data}.py and the submission predict.py in
    kddcup22-sdwpf-evaluation/paddlepaddle/tests)
    https://github.com/PaddlePaddle/PaddleSpatial  (accessed 2026-07-22)
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # Run from the package root (windevolve_sdwpf/):
    # 1. Smoke-test that it runs (3 turbines, 1 epoch, throwaway folder):
    python baseline_models/gru/gru_baseline.py \
        --turbines 0-2 --epochs 1 --checkpoints /tmp/gru_smoke
    # 2. The real run -- all 134 turbines, one pass, one seed:
    python baseline_models/gru/gru_baseline.py --device cuda
    # 3. Score it. Inference is stochastic, so take several draws:
    for s in 1 2 3 4 5; do
      SDWPF_GRU_INFER_SEED=$s python evaluator.py baseline_models/gru/gru_baseline.py
    done
"""
import os                                    # checkpoint paths, env vars
import sys                                   # package-root discovery
import time                                  # timing
import argparse                              # training CLI

import numpy as np                           # array math
import pandas as pd                          # CSV reading
import torch                                 # model + training
import torch.nn as nn                        # layers and loss
from torch.utils.data import Dataset, DataLoader

# This file lives in a subfolder (baseline_models/gru/), but test_data.py lives
# at the package root. Walk up until we find it, then put that on sys.path so
# the import works whether this file is run directly or loaded by evaluator.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
for _ in range(4):
    if os.path.exists(os.path.join(_ROOT, "test_data.py")):
        break
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from test_data import TestData               # official per-turbine window loader

DAY_LEN = 144        # 10-minute observations per day
START_COL = 3        # first modeled column (Wspd); 0-2 are TurbID/Day/Tmstamp

# Default location of the 245-day training CSV (override with --data).
DEFAULT_TRAIN_CSV = os.environ.get(
    "SDWPF_TRAIN_FILE",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/sdwpf_245days_v1.csv",
)

# The official global seed, set ONCE before the turbine loop
# (common.py / train.py: fix_seed = 3407).
OFFICIAL_FIX_SEED = 3407

# Folder holding model_0.pt ... model_133.pt (written by training, read by forecast).
CHECKPOINTS = os.environ.get("SDWPF_GRU_CHECKPOINTS", os.path.join(_HERE, "checkpoints"))

# Optional seed for inference. The official predict.py seeds nothing, so its
# forecasts are irreproducible run to run; set this to make a given run
# repeatable while keeping the dropout noise that the official code has.
INFER_SEED = os.environ.get("SDWPF_GRU_INFER_SEED")

# Clip negative predictions to zero. False matches the official baseline.
CLIP_NEGATIVE = False

# Module-level cache: models load once and are reused across all 142 windows.
_CACHE = {"models": None, "scalers": None, "settings": None, "device": None}


def default_settings():
    """
    Description:
        The official baseline hyperparameters for the 245-day KDD Cup file.
    Args:
        None
    Returns:
        A settings dict consumed by the model, data pipeline, and trainer.
    """
    return {
        "task": "MS",             # multivariate in, single variable out
        "target": "Patv",         # predicted variable
        "input_len": 144,         # history window fed to the GRU (1 day)
        "output_len": 288,        # forecast horizon (48h)
        "start_col": START_COL,   # first modeled column (Wspd)
        "in_var": 10,             # modeled input variables
        "out_var": 1,             # outputs (Patv)
        "day_len": DAY_LEN,       # observations per day
        "train_size": 214,        # training days (official 245-day split)
        "val_size": 31,           # validation days
        "total_size": 245,        # days per turbine in the file
        "lstm_layer": 2,          # GRU layers
        "dropout": 0.05,          # dropout before the projection
        "capacity": 134,          # turbines
        "train_epochs": 10,       # max epochs per turbine
        "batch_size": 32,         # minibatch size
        "patience": 3,            # early-stopping patience
        "lr": 1e-4,               # Adam learning rate
    }


class Scaler(object):
    """
    Description:
        The baseline's normalizer. Uses one GLOBAL scalar mean and std over all
        elements of the fitted array (not per-column), matching the original.
    """

    def __init__(self, mean=0.0, std=1.0):
        """
        Description:
            Initialize with optional pre-computed statistics.
        Args:
            mean: scalar mean (0.0 until fit)
            std: scalar standard deviation (1.0 until fit)
        Returns:
            None
        """
        self.mean = mean        # scalar mean over all fitted elements
        self.std = std          # scalar std over all fitted elements

    def fit(self, data):
        """
        Description:
            Compute the global scalar mean/std over the whole array.
        Args:
            data: 2-D array of the modeled columns for the training split
        Returns:
            None
        """
        self.mean = float(np.mean(data))
        self.std = float(np.std(data))

    def transform(self, data):
        """
        Description:
            Normalize using the fitted global statistics.
        Args:
            data: array to normalize
        Returns:
            The normalized array.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Description:
            Undo the normalization, returning values to kW.
        Args:
            data: normalized array (e.g. model output)
        Returns:
            The de-normalized array.
        """
        return (data * self.std) + self.mean


def load_modeled_columns(path, start_col=START_COL):
    """
    Description:
        Read the 245-day training CSV and return only the modeled columns, with
        NaNs replaced by 0 (as the baseline does before scaling).
    Args:
        path: path to sdwpf_245days_v1.csv
        start_col: index of the first modeled column (3 -> Wspd)
    Returns:
        A float array of shape (n_rows, n_modeled_cols), rows turbine-major.
    """
    df = pd.read_csv(path)
    df_data = df[df.columns[start_col:]]
    df_data = df_data.replace(to_replace=np.nan, value=0)
    return df_data.values.astype(np.float32)


class TurbineWindowDataset(Dataset):
    """
    Description:
        Rolling-window dataset over one turbine's split. Each item is an
        input_len-step input window and the following output_len-step target
        window (all modeled columns; the target column is selected at loss time).
    """

    def __init__(self, data, input_len, output_len):
        """
        Description:
            Wrap an already-scaled 2-D array for one turbine split.
        Args:
            data: scaled array of shape (n_steps, n_features) for this split
            input_len: history window length (144)
            output_len: forecast horizon (288)
        Returns:
            None
        """
        self.data = data                # scaled split for this turbine
        self.input_len = input_len      # input window length
        self.output_len = output_len    # forecast horizon

    def __len__(self):
        """
        Description:
            Number of complete (input, target) windows in this split.
        Args:
            None
        Returns:
            The window count as an int.
        """
        return len(self.data) - self.input_len - self.output_len + 1

    def __getitem__(self, index):
        """
        Description:
            Return one rolling window pair starting at `index`.
        Args:
            index: window start position
        Returns:
            A tuple (seq_x, seq_y) shaped (input_len, n_features) and
            (output_len, n_features).
        """
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        return self.data[s_begin:s_end], self.data[r_begin:r_end]


def build_turbine_splits(data, tid, settings):
    """
    Description:
        Build this turbine's train/validation datasets and its fitted Scaler,
        using the baseline's border arithmetic. The scaler is fit on the
        turbine's training rows only, then applied to both splits. The
        validation split starts input_len steps early so its first window has a
        full history, exactly as in the original.
    Args:
        data: the full modeled-columns array from load_modeled_columns()
        tid: turbine index in [0, capacity)
        settings: dict with input_len, output_len, day_len, train_size (days),
                  val_size (days), total_size (days)
    Returns:
        A tuple (train_ds, val_ds, scaler).
    """
    unit = settings["day_len"]                      # steps per day (144)
    total_size = settings["total_size"] * unit      # steps per turbine (245 days)
    train_size = settings["train_size"] * unit      # training steps (214 days)
    val_size = settings["val_size"] * unit          # validation steps (31 days)
    input_len = settings["input_len"]
    output_len = settings["output_len"]

    # Border arithmetic from the official loader (index 0 = train, 1 = val).
    border1s = [tid * total_size,
                tid * total_size + train_size - input_len]
    border2s = [tid * total_size + train_size,
                tid * total_size + train_size + val_size]

    # Fit the scaler on this turbine's TRAIN rows only.
    scaler = Scaler()
    scaler.fit(data[border1s[0]:border2s[0]])

    train_arr = scaler.transform(data[border1s[0]:border2s[0]])
    val_arr = scaler.transform(data[border1s[1]:border2s[1]])
    return (TurbineWindowDataset(train_arr, input_len, output_len),
            TurbineWindowDataset(val_arr, input_len, output_len),
            scaler)


class BaselineGruModel(nn.Module):
    """
    Description:
        The official SDWPF GRU baseline: GRU(in_var -> 48) x lstm_layer, dropout,
        then a linear projection to out_var, emitting the final output_len steps.
        The original used a time_major GRU with explicit transposes, which is
        equivalent to batch_first=True here.
    """

    def __init__(self, settings):
        """
        Description:
            Build the GRU baseline from the settings dict.
        Args:
            settings: dict with "output_len", "in_var", "out_var", "dropout",
                      and "lstm_layer"
        Returns:
            None
        """
        super(BaselineGruModel, self).__init__()
        self.output_len = settings["output_len"]      # forecast horizon (288)
        self.hidC = settings["in_var"]                # input feature count (10)
        self.hidR = 48                                # GRU hidden size (fixed at 48)
        self.out = settings["out_var"]                # output variables (1 = Patv)
        self.dropout = nn.Dropout(settings["dropout"])
        self.lstm = nn.GRU(input_size=self.hidC, hidden_size=self.hidR,
                           num_layers=settings["lstm_layer"], batch_first=True)
        self.projection = nn.Linear(self.hidR, self.out)

        # Deviation 6: paddle's nn.Linear defaults to XavierNormal weights and a
        # ZERO bias; PyTorch defaults to U(+/-1/sqrt(fan_in)) for both. Measured
        # on paddle 3.3.1: Linear(48,512) weight std 0.05944 vs XavierNormal's
        # sqrt(2/(48+512)) = 0.05976, bias all-zero.
        # The GRU's own init already matches across frameworks -- both use
        # U(+/-1/sqrt(hidden)), verified -- so it is deliberately left alone.
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x_enc):
        """
        Description:
            Pad the input with output_len zero steps, run the GRU over the padded
            sequence, project, and return the final output_len steps.
        Args:
            x_enc: input tensor of shape (batch, input_len, in_var), normalized
        Returns:
            A tensor of shape (batch, output_len, out_var).
        """
        x = torch.zeros(x_enc.shape[0], self.output_len, x_enc.shape[2],
                        dtype=x_enc.dtype, device=x_enc.device)
        x_enc = torch.cat((x_enc, x), dim=1)          # (B, input_len+output_len, C)
        dec, _ = self.lstm(x_enc)                     # (B, T, hidR)
        sample = self.projection(self.dropout(dec))   # (B, T, out)
        return sample[:, -self.output_len:, -self.out:]


def pick_device(requested=None):
    """
    Description:
        Choose the compute device, preferring an explicit request and otherwise
        falling back through CUDA and Apple MPS to CPU.
    Args:
        requested: "cuda", "mps", "cpu", or None for auto-detection
    Returns:
        A torch.device.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_turbines(spec, capacity):
    """
    Description:
        Parse a turbine selection string into a list of turbine indices.
    Args:
        spec: e.g. "0-133", "0-9", or "0,5,7" (None -> all turbines)
        capacity: total number of turbines (134)
    Returns:
        A sorted list of turbine indices.
    """
    if not spec:
        return list(range(capacity))
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(i for i in ids if 0 <= i < capacity)


def adjust_learning_rate(optimizer, epoch, lr):
    """
    Description:
        Apply the baseline's "type1" schedule: halve the learning rate each
        epoch (lr * 0.5**(epoch-1)).
    Args:
        optimizer: the Adam optimizer being updated
        epoch: 1-based epoch number just completed
        lr: the base learning rate
    Returns:
        None
    """
    new_lr = lr * (0.5 ** (epoch - 1))
    for group in optimizer.param_groups:
        group["lr"] = new_lr


def clip_gradients(model, max_norm=50.0):
    """
    Description:
        Clip gradients the way the official baseline does: PER PARAMETER.
        paddle.nn.ClipGradByNorm(50.0) bounds each parameter tensor separately.
        torch.nn.utils.clip_grad_norm_(model.parameters(), ...) bounds the
        GLOBAL norm instead -- that is paddle's ClipGradByGlobalNorm, a
        different rule -- so we clip one parameter at a time.
        Verified on paddle 3.3.1 with gradient norms 34.641 and 173.205 at
        clip_norm 50: ClipGradByNorm left the first untouched and clipped the
        second to 50.000; the global rule gives 9.806 and 49.029.
    Args:
        model: the model whose gradients are being clipped
        max_norm: the per-parameter norm bound (50.0, from the official settings)
    Returns:
        None
    """
    for p in model.parameters():
        if p.grad is not None:
            torch.nn.utils.clip_grad_norm_([p], max_norm=max_norm)


def run_epoch(model, loader, criterion, optimizer, device, out_var, train=True):
    """
    Description:
        Run one pass over a data loader, optionally updating the model,
        reproducing the official training loop exactly. Three deviations from a
        standard PyTorch loop are deliberate and are marked inline: gradients
        are never zeroed, the optimizer is applied twice per batch, and
        validation runs with dropout active.
    Args:
        model: the GRU being trained or evaluated
        loader: DataLoader yielding (seq_x, seq_y) batches
        criterion: the loss function (MSE)
        optimizer: Adam optimizer (ignored when train=False)
        device: torch device to run on
        out_var: number of output variables (1); selects the target column
        train: True to backpropagate, False for validation
    Returns:
        The mean loss over the epoch as a float.
    """
    # Deviation 2: the official train.py sets model.train() at the top of each
    # epoch and its val() never switches to eval, so the validation loss driving
    # early stopping is computed WITH dropout active. Hence train() either way.
    model.train()

    losses = []
    with torch.set_grad_enabled(train):
        for batch_x, batch_y in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            sample = model(batch_x)
            # Task 'MS': compare only the last out_var column(s) = Patv.
            target = batch_y[:, -sample.shape[1]:, -out_var:]
            loss = criterion(sample, target)
            if train:
                # Deviation 3: NO optimizer.zero_grad(). The official train.py
                # never calls clear_grad() anywhere in the run, so the gradient
                # buffer accumulates monotonically across every batch and every
                # epoch. Verified with lr forced to 0: norms grew 1.00x, 2.00x,
                # 3.00x, 4.00x over successive batches.
                loss.backward()
                clip_gradients(model)
                optimizer.step()
                # Deviation 4: paddle's `minimize(loss)` already applies the
                # update and the official loop then calls `step()` too, giving
                # two Adam applications per batch. Verified: 2.00x the weight
                # movement of a single step.
                clip_gradients(model)
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train_one_turbine(data, tid, settings, device, ckpt_dir):
    """
    Description:
        Train a single turbine's GRU with early stopping and save the best
        weights plus that turbine's scaler statistics.
    Args:
        data: the full modeled-columns array from load_modeled_columns()
        tid: turbine index to train
        settings: hyperparameter dict from default_settings()
        device: torch device
        ckpt_dir: folder to write model_{tid}.pt into
    Returns:
        The best validation loss as a float.
    """
    # Deviation 7: NO per-turbine seeding. The official seeds once
    # (fix_seed = 3407) before the turbine loop, so each turbine continues one
    # RNG stream and its weights depend on the turbines before it having drawn
    # first. main() does that seeding; re-seeding here would break the stream.
    train_ds, val_ds, scaler = build_turbine_splits(data, tid, settings)
    train_loader = DataLoader(train_ds, batch_size=settings["batch_size"],
                              shuffle=True, drop_last=True)
    # Deviation 8: the official get_data() passes shuffle=True for BOTH train
    # and val, with drop_last=True -- so a different random subset of validation
    # windows is discarded every epoch, adding noise to the early-stopping signal.
    val_loader = DataLoader(val_ds, batch_size=settings["batch_size"],
                            shuffle=True, drop_last=True)

    model = BaselineGruModel(settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    criterion = nn.MSELoss(reduction="mean")

    best_val = float("inf")     # lowest validation loss seen
    bad_epochs = 0              # epochs since the last improvement
    best_state = None           # weights corresponding to best_val

    for epoch in range(1, settings["train_epochs"] + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device,
                            settings["out_var"], train=True)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device,
                            settings["out_var"], train=False)
        print("    turbine {:3d} epoch {:2d}  train {:.6f}  val {:.6f}".format(
            tid, epoch, tr_loss, va_loss))
        if va_loss < best_val:
            best_val = va_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= settings["patience"]:
                print("    turbine {:3d} early stopped at epoch {}".format(tid, epoch))
                break
        adjust_learning_rate(optimizer, epoch + 1, settings["lr"])

    # Persist weights AND scaler stats so inference needs no training CSV.
    # `full_run` records whether this checkpoint came from a single pass over
    # all 134 turbines. Deviation 7 means only such a run reproduces the
    # official RNG stream; forecast() warns if it loads a partial one.
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state_dict": best_state if best_state is not None else model.state_dict(),
                "scaler_mean": scaler.mean,
                "scaler_std": scaler.std,
                "settings": settings,
                "full_run": settings.get("_full_run", True),
                "best_val_loss": best_val},
               os.path.join(ckpt_dir, "model_{}.pt".format(tid)))
    return best_val


def _has_checkpoints(ckpt_dir):
    """
    Description:
        Report whether a folder already holds any model_{tid}.pt files. Used to
        detect a resumed run, which does not reproduce the official RNG stream.
    Args:
        ckpt_dir: folder to inspect
    Returns:
        True if at least one checkpoint is present.
    """
    if not os.path.isdir(ckpt_dir):
        return False
    return any(n.startswith("model_") and n.endswith(".pt") for n in os.listdir(ckpt_dir))


def _announce(partial_turbines):
    """
    Description:
        Print, once, how this baseline is being run: dropout is live at
        inference (deviation 1), so forecast() is stochastic, and whether any
        checkpoint came from a partial training run (deviation 7).
    Args:
        partial_turbines: list of turbine ids whose checkpoints are stamped as
            coming from a run that did not cover all 134 turbines
    Returns:
        None
    """
    print("[gru] official baseline: dropout is ACTIVE at inference (the official "
          "predict.py never calls eval()), so forecast() is STOCHASTIC -- score "
          "several times and report mean +/- std.", flush=True)
    if INFER_SEED is not None:
        print("[gru] inference seeded with SDWPF_GRU_INFER_SEED={}".format(INFER_SEED),
              flush=True)
    if partial_turbines:
        print("[gru] WARNING: {} checkpoint(s) came from a PARTIAL training run "
              "(e.g. turbine {}). The official baseline seeds once (fix_seed={}) "
              "before a single pass over all 134 turbines, so a partial run does "
              "not reproduce its RNG stream. Fine for a smoke test; do not report "
              "the score.".format(len(partial_turbines), partial_turbines[0],
                                  OFFICIAL_FIX_SEED), flush=True)


def _load_all(capacity):
    """
    Description:
        Load every turbine's checkpoint once and populate the module cache with
        the models, their scaler statistics, and the training settings.
    Args:
        capacity: number of turbines expected (134)
    Returns:
        None
    """
    if _CACHE["models"] is not None:
        return
    device = pick_device()
    # Announce which checkpoint folder is being used, so the scorer's output
    # makes the source unambiguous (e.g. checkpoints vs checkpoints_clip).
    print("[gru] loading {} checkpoints from: {}".format(capacity, CHECKPOINTS), flush=True)
    models, scalers, settings, partial = [], [], None, []
    for tid in range(capacity):
        path = os.path.join(CHECKPOINTS, "model_{}.pt".format(tid))
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing checkpoint for turbine {} at {!r}. Train it first: "
                "python baseline_models/gru/gru_baseline.py".format(tid, path))
        blob = torch.load(path, map_location=device, weights_only=False)
        settings = blob["settings"]
        model = BaselineGruModel(settings).to(device)
        model.load_state_dict(blob["state_dict"])
        # Deviation 1: the official predict.py never calls model.eval(), and
        # paddle Layers default to training=True, so the published 47.850 was
        # produced with dropout ACTIVE at inference. Reproduced here, which is
        # what makes forecast() stochastic.
        model.train()
        models.append(model)
        scalers.append((blob["scaler_mean"], blob["scaler_std"]))
        if not blob.get("full_run", True):
            partial.append(tid)
    _CACHE.update({"models": models, "scalers": scalers,
                   "settings": settings, "device": device})
    _announce(partial)


def forecast(envs):
    """
    Description:
        Produce a 48h active-power forecast for every turbine using the trained
        per-turbine GRU baseline. This is the program contract used by
        evaluator.py, so the baseline is scored with the official metric.
    Args:
        envs: settings dict; must contain "path_to_test_x" (the infile path).
              Optional "capacity" overrides the turbine count.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", 134)
    _load_all(capacity)
    models, scalers = _CACHE["models"], _CACHE["scalers"]
    settings, device = _CACHE["settings"], _CACHE["device"]
    # In faithful mode dropout is live, so forecast() is stochastic. The
    # official predict.py seeds nothing; SDWPF_GRU_INFER_SEED makes a given run
    # repeatable without removing the noise.
    if INFER_SEED is not None:
        torch.manual_seed(int(INFER_SEED))
    input_len = settings["input_len"]
    output_len = settings["output_len"]
    out_var = settings["out_var"]

    # Official loader: per-turbine arrays of the modeled columns with NaN -> 0.
    test_x = TestData(path_to_data=envs["path_to_test_x"],
                      start_col=settings["start_col"],
                      farm_capacity=capacity)

    predictions = np.zeros((capacity, output_len, out_var), dtype=float)
    with torch.no_grad():
        for tid in range(capacity):
            seq, _ = test_x.get_turbine(tid)             # (n_steps, in_var), NaN->0
            mean, std = scalers[tid]
            seq = (np.asarray(seq, dtype=np.float32) - mean) / std
            last_observ = seq[-input_len:]               # most recent input window
            batch = torch.from_numpy(last_observ).float().unsqueeze(0).to(device)
            pred = models[tid](batch).squeeze(0).cpu().numpy()
            predictions[tid] = pred * std + mean         # inverse transform to kW

    if CLIP_NEGATIVE:
        predictions = np.clip(predictions, 0.0, None)
    return predictions


def main():
    """
    Description:
        Parse arguments, load the training data once, seed as the official run
        does, and train the turbines in order.
    Args:
        None
    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Train the per-turbine SDWPF GRU baseline (official behaviour)")
    parser.add_argument("--data", type=str, default=DEFAULT_TRAIN_CSV,
                        help="Path to sdwpf_245days_v1.csv")
    parser.add_argument("--checkpoints", type=str, default=CHECKPOINTS,
                        help="Folder for model_{tid}.pt checkpoints")
    parser.add_argument("--turbines", type=str, default=None,
                        help="SMOKE TESTS ONLY, e.g. '0-2'. The official run covers all 134 "
                             "turbines in one pass from a single seed, so a subset does not "
                             "reproduce it; such checkpoints are stamped partial and scoring "
                             "warns about them. Default: all 134")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs (official: 10). Smoke tests only")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (official: 32). Smoke tests only")
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu (default: auto)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain turbines whose checkpoints already exist")
    args = parser.parse_args()

    settings = default_settings()
    if args.epochs is not None:
        settings["train_epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size

    device = pick_device(args.device)
    turbines = parse_turbines(args.turbines, settings["capacity"])
    full_run = (len(turbines) == settings["capacity"]) and not _has_checkpoints(args.checkpoints)
    settings["_full_run"] = full_run

    # Deviation 7: the official seeds ONCE here, before the turbine loop, so
    # every turbine continues one RNG stream and its weights depend on the
    # turbines before it having drawn first.
    import random                                   # the official seeds this too
    random.seed(OFFICIAL_FIX_SEED)
    np.random.seed(OFFICIAL_FIX_SEED)
    torch.manual_seed(OFFICIAL_FIX_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(OFFICIAL_FIX_SEED)

    print("device      : {}".format(device))
    print("turbines    : {} (n={})".format(
        "{}..{}".format(turbines[0], turbines[-1]) if turbines else "none", len(turbines)))
    print("checkpoints : {}".format(args.checkpoints))
    print("seed        : {} (set once, before the turbine loop)".format(OFFICIAL_FIX_SEED))
    if not full_run:
        print("")
        print("  *** PARTIAL RUN -- SMOKE TEST ONLY ***")
        print("  The official baseline trains all {} turbines in a single pass from one".format(
            settings["capacity"]))
        print("  seed, so every turbine's weights depend on the ones before it. This run")
        print("  covers {} turbine(s){} and therefore does NOT reproduce the official".format(
            len(turbines), " and resumes an existing folder" if
            _has_checkpoints(args.checkpoints) else ""))
        print("  RNG stream. The checkpoints are stamped partial and scoring will warn.")
        print("  For the real run: delete the checkpoint folder and pass no --turbines.")
        print("")

    print("loading training data ...")
    t0 = time.perf_counter()
    data = load_modeled_columns(args.data, settings["start_col"])
    print("  loaded {} rows x {} cols in {:.1f}s".format(
        data.shape[0], data.shape[1], time.perf_counter() - t0))

    start_all = time.perf_counter()
    for n, tid in enumerate(turbines, 1):
        ckpt = os.path.join(args.checkpoints, "model_{}.pt".format(tid))
        if os.path.exists(ckpt) and not args.overwrite:
            print("[{}/{}] turbine {:3d}: checkpoint exists, skipping".format(n, len(turbines), tid))
            continue
        t_turb = time.perf_counter()
        print("[{}/{}] training turbine {:3d}".format(n, len(turbines), tid))
        best = train_one_turbine(data, tid, settings, device, args.checkpoints)
        print("    done in {:.1f}s (best val {:.6f})".format(time.perf_counter() - t_turb, best))

    print("all requested turbines finished in {:.1f}s".format(time.perf_counter() - start_all))


if __name__ == "__main__":
    main()
