# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/gru_baseline.py

Description:
    Single-file PyTorch port of the official Baidu KDD Cup 2022 SDWPF GRU
    baseline: ONE independent GRU per turbine (134 models). Contains everything
    needed to train the baseline and to score it through evaluator.py:
      * Scaler          -- the baseline's normalizer (global scalar mean/std)
      * TurbineWindowDataset / build_turbine_splits -- per-turbine 214/31 splits
      * BaselineGruModel -- GRU(10->48) x2 + dropout + Linear(48->1)
      * train_one_turbine / main() -- the official training recipe + CLI
      * forecast(envs)  -- the program contract, so evaluator.py can score it

    Two details are reproduced deliberately because they change the numbers:
      1. The Scaler uses a SINGLE GLOBAL scalar mean/std over the whole 2-D
         array (np.mean(data)), NOT per-column statistics.
      2. That scaler is fit per turbine on THAT turbine's training split only,
         and predictions are inverse-transformed with the same scalars.
    NaNs are replaced by 0 before scaling, as in the original.

    Deviation from the original (a pure speed win, mathematically identical):
    each checkpoint stores the scaler statistics alongside the weights, so
    inference never re-reads the 4.7M-row training CSV.

    Predictions are NOT clipped at zero, matching the official predict.py.
    Set CLIP_NEGATIVE = True to test that variant (a deviation from baseline).

Citation:
    Port of PaddlePaddle/PaddleSpatial, apps/wpf_baseline_gru/
    ({model,train,common,wind_turbine_data}.py and the submission predict.py in
    kddcup22-sdwpf-evaluation/paddlepaddle/tests)
    https://github.com/PaddlePaddle/PaddleSpatial  (accessed 2026-07-22)
    Dataset: Zhou et al., arXiv:2208.04360 (2022);
             Scientific Data 11:649 (2024), doi:10.1038/s41597-024-03427-5.

Example usage:
    # 1. Validate the pipeline cheaply (a few turbines, one epoch):
    python gru_baseline.py --turbines 0-2 --epochs 1
    # 2. Time a single full turbine to decide local vs cluster:
    time python gru_baseline.py --turbines 3 --overwrite
    # 3. Full faithful training run (resumable; skips existing checkpoints):
    python gru_baseline.py --turbines 0-133 --device cuda
    # 4. Score the trained baseline with the official metric:
    python evaluator.py gru_baseline.py
"""
import os                                    # checkpoint paths, env vars
import time                                  # timing
import argparse                              # training CLI

import numpy as np                           # array math
import pandas as pd                          # CSV reading
import torch                                 # model + training
import torch.nn as nn                        # layers and loss
from torch.utils.data import Dataset, DataLoader

from test_data import TestData               # official per-turbine window loader

DAY_LEN = 144        # 10-minute observations per day
START_COL = 3        # first modeled column (Wspd); 0-2 are TurbID/Day/Tmstamp

# Default location of the 245-day training CSV (override with --data).
DEFAULT_TRAIN_CSV = os.environ.get(
    "SDWPF_TRAIN_FILE",
    "/Users/dorsam/Documents/SRIP 2026/wind_forecasting_data/SDWPF_dataset/"
    "sdwpf_kddcup/sdwpf_245days_v1.csv",
)

# Folder holding model_0.pt ... model_133.pt (written by training, read by forecast).
CHECKPOINTS = os.environ.get(
    "SDWPF_GRU_CHECKPOINTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
)

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
    scaler = Scaler() # 
    scaler.fit(data[border1s[0]:border2s[0]]) # finds the mean and std of the training split

    train_arr = scaler.transform(data[border1s[0]:border2s[0]])
    val_arr = scaler.transform(data[border1s[1]:border2s[1]]) # apply the same scaler to the validation split
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


def run_epoch(model, loader, criterion, optimizer, device, out_var, train=True):
    """
    Description:
        Run one pass over a data loader, optionally updating the model.
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
    model.train() if train else model.eval()
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train_one_turbine(data, tid, settings, device, ckpt_dir, seed=42):
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
        seed: RNG seed for reproducible initialization
    Returns:
        The best validation loss as a float.
    """
    torch.manual_seed(seed + tid)                  # reproducible per-turbine init
    train_ds, val_ds, scaler = build_turbine_splits(data, tid, settings)
    train_loader = DataLoader(train_ds, batch_size=settings["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=settings["batch_size"],
                            shuffle=False, drop_last=True)

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
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({"state_dict": best_state if best_state is not None else model.state_dict(),
                "scaler_mean": scaler.mean,
                "scaler_std": scaler.std,
                "settings": settings,
                "best_val_loss": best_val},
               os.path.join(ckpt_dir, "model_{}.pt".format(tid)))
    return best_val


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
    models, scalers, settings = [], [], None
    for tid in range(capacity):
        path = os.path.join(CHECKPOINTS, "model_{}.pt".format(tid))
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing checkpoint for turbine {} at {!r}. Train it first, e.g. "
                "python gru_baseline.py --turbines 0-133".format(tid, path))
        blob = torch.load(path, map_location=device, weights_only=False)
        settings = blob["settings"]
        model = BaselineGruModel(settings).to(device)
        model.load_state_dict(blob["state_dict"])
        model.eval()                       # disable dropout for inference
        models.append(model)
        scalers.append((blob["scaler_mean"], blob["scaler_std"]))
    _CACHE.update({"models": models, "scalers": scalers,
                   "settings": settings, "device": device})


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
        Parse arguments, load the training data once, and train the selected
        turbines, skipping any whose checkpoint already exists.
    Args:
        None
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train the per-turbine SDWPF GRU baseline")
    parser.add_argument("--data", type=str, default=DEFAULT_TRAIN_CSV,
                        help="Path to sdwpf_245days_v1.csv")
    parser.add_argument("--checkpoints", type=str, default=CHECKPOINTS,
                        help="Folder for model_{tid}.pt checkpoints")
    parser.add_argument("--turbines", type=str, default=None,
                        help="Turbines to train, e.g. '0-133', '0-9', '0,5,7' (default: all)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain turbines whose checkpoints already exist")
    args = parser.parse_args()

    settings = default_settings()
    if args.epochs is not None:
        settings["train_epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size

    device = pick_device(args.device) # choose compute device
    turbines = parse_turbines(args.turbines, settings["capacity"]) # parse turbine selection
    print("device      : {}".format(device))
    print("turbines    : {} (n={})".format(
        "{}..{}".format(turbines[0], turbines[-1]) if turbines else "none", len(turbines)))
    print("checkpoints : {}".format(args.checkpoints))

    print("loading training data ...")
    t0 = time.perf_counter()
    data = load_modeled_columns(args.data, settings["start_col"]) # load CSV and replace NaN with 0
    print("  loaded {} rows x {} cols in {:.1f}s".format(
        data.shape[0], data.shape[1], time.perf_counter() - t0))

    start_all = time.perf_counter() # start timing the whole training process
    for n, tid in enumerate(turbines, 1):
        ckpt = os.path.join(args.checkpoints, "model_{}.pt".format(tid)) # checkpoint path for this turbine
        if os.path.exists(ckpt) and not args.overwrite:
            print("[{}/{}] turbine {:3d}: checkpoint exists, skipping".format(n, len(turbines), tid))
            continue
        t_turb = time.perf_counter() # start timing this turbine's training
        print("[{}/{}] training turbine {:3d}".format(n, len(turbines), tid))
        best = train_one_turbine(data, tid, settings, device, args.checkpoints, seed=args.seed) # train the turbine and get the best validation loss
        print("    done in {:.1f}s (best val {:.6f})".format(time.perf_counter() - t_turb, best))

    print("all requested turbines finished in {:.1f}s".format(time.perf_counter() - start_all))


if __name__ == "__main__":
    main()