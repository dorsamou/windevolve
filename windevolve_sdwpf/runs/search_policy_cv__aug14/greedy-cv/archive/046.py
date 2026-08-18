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
import random                                # seeding

import numpy as np                           # array math
import pandas as pd                          # CSV reading
import torch                                 # model + training
import torch.nn as nn                        # layers and loss
from torch.utils.data import Dataset, DataLoader

# baselines/gru/ -> baselines/ -> repo root. Add the repo root (paths.py) and
# evaluation/ (test_data.py) so imports work whether this file is run directly
# or loaded by path from the evaluator.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = "/home/dorsa/srip/windevolve_sdwpf"
for _p in (_ROOT, os.path.join(_ROOT, "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_data import TestData               # official per-turbine window loader
from paths import VAL_TRAIN_FILE

DAY_LEN = 144        # 10-minute observations per day
START_COL = 3        # first modeled column (Wspd); 0-2 are TurbID/Day/Tmstamp

# Default location of the 245-day training CSV (override with --data).
# The TRUNCATED training csv (pre-validation rows only). Never the full 245
# days -- that would train on the validation targets this model is scored on.
DEFAULT_TRAIN_CSV = VAL_TRAIN_FILE

# The official global seed, set ONCE before the turbine loop
# (common.py / train.py: fix_seed = 3407).
# Base training seed. The official value is 3407. Per-turbine seeding uses
# OFFICIAL_FIX_SEED + tid, so changing this shifts every turbine's seed together
# and gives an independent training run -- which is what a noise-floor
# measurement needs. Override with SDWPF_GRU_SEED or --seed.
OFFICIAL_FIX_SEED = int(os.environ.get("SDWPF_GRU_SEED", "3407"))

# Folder holding model_0.pt ... model_133.pt (written by training, read by forecast).
# ckpt_005: refitted at hidden=128 under the relaxed 40/8/0.9 schedule, five
# training runs, each on its own fold's fold_train.csv and nothing later. The
# seed's ./checkpoints/ is immutable and is left untouched. The {fold}
# placeholder is what keeps fold k scoring on weights that never saw fold k's
# validation block -- never hardcode one fold's directory here.
CHECKPOINTS = os.environ.get(
    "SDWPF_GRU_CHECKPOINTS", os.path.join(_HERE, ".cache", "ckpt_005", "{fold}"))

# Optional seed for inference. The official predict.py seeds nothing, so its
# forecasts are irreproducible run to run; set this to make a given run
# repeatable while keeping the dropout noise that the official code has.
INFER_SEED = os.environ.get("SDWPF_GRU_INFER_SEED")

# Keep dropout ACTIVE at inference, as the official predict.py does. False (the
# default) calls model.eval(), so forecast() is deterministic. Set True only to
# reproduce the published baseline's stochastic behaviour.
INFER_DROPOUT = os.environ.get("SDWPF_GRU_INFER_DROPOUT", "").lower() in ("1", "true", "yes")

# Clip negative predictions to zero. False matches the official baseline.
# Measured no-op for this model: its minimum output over a full horizon is
# ~245 kW, so nothing is ever clipped.
CLIP_NEGATIVE = False

# ---------------------------------------------------------------------------
# LONG-HORIZON LEVEL PROBE
#
# Past ~24h this model emits a flat per-turbine constant (measured: 1.4 kW of
# variation across windows, against 358 kW in the first hour), so two thirds of
# every scored horizon is that one number and the score is dominated by where
# it sits. Experiment 002 moved it UP by 66.5 kW on average and lost 1.0 MW on
# all five folds, which says the level matters a great deal but not whether the
# model is already at the optimum or sitting above it. This is the mirror
# measurement: the same magnitude, downward, and uniform across turbines so it
# isolates LEVEL alone (002 also widened the cross-turbine spread from 37.7 to
# 62.6 kW, confounding the two).
#
#   symmetric loss  -> the model's constant is already optimal, avenue closed
#   improvement     -> the optimum is lower, and worth following down
# ---------------------------------------------------------------------------
GRU_AMPLIFY = 1.40       # matches the linear member's fitted beta scale
GRU_TAIL = 48            # trailing steps that define the network's flat anchor
TAIL_SHIFT = -66.5       # kW added to the tail, ramped in over TAIL_RAMP steps
TAIL_RAMP = 144          # steps over which the shift ramps 0 -> 1 (24h)

# Module-level cache: models load once and are reused across all 142 windows.
_CACHE = {"models": None, "scalers": None, "settings": None,
          "device": None, "path": None}

def _resolve_path(envs, template):
    """
    Description:
        Resolve a checkpoint directory, substituting "{fold}" when the evaluator
        supplies a fold name in envs. This is what lets ONE process score every
        fold of a cross-validation layout: without it the directory is fixed at
        import time and all five folds would silently load the same weights --
        which is a leak, not a score.
    Args:
        envs: the settings dict passed to forecast() (may be None)
        template: the configured path, optionally containing "{fold}"
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
        # NOTE: train_size/val_size/total_size describe the FILE passed to
        # --data, not the competition's 245-day file specifically. total_size in
        # particular is the per-turbine STRIDE used to index the flat array in
        # build_turbine_splits(); if it disagrees with the file, every turbine
        # after the first is read from the wrong offset and training silently
        # trains on another turbine's data. main() therefore overrides all three
        # from the file's actual length -- see resolve_day_sizes().
        "lstm_layer": 2,          # GRU layers (official: 2; 1 differs by 0.018 MW)
        "hidden": 48,             # GRU hidden size (official: 48) -- see note below
        "dropout": 0.05,          # dropout before the projection
        "capacity": 134,          # turbines
        "batch_size": 32,         # minibatch size
        "clip_norm": 50.0,        # gradient norm bound
        "lr": 1e-4,               # Adam learning rate (official: 1e-4)

        # ------------------------------------------------------------------
        # TRAINING BUDGET AND SCHEDULE. These three differ from the official
        # values (10 epochs, patience 3, decay 0.5) and are the best MEASURED
        # configuration on the 197/48 validation split, at 44.537 MW against
        # 45.136 for the official settings. See docs/baseline_report.md 4.6.
        #
        # The official schedule was the binding constraint, not the learning
        # rate or the architecture. lr=1e-4 halved every epoch is ~1/500th of
        # its starting value by epoch 10, so the model stopped improving long
        # before it converged:
        #
        #   lr=3e-4, decay 0.5          44.642
        #   lr=1e-3, decay 0.5          44.667
        #   lr=1e-4, NO decay           44.604   <- rate was never the problem
        #   40 epochs, patience 8,
        #     decay 0.9                 44.537   <- adopted
        #
        # Every optimiser variant gains ~0.5 MW over the official settings, and
        # the best of them keeps the official learning rate and only relaxes the
        # schedule.
        # ------------------------------------------------------------------
        "train_epochs": 40,       # max epochs per turbine (official: 10)
        "patience": 8,            # early-stopping patience (official: 3)
        "lr_decay": 0.9,          # per-epoch multiplier (official: 0.5)

        # NOT ADOPTED, though it scored better in isolation: hidden=128 gave
        # 44.745 against 45.136 for hidden=48, monotonic in 48 -> 96 -> 128 and
        # still improving at the edge of the swept range. It was measured under
        # the OFFICIAL optimiser schedule, so combining it with the budget above
        # is an extrapolation no sweep point tested. Raising capacity and
        # re-measuring is an open item; until then the default is the
        # configuration that was actually scored at 44.537.
        #
        # Also measured and NOT adopted: input_len. 144 -> 288 -> 432 moves the
        # score by 0.043 MW in total, the smallest effect in the entire study.
        # Reading 24h to forecast 48h is not the binding constraint.

        # ------------------------------------------------------------------
        # TRAINING-LOOP CORRECTNESS. The official Baidu code has four real
        # defects (gru_baseline.md section 6). This project's GRU is a
        # comparator for the evolutionary search and a candidate seed program,
        # not a reproduction of a published number -- so the defaults below are
        # the CORRECT behaviour, and the official behaviour is opt-in via
        # --official (or settings["official_loop"]).
        #
        #   zero_grad    True  = zero the gradient buffer each step. The
        #                official train.py never calls clear_grad(), so its
        #                gradients accumulate across every batch AND epoch --
        #                by batch n the applied gradient is the sum of all n
        #                preceding ones. This is the worst of the four.
        #   double_step  False = one Adam application per batch. The official
        #                gets two (paddle's minimize() updates, then step()).
        #   global_clip  True  = clip the global gradient norm, PyTorch's usual
        #                rule. The official clips each parameter separately.
        #   eval_val     True  = dropout off while measuring validation loss,
        #                so early stopping reads signal rather than noise.
        #   shuffle_val  False = measure the same validation windows every
        #                epoch. The official shuffles with drop_last, discarding
        #                a different random subset each time.
        #   per_turbine_seed True = seed from OFFICIAL_FIX_SEED + tid inside
        #                each turbine. The official seeds ONCE before the loop,
        #                which makes turbine 100's weights depend on turbines
        #                0-99 having drawn first -- so a partial run is not
        #                reproducible and turbines cannot be trained in
        #                parallel. Per-turbine seeding fixes both, and is what
        #                makes --turbines sharding valid.
        # ------------------------------------------------------------------
        "zero_grad": True,
        "double_step": False,
        "global_clip": True,
        "eval_val": True,
        "shuffle_val": False,
        "per_turbine_seed": True,
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
        self.hidR = settings.get("hidden", 48)        # GRU hidden size (official 48)
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


def resolve_day_sizes(data, settings, train_days=None, val_days=None):
    """
    Description:
        Derive total_size / train_size / val_size (all in DAYS) from the file
        that was actually loaded, rather than trusting the 245-day defaults.

        This exists because total_size doubles as the per-turbine stride in
        build_turbine_splits(). Training on the truncated 196-day val_train.csv
        with total_size still at 245 would index turbine t at step t*35280 in an
        array that only has t*28224 per turbine -- every turbine but the first
        would be fitted on a misaligned slice of somebody else's series, and
        nothing would raise.

        The inner train/val boundary is the split used for EARLY STOPPING only.
        It is carved out of the file it is given, so when that file is
        val_train.csv the early-stopping signal never touches the held-out
        validation period. By default the official 214:245 proportion is
        preserved (196 days -> 171 train / 25 early-stopping), which keeps the
        recipe unchanged apart from having less data.

    Args:
        data: the modeled-columns array from load_modeled_columns()
        settings: settings dict to update in place
        train_days: explicit inner-training days (None -> official proportion)
        val_days: explicit inner-validation days (None -> the remainder)
    Returns:
        The updated settings dict.
    """
    unit = settings["day_len"]
    capacity = settings["capacity"]

    rows = data.shape[0]
    if rows % capacity != 0:
        raise ValueError(
            "file has {} rows, not divisible by capacity {} -- turbines must "
            "have equal-length blocks".format(rows, capacity))
    steps = rows // capacity
    if steps % unit != 0:
        raise ValueError(
            "file has {} steps per turbine, not a whole number of {}-step "
            "days".format(steps, unit))
    total_days = steps // unit

    if train_days is None and val_days is None:
        # Preserve the official 214/245 proportion on whatever length we got.
        train_days = int(round(total_days * 214.0 / 245.0))
        val_days = total_days - train_days
    elif val_days is None:
        val_days = total_days - train_days
    elif train_days is None:
        train_days = total_days - val_days

    if train_days <= 0 or val_days <= 0:
        raise ValueError("train_days={} val_days={} -- both must be positive".format(
            train_days, val_days))
    if train_days + val_days != total_days:
        raise ValueError(
            "train_days ({}) + val_days ({}) = {} does not equal the file's "
            "{} days; the borders would leave a gap or overrun".format(
                train_days, val_days, train_days + val_days, total_days))

    # Each split must hold at least one complete (input, target) window.
    need = settings["input_len"] + settings["output_len"]
    if train_days * unit < need:
        raise ValueError("inner-train split ({} steps) is shorter than one "
                         "window ({} steps)".format(train_days * unit, need))
    # The val split starts input_len steps early, so it needs output_len more.
    if val_days * unit + settings["input_len"] < need:
        raise ValueError("inner-val split ({} steps) is shorter than one "
                         "window ({} steps)".format(val_days * unit, need))

    settings["total_size"] = total_days
    settings["train_size"] = train_days
    settings["val_size"] = val_days
    return settings


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


def adjust_learning_rate(optimizer, epoch, lr, decay=0.5):
    """
    Description:
        Apply the baseline's "type1" schedule: multiply the learning rate by
        `decay` each epoch (lr * decay**(epoch-1)). decay=0.5 is official and
        means the rate is ~1/500th of its original value by epoch 10 -- which,
        starting from 1e-4, is why the model barely moves in the later epochs.
        decay=1.0 disables the schedule.
    Args:
        optimizer: the Adam optimizer being updated
        epoch: 1-based epoch number just completed
        lr: the base learning rate
        decay: per-epoch multiplier
    Returns:
        None
    """
    new_lr = lr * (decay ** (epoch - 1))
    for group in optimizer.param_groups:
        group["lr"] = new_lr


def clip_gradients(model, max_norm=50.0, global_norm=False):
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
        global_norm: True to use PyTorch's usual global rule instead
    Returns:
        None
    """
    if global_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        return
    for p in model.parameters():
        if p.grad is not None:
            torch.nn.utils.clip_grad_norm_([p], max_norm=max_norm)


def run_epoch(model, loader, criterion, optimizer, device, out_var, train=True,
              settings=None):
    """
    Description:
        Run one pass over a data loader, optionally updating the model,
        reproducing the official training loop exactly under the default
        settings. Three deviations from a standard PyTorch loop are deliberate
        and are marked inline: gradients are never zeroed, the optimizer is
        applied twice per batch, and validation runs with dropout active. Each
        is switchable via the settings dict so its cost can be measured.
    Args:
        model: the GRU being trained or evaluated
        loader: DataLoader yielding (seq_x, seq_y) batches
        criterion: the loss function (MSE)
        optimizer: Adam optimizer (ignored when train=False)
        device: torch device to run on
        out_var: number of output variables (1); selects the target column
        train: True to backpropagate, False for validation
        settings: hyperparameter dict; the deviation switches are read from it
    Returns:
        The mean loss over the epoch as a float.
    """
    s = settings or {}
    zero_grad = s.get("zero_grad", False)
    double_step = s.get("double_step", True)
    global_clip = s.get("global_clip", False)
    clip_norm = s.get("clip_norm", 50.0)

    # Deviation 2: the official train.py sets model.train() at the top of each
    # epoch and its val() never switches to eval, so the validation loss driving
    # early stopping is computed WITH dropout active. Hence train() either way,
    # unless eval_val asks otherwise.
    if train or not s.get("eval_val", False):
        model.train()
    else:
        model.eval()

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
                # Deviation 3: NO optimizer.zero_grad() by default. The official
                # train.py never calls clear_grad() anywhere in the run, so the
                # gradient buffer accumulates monotonically across every batch
                # and every epoch. Verified with lr forced to 0: norms grew
                # 1.00x, 2.00x, 3.00x, 4.00x over successive batches.
                if zero_grad:
                    optimizer.zero_grad()
                loss.backward()
                clip_gradients(model, clip_norm, global_clip)
                optimizer.step()
                # Deviation 4: paddle's `minimize(loss)` already applies the
                # update and the official loop then calls `step()` too, giving
                # two Adam applications per batch. Verified: 2.00x the weight
                # movement of a single step.
                if double_step:
                    clip_gradients(model, clip_norm, global_clip)
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
    # Seed per turbine, so this turbine's result does not depend on which
    # turbines ran before it. That makes a partial run reproducible and lets
    # --turbines shards run in parallel -- the training loop is launch-bound
    # rather than compute-bound, so sharding is close to a linear speedup.
    # settings["per_turbine_seed"] = False restores the official single-stream
    # seeding, in which case main() has already seeded and we must not re-seed.
    if settings.get("per_turbine_seed", True):
        seed = OFFICIAL_FIX_SEED + int(tid)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    train_ds, val_ds, scaler = build_turbine_splits(data, tid, settings)
    train_loader = DataLoader(train_ds, batch_size=settings["batch_size"],
                              shuffle=True, drop_last=True)
    # Deviation 8: the official get_data() passes shuffle=True for BOTH train
    # and val, with drop_last=True -- so a different random subset of validation
    # windows is discarded every epoch, adding noise to the early-stopping signal.
    val_loader = DataLoader(val_ds, batch_size=settings["batch_size"],
                            shuffle=settings.get("shuffle_val", True),
                            drop_last=settings.get("shuffle_val", True))

    model = BaselineGruModel(settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    criterion = nn.MSELoss(reduction="mean")

    best_val = float("inf")     # lowest validation loss seen
    bad_epochs = 0              # epochs since the last improvement
    best_state = None           # weights corresponding to best_val

    for epoch in range(1, settings["train_epochs"] + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device,
                            settings["out_var"], train=True, settings=settings)
        va_loss = run_epoch(model, val_loader, criterion, optimizer, device,
                            settings["out_var"], train=False, settings=settings)
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
        adjust_learning_rate(optimizer, epoch + 1, settings["lr"],
                             settings.get("lr_decay", 0.5))

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
    if INFER_DROPOUT:
        print("[gru] dropout is ACTIVE at inference (official behaviour), so "
              "forecast() is STOCHASTIC -- score several times and report "
              "mean +/- std.", flush=True)
    if INFER_SEED is not None:
        print("[gru] inference seeded with SDWPF_GRU_INFER_SEED={}".format(INFER_SEED),
              flush=True)
    if partial_turbines:
        print("[gru] NOTE: {} checkpoint(s) came from a partial training run "
              "(e.g. turbine {}). With per-turbine seeding this is reproducible "
              "and shardable, so it is only a problem if the run is INCOMPLETE. "
              "Under official single-stream seeding it would also fail to "
              "reproduce the published RNG stream.".format(
                  len(partial_turbines), partial_turbines[0]), flush=True)


def _load_all(capacity, envs=None):
    """
    Description:
        Load every turbine's checkpoint and populate the module cache with the
        models, their scaler statistics, and the training settings. The cache is
        keyed on the RESOLVED checkpoint directory, so scoring a second fold in
        the same process reloads that fold's weights rather than serving the
        previous fold's.
    Args:
        capacity: number of turbines expected (134)
        envs: settings dict from forecast(); supplies "fold" when present
    Returns:
        None
    """
    ckpt_dir = _resolve_path(envs, CHECKPOINTS)
    if _CACHE["models"] is not None and _CACHE["path"] == ckpt_dir:
        return
    device = pick_device()
    # Announce which checkpoint folder is being used, so the scorer's output
    # makes the source unambiguous (e.g. ckpt_val vs ckpt_F3).
    print("[gru] loading {} checkpoints from: {}".format(capacity, ckpt_dir), flush=True)
    models, scalers, settings, partial = [], [], None, []
    for tid in range(capacity):
        path = os.path.join(ckpt_dir, "model_{}.pt".format(tid))
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Missing checkpoint for turbine {} at {!r}. Train it first: "
                "python baseline_models/gru/gru_baseline.py".format(tid, path))
        blob = torch.load(path, map_location=device, weights_only=False)
        settings = blob["settings"]
        model = BaselineGruModel(settings).to(device)
        model.load_state_dict(blob["state_dict"])
        # The official predict.py never calls model.eval(), and paddle Layers
        # default to training=True, so the published 47.850 was produced with
        # dropout ACTIVE at inference -- which makes forecast() stochastic and
        # every score a draw rather than a number. INFER_DROPOUT restores that
        # for the provenance row; the default is a deterministic forecast.
        model.train() if INFER_DROPOUT else model.eval()
        models.append(model)
        scalers.append((blob["scaler_mean"], blob["scaler_std"]))
        if not blob.get("full_run", True):
            partial.append(tid)
    _CACHE.update({"models": models, "scalers": scalers,
                   "settings": settings, "device": device, "path": ckpt_dir})
    _announce(partial)


def _gru_forecast(envs):
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
    _load_all(capacity, envs)
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

    # Faded in as the GRU's own forecast decays to its flat constant: ~0 at h=0
    # where the network still knows where the turbine is, full by h=144 where it
    # has become climatology.
    steps = np.arange(1, output_len + 1)
    shift = (TAIL_SHIFT * np.minimum(1.0, steps / float(TAIL_RAMP))).reshape(output_len, 1)

    predictions = np.zeros((capacity, output_len, out_var), dtype=float)
    with torch.no_grad():
        for tid in range(capacity):
            seq, _ = test_x.get_turbine(tid)             # (n_steps, in_var), NaN->0
            mean, std = scalers[tid]
            seq = (np.asarray(seq, dtype=np.float32) - mean) / std
            last_observ = seq[-input_len:]               # most recent input window
            batch = torch.from_numpy(last_observ).float().unsqueeze(0).to(device)
            pred = models[tid](batch).squeeze(0).cpu().numpy() * std + mean
            # The same correction that the linear member needs, for the same
            # reason. This GRU is trained on MSE, so it forecasts the
            # conditional MEAN, while the score's MAE half rewards the
            # conditional median -- which for wind power stays nearer the
            # current state. Past ~24h the network has decayed to a flat
            # per-turbine anchor, so amplifying its departure from that anchor
            # is the direct analogue of inflating beta.
            #
            # The magnitude is bounded by full persistence, exactly as beta is
            # capped at 1.0: the corrected forecast may not sit further from the
            # anchor than the turbine's CURRENT deviation from it, so a
            # confident short-range prediction is never inflated past what has
            # actually been observed.
            anchor = pred[-GRU_TAIL:].mean()
            observed = float(np.asarray(seq[-1:], dtype=np.float64).ravel()[-1]) * std + mean
            room = abs(observed - anchor)
            spread = np.clip((pred - anchor) * GRU_AMPLIFY, -room, room)
            predictions[tid] = anchor + spread + shift

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
    parser.add_argument("--train-days", type=int, default=None,
                        help="Inner-training days used for fitting. Default: the official "
                             "214/245 proportion of whatever --data actually contains "
                             "(245-day file -> 214; 196-day val_train.csv -> 171)")
    parser.add_argument("--val-days", type=int, default=None,
                        help="Inner-validation days used ONLY for early stopping. Must sum "
                             "with --train-days to the file's day count. Default: the "
                             "remainder (245-day file -> 31; 196-day file -> 25)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base training seed (official: 3407). Per-turbine seeds "
                             "are seed+tid, so changing this gives an independent "
                             "training run -- use it to measure the noise floor")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience in epochs (official: 3)")
    parser.add_argument("--input-len", type=int, default=None,
                        help="History steps fed to the GRU (official: 144 = 24h, to "
                             "predict 288 = 48h). Larger gives the model more context")
    parser.add_argument("--hidden", type=int, default=None,
                        help="GRU hidden size (official: 48)")
    parser.add_argument("--layers", type=int, default=None,
                        help="Stacked GRU layers (official: 2)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout before the projection (official: 0.05)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Adam learning rate (official: 1e-4)")
    parser.add_argument("--lr-decay", type=float, default=None,
                        help="Per-epoch LR multiplier (official: 0.5; 1.0 disables)")
    parser.add_argument("--official", action="store_true",
                        help="Restore the official Baidu loop, defects and all: "
                             "gradients never zeroed, two optimizer steps per batch, "
                             "per-parameter clipping, dropout live during validation, "
                             "and a single RNG stream seeded once before the turbine "
                             "loop. Use only to reproduce the published 47.850")
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu (default: auto)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain turbines whose checkpoints already exist")
    args = parser.parse_args()

    settings = default_settings()

    # --official first, as a base preset: it restores the whole official
    # configuration (four loop defects, single-stream seeding, and the training
    # budget/schedule, which now differ by default -- 40/8/0.9 vs 10/3/0.5).
    # Explicit flags are applied AFTER it, so they still win.
    if args.official:
        settings.update({"zero_grad": False, "double_step": True,
                         "global_clip": False, "eval_val": False,
                         "shuffle_val": True, "per_turbine_seed": False,
                         "train_epochs": 10, "patience": 3, "lr_decay": 0.5})
        print("*** --official: reproducing the Baidu loop INCLUDING its four "
              "known defects; train all 134 turbines in one pass or the RNG "
              "stream will not match ***")

    if args.epochs is not None:
        settings["train_epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    for key, val in (("input_len", args.input_len), ("hidden", args.hidden),
                     ("patience", args.patience),
                     ("lstm_layer", args.layers), ("dropout", args.dropout),
                     ("lr", args.lr), ("lr_decay", args.lr_decay)):
        if val is not None:
            settings[key] = val
    if args.seed is not None:
        globals()["OFFICIAL_FIX_SEED"] = args.seed
        print("base seed    : {} (non-default)".format(args.seed))

    device = pick_device(args.device)
    turbines = parse_turbines(args.turbines, settings["capacity"])
    full_run = (len(turbines) == settings["capacity"]) and not _has_checkpoints(args.checkpoints)
    settings["_full_run"] = full_run

    # Under the default per-turbine seeding, train_one_turbine() seeds itself
    # from OFFICIAL_FIX_SEED + tid, so this global seed is only needed for
    # --official, which reproduces the single RNG stream the Baidu code uses.
    if not settings.get("per_turbine_seed", True):
        random.seed(OFFICIAL_FIX_SEED)
        np.random.seed(OFFICIAL_FIX_SEED)
        torch.manual_seed(OFFICIAL_FIX_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(OFFICIAL_FIX_SEED)

    print("device      : {}".format(device))
    print("turbines    : {} (n={})".format(
        "{}..{}".format(turbines[0], turbines[-1]) if turbines else "none", len(turbines)))
    print("checkpoints : {}".format(args.checkpoints))
    per_turb = settings.get("per_turbine_seed", True)
    print("seed        : {}{}".format(
        OFFICIAL_FIX_SEED,
        " + turbine id (per-turbine; shards are independent)" if per_turb
        else " (set once before the turbine loop; official single stream)"))
    if not full_run:
        print("")
        if per_turb:
            print("  Partial run: {} of {} turbines{}.".format(
                len(turbines), settings["capacity"],
                ", resuming an existing folder" if
                _has_checkpoints(args.checkpoints) else ""))
            print("  With per-turbine seeding this is reproducible and shardable --")
            print("  each turbine seeds itself, so its weights do not depend on which")
            print("  turbines ran before it. Only an INCOMPLETE set is a problem:")
            print("  scoring needs all {} checkpoints present.".format(settings["capacity"]))
        else:
            print("  *** PARTIAL RUN UNDER --official -- SMOKE TEST ONLY ***")
            print("  --official seeds once before the loop, so every turbine's weights")
            print("  depend on the ones before it. This run covers {} turbine(s) and".format(
                len(turbines)))
            print("  therefore does NOT reproduce the official RNG stream. For the real")
            print("  run: delete the checkpoint folder and pass no --turbines.")
        print("")

    print("loading training data ...")
    t0 = time.perf_counter()
    data = load_modeled_columns(args.data, settings["start_col"])
    print("  loaded {} rows x {} cols in {:.1f}s".format(
        data.shape[0], data.shape[1], time.perf_counter() - t0))

    # Override the 245-day defaults with the file's real geometry. Do this
    # BEFORE any turbine is touched: total_size is the per-turbine stride, so a
    # stale value silently misaligns every turbine after the first.
    resolve_day_sizes(data, settings, args.train_days, args.val_days)
    print("split       : {} days total -> {} inner-train / {} early-stopping".format(
        settings["total_size"], settings["train_size"], settings["val_size"]))
    if settings["total_size"] != 245:
        print("              (non-competition file length -- if this is val_train.csv, "
              "the held-out period is NOT visible to training or early stopping)")

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


# Predictors, as (column, trailing window in steps). The one-hour power term
# carries the short-horizon signal, the one-day term a slower background level,
# and wind speed separates "no wind" from "not currently generating".
# Entries are (column, trailing window in steps, scope). "turbine" scope uses
# each turbine's own history; "farm" scope uses the mean across all 134 at the
# same instant, which is a different KIND of information -- cross-sectional
# rather than temporal. It separates a faulted turbine from a becalmed one: both
# read near-zero power, but if the rest of the farm is generating then the wind
# is there and this machine should revert towards it when it returns. Neither
# its own recent power nor its own wind speed can express that, and ~23.5% of
# training rows are in some not-generating state.
# Blade pitch joins recent power because current power alone cannot separate a
# turbine that is briefly becalmed from one that is feathered and shut down --
# both read near zero. "Resid" sharpens the same idea: it is the turbine's power
# minus what the farm's own fitted power curve says its wind speed should be
# producing, so it is negative exactly when a machine is underperforming the
# wind available to it. Raw wind speed on its own added nothing here (015); it
# is the INTERACTION with power, not the wind, that identifies the state.
PREDICTORS = (("Patv", 6, "turbine"), ("Pab1", 6, "turbine"), ("Resid", 6, "turbine"))

# Derived predictors and the raw columns they are computed from.
DERIVED = {"Resid": ("Patv", "Wspd")}
CURVE_BINS = 40          # wind-speed bins for the fitted power curve
RECENT_STEPS = max(w for _, w, _ in PREDICTORS)  # history needed from the infile

# Ridge added to the normal equations, relative to their scale. The one-hour and
# one-day power levels are strongly correlated, so the 3x3 system is
# ill-conditioned; this is numerical hygiene, not regularisation of substance.
RIDGE = 1e-6

# Beta is shrunk towards zero -- towards climatology -- before use. The
# coefficients are not stable across periods: fitting them separately on each
# half of every fold's training window gives 6h ratios of 0.78, 1.52, 1.64,
# 1.81 and 1.49 between halves, a relative spread of roughly 35% around one.
# Scaling beta DOWN by that instability (0.89) was worse on all five folds, so
# the fitted coefficient is not too large. The drift has a direction: in four
# folds of five the LATER half is the more persistent one, and every validation
# block sits immediately after its training window, so a curve pooled over the
# whole record understates the persistence it will meet. This is the mirror of
# that failed probe -- the same factor, inverted -- to establish whether the
# optimum lies above the fitted value or whether the fit is simply already at it.
SCALE_GRID = (0.6, 3.0, 25)   # multiplier search range and resolution
SCALE_STRIDE = 16             # subsampling for the search (~110k rows per horizon)

# Fitted coefficients, keyed by (fold, training csv) -- NEVER a bare global.
# All five folds are scored in ONE process, so a fold-blind cache would serve
# fold 1's coefficients to every later fold and the numbers would be silently
# wrong while still looking reasonable.
_FIT_CACHE = {}


def _columns(path, capacity, names):
    """
    Description:
        Read the named columns of a turbine-major SDWPF csv and reshape each to
        one row per turbine, filling missing readings with zero -- the same
        convention the official loader applies before modelling.
    Args:
        path: csv to read (a fold_train.csv or a window infile)
        capacity: number of turbines (134)
        names: column names to return
    Returns:
        A dict mapping name -> (capacity, steps_per_turbine) float array.
    """
    frame = pd.read_csv(path, usecols=list(names))
    per = len(frame) // capacity
    out = {}
    for name in names:
        values = frame[name].values[:capacity * per].astype(np.float64)
        out[name] = np.nan_to_num(values, nan=0.0).reshape(capacity, per)
    return out


def _raw_names():
    """
    Description:
        The csv columns that must actually be read, expanding any derived
        predictor into the raw columns it is built from.
    Args:
        None
    Returns:
        A tuple of column names.
    """
    names = []
    for name, _, _ in PREDICTORS:
        names.extend(DERIVED.get(name, (name,)))
    return tuple(dict.fromkeys(names))


def _power_curve(data):
    """
    Description:
        A power curve PER TURBINE: mean power against wind speed, on wind-speed
        bins shared across the farm.

        Bin edges are pooled quantiles so every turbine's curve is measured on
        the same grid and the residuals stay comparable; only the bin MEANS are
        per turbine. Turbines here do not share one curve -- position in the
        array decides how often a machine sits in another's wake, and a wake
        loss is a permanent feature of the site rather than a fault. A pooled
        curve books that loss as underperformance and leaves the residual
        carrying a constant per-turbine offset; a per-turbine curve removes it,
        so the residual means what it is meant to mean.
    Args:
        data: dict of (capacity, steps) arrays from _columns()
    Returns:
        A (centres, means) pair of (n_bins,) and (capacity, n_bins) arrays.
    """
    wind, power = data["Wspd"], data["Patv"]
    edges = np.quantile(wind.ravel(), np.linspace(0.0, 1.0, CURVE_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.clip(np.searchsorted(edges, wind.ravel(), side="right") - 1,
                    0, CURVE_BINS - 1).reshape(wind.shape)
    pooled_counts = np.maximum(np.bincount(index.ravel(), minlength=CURVE_BINS), 1)
    centres = np.bincount(index.ravel(), weights=wind.ravel(),
                          minlength=CURVE_BINS) / pooled_counts
    pooled_means = np.bincount(index.ravel(), weights=power.ravel(),
                               minlength=CURVE_BINS) / pooled_counts
    means = np.empty((wind.shape[0], CURVE_BINS), dtype=np.float64)
    for tid in range(wind.shape[0]):
        counts = np.bincount(index[tid], minlength=CURVE_BINS)
        totals = np.bincount(index[tid], weights=power[tid], minlength=CURVE_BINS)
        means[tid] = np.where(counts > 0, totals / np.maximum(counts, 1), pooled_means)
    order = np.argsort(centres)
    return centres[order], means[:, order]


def _add_derived(data, curve):
    """
    Description:
        Attach derived predictor columns to a loaded data dict, in place.
    Args:
        data: dict of (capacity, steps) arrays from _columns()
        curve: the (centres, means) power curve from _power_curve()
    Returns:
        The same dict, with derived columns added.
    """
    centres, means = curve
    expected = np.empty_like(data["Patv"])
    for tid in range(data["Patv"].shape[0]):
        expected[tid] = np.interp(data["Wspd"][tid], centres, means[tid])
    data["Resid"] = data["Patv"] - expected
    return data


def _trailing_mean(series, steps):
    """
    Description:
        Trailing mean of `steps` observations, as a sliding window over each
        turbine's series. Column i is the mean over times i .. i+steps-1, so it
        represents the level known at time i+steps-1.

        Averaging rather than reading a single instant trades a small lag for a
        large reduction in sensor noise; at 10-minute resolution a lone reading
        is a poor estimate of the state that persists.
    Args:
        series: (capacity, steps_per_turbine) array
        steps: window length
    Returns:
        A (capacity, steps_per_turbine - steps + 1) float array.
    """
    csum = np.cumsum(np.concatenate(
        [np.zeros((series.shape[0], 1)), series], axis=1), axis=1)
    return (csum[:, steps:] - csum[:, :-steps]) / float(steps)


def _design(data, capacity, width, offset):
    """
    Description:
        Build the design matrix: one row per predictor, each a trailing mean in
        deviation form, all aligned so that column k describes the state known
        at time offset + k.
    Args:
        data: dict of (capacity, steps) arrays, from _columns()
        capacity: number of turbines (134)
        width: number of time positions to emit
        offset: index of the first time position (RECENT_STEPS - 1)
    Returns:
        A (n_predictors, capacity * width) float array of deviations, and the
        per-turbine predictor means used to centre it.
    """
    rows, centres = [], []
    for name, steps, scope in PREDICTORS:
        level = _trailing_mean(data[name], steps)
        if scope == "farm":
            level = np.broadcast_to(level.mean(axis=0), level.shape)
        # level column i is the state at time i + steps - 1; we want time
        # offset + k, so start at offset - (steps - 1).
        start = offset - (steps - 1)
        level = level[:, start:start + width]
        centre = level.mean(axis=1)
        rows.append((level - centre[:, None]).ravel())
        centres.append(centre)
    return np.asarray(rows), centres


def _fit(envs, capacity, output_len):
    """
    Description:
        Fit the forecaster on this fold's training file: each turbine's mean
        power, each predictor's per-turbine centre, and the coefficient vector
        beta(h) for every horizon step.

        Coefficients are pooled across turbines rather than fitted 134 times.
        Each turbine contributes its own deviations, so what is shared is the
        decay of predictability, and pooling is what makes the far end of the
        horizon stable.

        The predictor block is held at a single fixed time range for every h, so
        the normal-equations matrix is formed once and only the right-hand side
        changes with the horizon. That costs the last output_len positions of
        the training series -- about 2% -- and makes the fit a few hundred times
        cheaper than refitting per step.
    Args:
        envs: the settings dict passed to forecast()
        capacity: number of turbines (134)
        output_len: forecast horizon in steps (288)
    Returns:
        A (mu, centres, beta, curve) tuple: per-turbine mean power (capacity,),
        per-predictor per-turbine centres, and coefficients (n_pred, output_len).
    """
    key = (envs.get("fold"), envs.get("path_to_train"))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]

    data = _columns(envs["path_to_train"], capacity, _raw_names())
    curve = _power_curve(data)
    _add_derived(data, curve)
    power = data["Patv"]
    mu = power.mean(axis=1)
    dev = power - mu[:, None]

    offset = RECENT_STEPS - 1
    n_steps = power.shape[1]
    width = n_steps - offset - output_len          # positions with a full future
    if width <= 0:
        raise ValueError(
            "training file has {} steps per turbine, too few for a {}-step "
            "horizon plus {} steps of history".format(n_steps, output_len, offset))

    design, centres = _design(data, capacity, width, offset)
    gram = design @ design.T
    gram += np.eye(len(gram)) * (RIDGE * np.trace(gram) / len(gram))

    beta = np.zeros((len(design), output_len), dtype=np.float64)
    for h in range(1, output_len + 1):
        target = dev[:, offset + h:offset + h + width].ravel()
        beta[:, h - 1] = np.linalg.solve(gram, design @ target)

    # The OLS fit above minimises squared error, but the competition scores
    # 0.5*(RMSE + MAE). Those disagree: squared error is minimised by the
    # conditional MEAN, absolute error by the conditional MEDIAN, and for wind
    # power the median sits nearer the current state because the mean is dragged
    # by the long tail of reversion. A fixed inflation of 1.87 was measured to
    # help, on all five folds -- but a single number for all 288 horizons is a
    # guess. Here the multiplier is FITTED per horizon against the actual scored
    # objective, on this fold's own training rows.
    #
    # Only a scale is searched, not the coefficients themselves: the OLS
    # direction is well determined by ~1.8M rows, it is the magnitude that the
    # objective disagrees about. A grid is enough because the objective is
    # convex in the scale and the spacing is far finer than the effect.
    sub = design[:, ::SCALE_STRIDE]
    grid = np.linspace(SCALE_GRID[0], SCALE_GRID[1], SCALE_GRID[2])
    for h in range(output_len):
        base = beta[:, h] @ sub
        truth = dev[:, offset + h + 1:offset + h + 1 + width].ravel()[::SCALE_STRIDE]
        best, best_obj = 1.0, np.inf
        for m in grid:
            resid = truth - m * base
            obj = 0.5 * (np.sqrt(np.mean(resid * resid)) + np.mean(np.abs(resid)))
            if obj < best_obj:
                best, best_obj = m, obj
        beta[:, h] *= best

    print("[fit] fold {}: {} turbines, {} positions; beta on own 1h power "
          "10min {:.3f} 6h {:.3f} 24h {:.3f} 48h {:.3f}".format(
              envs.get("fold"), capacity, width, beta[0, 0],
              beta[0, min(35, output_len - 1)], beta[0, min(143, output_len - 1)],
              beta[0, -1]), flush=True)
    _FIT_CACHE[key] = (mu, centres, beta, curve)
    return mu, centres, beta, curve


def _linear_forecast(envs):
    """
    Description:
        Produce a 48h active-power forecast for every turbine by regressing on
        the state of its recent history. This is the program contract used by
        evaluator.py.
    Args:
        envs: settings dict; must contain "path_to_test_x" (the infile) and
              "path_to_train" (this fold's training csv). Optional "capacity"
              and "output_len" override the defaults.
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    capacity = envs.get("capacity", 134)
    output_len = envs.get("output_len", 288)
    mu, centres, beta, curve = _fit(envs, capacity, output_len)

    data = _add_derived(_columns(envs["path_to_test_x"], capacity, _raw_names()), curve)

    # State at the end of history, as deviations from the same per-turbine
    # centres the coefficients were fitted against.
    state = np.empty((len(PREDICTORS), capacity), dtype=np.float64)
    for j, (name, steps, scope) in enumerate(PREDICTORS):
        level = data[name][:, -steps:].mean(axis=1)
        if scope == "farm":
            level = np.full(capacity, level.mean())
        state[j] = level - centres[j]

    predictions = mu[:, None] + state.T @ beta
    return predictions[:, :, None]


# ---------------------------------------------------------------------------
# ENSEMBLE OF TWO MODEL CLASSES
#
# Averaging the GRU's kW forecast with the linear model's. Experiment 012
# averaged two GRUs that differed only in input normalization and lost on every
# fold: their inner-validation losses agreed to within 0.7% in kW because they
# were making the same predictions, and averaging near-identical predictors
# buys nothing.
#
# These two are a different proposition -- 134 recurrent networks reading ten
# channels, against a single pooled regression on the last hour of power and
# the farm's contemporaneous level. They land within 0.05 MW of each other by
# different routes and disagree about which folds they win, and averaging them
# beat BOTH parents (43.72 against 43.99 and 44.03).
# ---------------------------------------------------------------------------

def forecast(envs):
    """
    Description:
        Produce a 48h active-power forecast by averaging the per-turbine GRU
        baseline and the linear model. This is the program contract used by
        evaluator.py.
    Args:
        envs: settings dict; must contain "path_to_test_x" and "path_to_train".
    Returns:
        A float array of shape (capacity, output_len, 1) with predicted Patv (kW).
    """
    return 0.5 * (np.asarray(_gru_forecast(envs)) + np.asarray(_linear_forecast(envs)))
