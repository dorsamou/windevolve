# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/hik/dmst.py

Description:
    Module 2 of 4 of FDSTT (team HIK, rank 1, Baidu KDD Cup 2022): the Deep
    Multi-relational Spatio-Temporal network. A GRU sequence-to-sequence model,
    ONE set of weights shared across all 134 turbines, with turbine identity
    supplied by a learned embedding.

    HIK published no code. This is reconstructed from the paper, Section 3.3
    and Figure 1. Every place the paper is silent is a keyword argument with a
    default, marked AMBIGUITY below -- so the reconstruction's assumptions are
    visible and switchable rather than baked in.

Architecture (paper Section 4.1.2 for the stated hyperparameters):

    encoder input  (B, 432, 19) = embedding 5 + aggregated wspd 4 + other 10
      |  nn.GRU(19 -> 64), 1 layer
      |  keep ONLY the final hidden state; the other 431 are discarded
      v
    h_T (B, 64)
      |  nn.Linear(64 -> 32)          <- AMBIGUITY 3, forced by 64 != 32
      v
    h_0 (B, 32)
      |  288 x [ GRUCell(9 -> 32) -> Linear(32 -> 1) ]   autoregressive
      v
    (B, 288, 1)

    The 10 "other" features, in the paper's order (Section 4.1.2):
        Etmp, Itmp, Wdir, Ndir, Pab1, Prtv, Patv, Hour, x, y
    Note Pab2/Pab3 are unused, and Patv (the target) IS an encoder input.

    The 3 future covariates the decoder gets:
        Hour, x, y
    That is all that is knowable about the future -- the competition provides
    no weather forecast. This is the central difficulty of the task and the
    reason the decoder input is 9 wide against the encoder's 19.

Ambiguities, all switchable (see AMBIGUITIES.md for the full log):
    1. AGGREGATE in Eqs. 4-6 unspecified          -> mean
    2. Eqs. 4-6 concat x_i once per graph, so the
       turbine's own wspd appears twice           -> dup_self=True (paper-literal)
    3. encoder hidden 64 vs decoder hidden 32     -> nn.Linear bridge
    5. Eqs. 10-12 index typos                     -> standard autoregression
    6. loss never stated                          -> FilterHuberLoss, grader mask
    8. layer count never stated                   -> 1
    9. teacher forcing never stated               -> ratio configurable, default 1.0
   10. output clipping never stated               -> relu_output=True

Usage:
    python leaderboard/hik/dmst.py          # shape + gradient self-test
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAPACITY = 134
INPUT_LEN = 432          # paper: T = 432 (3 days at 10-min resolution)
OUTPUT_LEN = 288         # paper: tau = 288 (2 days)
N_OTHER = 10             # Etmp Itmp Wdir Ndir Pab1 Prtv Patv Hour x y
N_FUT = 3                # Hour x y

OTHER_COLS = ["Etmp", "Itmp", "Wdir", "Ndir", "Pab1", "Prtv", "Patv",
              "Hour", "x", "y"]


# --------------------------------------------------------------------------
# graph aggregation -- Eqs. 4-6
# --------------------------------------------------------------------------
def aggregate_wspd(wspd, g_d, g_s, dup_self=True):
    """
    Description:
        The yellow "Multi-relational Graph" box in Figure 1. Takes each
        turbine's raw wind speed and appends the mean wind speed of its
        spatial neighbours and of its semantic neighbours.

        Cheap and fixed, so this belongs in the data pipeline (precompute
        once per window), not inside the model's forward pass.

    Args:
        wspd: (T, 134) wind speed for every turbine at every timestep.
        g_d: (134, 134) binary spatial adjacency; row i marks N_d(i).
        g_s: (134, 134) binary semantic adjacency; row i marks N_s(i).
        dup_self: AMBIGUITY 2. True reproduces Eqs. 4-6 literally --
            CONCAT(x_i, agg_d) then CONCAT(x_i, agg_s) then CONCAT of both,
            which includes x_i twice, giving 4 columns. False deduplicates
            to [x_i, agg_d, agg_s], giving 3.

    Returns:
        (T, 134, 4) if dup_self else (T, 134, 3).
    """
    wspd = np.asarray(wspd, dtype=np.float32)
    deg_d = np.maximum(g_d.sum(1, keepdims=True), 1.0)      # (134, 1)
    deg_s = np.maximum(g_s.sum(1, keepdims=True), 1.0)

    agg_d = (wspd @ g_d.T) / deg_d.T                        # (T, 134)
    agg_s = (wspd @ g_s.T) / deg_s.T

    parts = ([wspd, agg_d, wspd, agg_s] if dup_self
             else [wspd, agg_d, agg_s])
    return np.stack(parts, axis=-1).astype(np.float32)


def enc_feature_dim(d_emb=5, dup_self=True, n_other=N_OTHER):
    """Width of the assembled encoder input: embedding + wspd block + other."""
    return d_emb + (4 if dup_self else 3) + n_other


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class DMST(nn.Module):
    """
    Description:
        Deep Multi-relational Spatio-Temporal network. One model, all 134
        turbines: the turbine index enters only through nn.Embedding, so every
        turbine's samples train every weight.

    Args:
        n_turbines: rows in the embedding table.
        d_emb: paper Section 4.1.2, "wind turbine embedding size is 5".
        n_wspd: 4 if dup_self else 3; must match aggregate_wspd.
        enc_hidden / dec_hidden: paper, 64 and 32.
        num_layers: AMBIGUITY 8. Never stated; 1 is the plain reading.
        relu_output: AMBIGUITY 10. Power cannot be negative.
    """

    def __init__(self, n_turbines=CAPACITY, d_emb=5, n_wspd=4,
                 n_other=N_OTHER, n_fut=N_FUT, enc_hidden=64, dec_hidden=32,
                 num_layers=1, horizon=OUTPUT_LEN, relu_output=True):
        super().__init__()
        self.horizon = horizon
        self.relu_output = relu_output
        self.num_layers = num_layers
        self.dec_hidden = dec_hidden

        enc_in = d_emb + n_wspd + n_other        # 19 with paper defaults
        dec_in = d_emb + 1 + n_fut               # 9  with paper defaults
        self.enc_in, self.dec_in = enc_in, dec_in

        self.emb = nn.Embedding(n_turbines, d_emb)
        self.encoder = nn.GRU(enc_in, enc_hidden, num_layers=num_layers,
                              batch_first=True)
        self.bridge = nn.Linear(enc_hidden, dec_hidden)
        self.decoder = nn.GRUCell(dec_in, dec_hidden)
        self.head = nn.Linear(dec_hidden, 1)     # the pink "LR" box in Fig. 1

        nn.init.normal_(self.emb.weight, std=0.1)

    def forward(self, turb_idx, enc_feats, fut_cov, last_power,
                target=None, teacher_forcing=0.0):
        """
        Description:
            Encode INPUT_LEN steps of history, then autoregressively decode
            `horizon` power values.

        Args:
            turb_idx: (B,) long, ZERO-BASED -- pass TurbID - 1.
            enc_feats: (B, T, n_wspd + n_other) float. Already
                graph-aggregated and normalised by the data pipeline.
            fut_cov: (B, horizon, n_fut) float -- Hour, x, y per future step.
            last_power: (B, 1) float -- y^t_i, the last OBSERVED power. The
                only unhatted quantity the decoder ever receives.
            target: (B, horizon, 1) or None. Required if teacher_forcing > 0.
            teacher_forcing: AMBIGUITY 9. Probability, per step, of feeding the
                ground truth instead of the model's own output. 0.0 matches
                Figure 1's hats; 1.0 trains faster but never lets the model
                practise recovering from its own errors, which over 288 steps
                is exactly the regime it will face at test time.

        Returns:
            (B, horizon, 1) predicted active power.
        """
        B, T, _ = enc_feats.shape
        if teacher_forcing > 0 and target is None:
            raise ValueError("teacher_forcing > 0 requires target")

        g = self.emb(turb_idx)                                # (B, d_emb)
        g_seq = g.unsqueeze(1).expand(B, T, g.size(-1))       # same at all t
        x = torch.cat([g_seq, enc_feats], dim=-1)             # (B, T, 19)

        _, h_n = self.encoder(x)                              # (L, B, 64)
        h = self.bridge(h_n[-1])                              # (B, 32)

        y = last_power
        outs = []
        for t in range(self.horizon):
            step = torch.cat([g, y, fut_cov[:, t]], dim=-1)   # (B, 9)
            h = self.decoder(step, h)
            y = self.head(h)                                  # (B, 1)
            if self.relu_output:
                y = F.relu(y)
            outs.append(y)
            if teacher_forcing > 0:
                use = torch.rand(B, 1, device=y.device) < teacher_forcing
                y = torch.where(use, target[:, t], y)
        return torch.stack(outs, dim=1)                       # (B, horizon, 1)


# --------------------------------------------------------------------------
# loss -- AMBIGUITY 6
# --------------------------------------------------------------------------
class FilterHuberLoss(nn.Module):
    """
    Description:
        Huber loss with the OFFICIAL GRADER'S MASK applied to the targets.
        HIK never state their loss; this is ported from BUAA_BIGSCity, and
        essentially every team above rank 15 does some version of it.

        The point: the grader throws away records that are missing, out of
        range, or curtailed. Training on those records spends capacity fitting
        points that earn nothing and actively distort the fit -- a turbine shut
        off for maintenance reads 0 kW in a 12 m/s wind, and no model should
        learn to reproduce that.

        Masked out:
            Patv < 0
            any of Pab1/2/3 > 89 degrees
            |Wdir| > 180 or |Ndir| > 720
            Patv == 0 while Wspd > 2.5   (curtailed; output is unknowable)

    Note:
        Reduction is mean(loss * mask), dividing by the TOTAL element count
        rather than the valid count -- so masked points shrink the gradient
        magnitude instead of being cleanly excluded. That is what BIGSCity
        shipped. Set exclude_masked=True to divide by the valid count instead.
    """

    def __init__(self, delta=5.0, exclude_masked=False):
        super().__init__()
        self.delta = delta
        self.exclude_masked = exclude_masked

    @staticmethod
    def valid_mask(raw):
        """
        Args:
            raw: (B, horizon, C) UNFILLED target block, with columns
                [Patv, Pab1, Pab2, Pab3, Wdir, Ndir, Wspd].
        Returns:
            (B, horizon, 1) float mask, 1.0 where the grader counts the point.
        """
        patv, pab1, pab2, pab3, wdir, ndir, wspd = [
            raw[..., i:i + 1] for i in range(7)]
        ok = torch.ones_like(patv)
        ok = ok * (~torch.isnan(patv)).float()
        ok = ok * (patv >= 0).float()
        ok = ok * (pab1 <= 89).float() * (pab2 <= 89).float() * (pab3 <= 89).float()
        ok = ok * (wdir.abs() <= 180).float() * (ndir.abs() <= 720).float()
        ok = ok * (~((patv <= 0) & (wspd > 2.5))).float()
        return ok

    def forward(self, pred, target, mask):
        loss = F.huber_loss(pred, torch.nan_to_num(target), delta=self.delta,
                            reduction="none")
        loss = loss * mask
        if self.exclude_masked:
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    B = 8

    for dup in (True, False):
        n_wspd = 4 if dup else 3
        m = DMST(n_wspd=n_wspd, horizon=12)
        turb = torch.randint(0, CAPACITY, (B,))
        enc = torch.randn(B, INPUT_LEN, n_wspd + N_OTHER)
        fut = torch.randn(B, 12, N_FUT)
        last = torch.rand(B, 1) * 1500

        out = m(turb, enc, fut, last)
        assert out.shape == (B, 12, 1), out.shape
        assert (out >= 0).all(), "relu_output should clip negatives"
        print("dup_self=%-5s enc_in=%2d dec_in=%d out=%s"
              % (dup, m.enc_in, m.dec_in, tuple(out.shape)))

    m = DMST(horizon=12)
    turb = torch.randint(0, CAPACITY, (B,))
    enc = torch.randn(B, INPUT_LEN, 4 + N_OTHER)
    fut = torch.randn(B, 12, N_FUT)
    last = torch.rand(B, 1) * 1500
    tgt = torch.rand(B, 12, 1) * 1500

    out = m(turb, enc, fut, last, target=tgt, teacher_forcing=1.0)
    assert out.shape == (B, 12, 1)
    print("teacher forcing path ok")

    raw = torch.stack([
        torch.rand(B, 12) * 1400,          # Patv
        torch.zeros(B, 12),                # Pab1
        torch.zeros(B, 12),                # Pab2
        torch.zeros(B, 12),                # Pab3
        torch.randn(B, 12) * 50,           # Wdir
        torch.randn(B, 12) * 200,          # Ndir
        torch.rand(B, 12) * 15,            # Wspd
    ], dim=-1)
    raw[0, :3, 0] = 0.0                    # curtailed rows
    raw[0, :3, 6] = 9.0
    raw[1, 5, 1] = 95.0                    # pitched out

    mask = FilterHuberLoss.valid_mask(raw)
    print("mask keeps %.1f%% of points; row0 first3 = %s, row1 step5 = %.0f"
          % (100 * mask.mean(), mask[0, :3, 0].tolist(), mask[1, 5, 0]))

    crit = FilterHuberLoss()
    loss = crit(out, raw[..., :1], mask)
    loss.backward()

    grads = {n: p.grad for n, p in m.named_parameters()}
    missing = [n for n, gr in grads.items() if gr is None]
    assert not missing, "no gradient reached: %s" % missing
    touched = int((grads["emb.weight"].abs().sum(1) > 0).sum())
    print("loss %.4f | all params have grad | embedding rows touched %d/%d"
          % (loss.item(), touched, CAPACITY))

    n = sum(p.numel() for p in m.parameters())
    print("parameters: %d total (%d encoder, %d embedding)"
          % (n,
             sum(p.numel() for p in m.encoder.parameters()),
             m.emb.weight.numel()))


if __name__ == "__main__":
    _selftest()
