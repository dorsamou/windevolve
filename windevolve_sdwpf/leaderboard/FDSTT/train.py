# -*- coding: utf-8 -*-
"""
path: windevolve_sdwpf/leaderboard/FDSTT/train.py

Description:
    Trains the DMST module of FDSTT. Two modes, matching this repo's
    refit-after-selection protocol:

    SEARCH        --fit-dir fit_val
        Fits on val_train.csv (days 1-197). Carves an INNER split off the end
        of that period for early stopping, so the 24 val windows stay a true
        holdout and remain honest for cross-model comparison. Writes the epoch
        that early stopping chose into the checkpoint.

    FINAL REFIT   --fit-dir fit_full --epochs-fixed N
        Fits on all 245 days. There is no held-out data left, so early
        stopping is disabled by construction; N comes from the search run.

    The inner-split score is reported on the competition scale -- per-turbine
    0.5*(RMSE+MAE) summed over turbines and divided by 1000 -- so it is
    directly readable against the ~45 MW leaderboard numbers rather than being
    a dimensionless loss.

Usage:
    # 1. search fit
    python leaderboard/FDSTT/train.py --fit-dir fit_val --out ckpt_val

    # 2. final refit, N taken from ckpt_val/best.pt
    python leaderboard/FDSTT/train.py --fit-dir fit_full --out ckpt_full \\
        --fit-file $SDWPF_TRAIN_FILE --epochs-fixed 23
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_HERE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from data import load_panel, load_fit, FDSTTWindows, CAPACITY, STEPS_PER_DAY  # noqa: E402
from dmst import DMST, FilterHuberLoss, INPUT_LEN, OUTPUT_LEN, N_OTHER        # noqa: E402


def device_of(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def split_inner(ds, inner_days, stride):
    """
    Description:
        Split the dataset by ORIGIN, not at random. Random splitting would put
        near-identical overlapping windows on both sides and make the inner
        score meaningless -- consecutive origins share 431 of 432 input steps.

    Args:
        inner_days: days at the end of the fit period reserved for early
            stopping. 0 disables the split (refit mode).
    Returns:
        (train_subset, inner_subset or None)
    """
    if inner_days <= 0:
        return ds, None
    origins = np.array(sorted({o for o, _ in ds.index}))
    cutoff = origins.max() - inner_days * STEPS_PER_DAY
    tr = [i for i, (o, _) in enumerate(ds.index) if o <= cutoff]
    va = [i for i, (o, _) in enumerate(ds.index) if o > cutoff]
    if not va:
        raise ValueError("inner split empty; reduce --inner-days")
    return Subset(ds, tr), Subset(ds, va)


@torch.no_grad()
def competition_scale_score(model, loader, power_scale, dev):
    """
    Description:
        Masked 0.5*(RMSE+MAE) accumulated per turbine, then summed across
        turbines and divided by 1000 -- the official metric's shape, so the
        number reads on the same scale as the leaderboard.
    Returns:
        (score, mae, rmse) in MW / kW / kW.
    """
    model.eval()
    se = np.zeros(CAPACITY); ae = np.zeros(CAPACITY); cnt = np.zeros(CAPACITY)
    for turb, enc, fut, last, raw in loader:
        enc, fut, last = enc.to(dev), fut.to(dev), last.to(dev)
        pred = model(turb.to(dev), enc, fut, last) * power_scale
        mask = FilterHuberLoss.valid_mask(raw).to(dev)
        tgt = torch.nan_to_num(raw[..., :1]).to(dev)
        err = (pred - tgt) * mask
        t = turb.numpy()
        np.add.at(se, t, (err ** 2).sum((1, 2)).cpu().numpy())
        np.add.at(ae, t, err.abs().sum((1, 2)).cpu().numpy())
        np.add.at(cnt, t, mask.sum((1, 2)).cpu().numpy())
    ok = cnt > 0
    rmse = np.sqrt(se[ok] / cnt[ok]); mae = ae[ok] / cnt[ok]
    model.train()
    return float((0.5 * (rmse + mae)).sum() / 1000), float(mae.mean()), float(rmse.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-dir", default=os.path.join(_HERE, "fit"))
    ap.add_argument("--fit-file", default=None,
                    help="CSV to build windows from. Default: the file "
                         "recorded in <fit-dir>/meta.json, so windows and "
                         "fitted artifacts can never come from different data.")
    ap.add_argument("--out", default=os.path.join(_HERE, "ckpt"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epochs-fixed", type=int, default=0,
                    help="refit mode: train exactly N epochs, no early stopping")
    ap.add_argument("--inner-days", type=int, default=17,
                    help="days at the end of the fit period for early stopping")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)     # paper Section 4.1.2
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--teacher-forcing", type=float, default=0.5)
    ap.add_argument("--horizon", type=int, default=OUTPUT_LEN)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device_of(args.device)
    os.makedirs(args.out, exist_ok=True)

    art = load_fit(args.fit_dir)
    src = args.fit_file or art["fit_file"]
    print("fit artifacts : {}".format(args.fit_dir))
    print("windows from  : {} ({} days)".format(os.path.basename(src),
                                                art["fit_days"]))

    refit = args.epochs_fixed > 0
    if refit:
        print("MODE: final refit, {} epochs, early stopping disabled"
              .format(args.epochs_fixed))
    else:
        print("MODE: search, early stopping on last {} days"
              .format(args.inner_days))

    panel = load_panel(src)
    ds = FDSTTWindows(panel, art, horizon=args.horizon, stride=args.stride,
                      dup_self=art["dup_self"])
    tr, inner = split_inner(ds, 0 if refit else args.inner_days, args.stride)
    print("samples: {} train, {} inner".format(
        len(tr), 0 if inner is None else len(inner)))

    dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    pin_memory=(dev.type == "cuda"))
    dl_inner = (None if inner is None else
                DataLoader(inner, batch_size=args.batch_size,
                           num_workers=args.workers))

    ps = float(art["power_scale"])
    n_wspd = 4 if art["dup_self"] else 3
    model = DMST(n_wspd=n_wspd, n_other=N_OTHER, horizon=args.horizon).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = FilterHuberLoss(delta=5.0 / ps)     # paper's delta = 5 kW, rescaled
    print("model: {} parameters | power_scale {:.1f} kW | device {}".format(
        sum(p.numel() for p in model.parameters()), ps, dev))

    best, best_epoch, bad, hist = float("inf"), 0, 0, []
    n_epochs = args.epochs_fixed if refit else args.epochs

    for ep in range(1, n_epochs + 1):
        t0, tot, nb = time.time(), 0.0, 0
        for turb, enc, fut, last, raw in dl:
            turb, enc = turb.to(dev), enc.to(dev)
            fut, last, raw = fut.to(dev), last.to(dev), raw.to(dev)
            tgt = torch.nan_to_num(raw[..., :1]) / ps
            mask = FilterHuberLoss.valid_mask(raw)

            opt.zero_grad()
            pred = model(turb, enc, fut, last, target=tgt,
                         teacher_forcing=args.teacher_forcing)
            loss = crit(pred, tgt, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1

        line = {"epoch": ep, "train_loss": tot / max(nb, 1),
                "secs": round(time.time() - t0, 1)}

        if dl_inner is not None:
            s, mae, rmse = competition_scale_score(model, dl_inner, ps, dev)
            line.update(inner_score=round(s, 4), inner_mae=round(mae, 2),
                        inner_rmse=round(rmse, 2))
            if s < best - 1e-4:
                best, best_epoch, bad = s, ep, 0
                torch.save({"model": model.state_dict(), "epoch": ep,
                            "inner_score": s, "args": vars(args),
                            "n_wspd": n_wspd, "power_scale": ps,
                            "fit_dir": os.path.abspath(args.fit_dir)},
                           os.path.join(args.out, "best.pt"))
            else:
                bad += 1
        else:
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "inner_score": None, "args": vars(args),
                        "n_wspd": n_wspd, "power_scale": ps,
                        "fit_dir": os.path.abspath(args.fit_dir)},
                       os.path.join(args.out, "best.pt"))

        hist.append(line)
        print("  " + json.dumps(line))
        with open(os.path.join(args.out, "history.json"), "w") as fh:
            json.dump(hist, fh, indent=2)

        if dl_inner is not None and bad >= args.patience:
            print("early stop at epoch {} (best {} @ epoch {})".format(
                ep, round(best, 4), best_epoch))
            break

    if dl_inner is not None:
        print("\nBEST inner score {:.4f} at epoch {}".format(best, best_epoch))
        print("For the final refit, pass:  --epochs-fixed {}".format(best_epoch))
    print("checkpoint: {}".format(os.path.join(args.out, "best.pt")))


if __name__ == "__main__":
    main()
