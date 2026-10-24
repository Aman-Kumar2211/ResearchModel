"""Fine-tune one lightweight model on the CLEAN training photos.

Two stages: (1) train only the new 2-class head with the backbone frozen,
(2) fine-tune everything at a low learning rate with early stopping on the
validation boxes. Same manifest + same seed + same package versions + same
device => same result.

  python train.py --model mobilenetv3_small_100 --seed 0
  python train.py --model efficientnet_b0 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CLASS_NAMES, IMG_SIZE, metrics, sha256_file
from torch_utils import (SealDataset, build_model, env_info, eval_transform, set_seed,
                         train_transform)


def run_epoch(model, loader, criterion, device, optimizer=None, frozen_backbone=False):
    """One pass over loader. With an optimizer = training; without = evaluation."""
    training = optimizer is not None
    # With a frozen backbone the whole network stays in eval mode, so BatchNorm
    # statistics and dropout stay exactly as pretrained; only the head learns.
    model.train(training and not frozen_backbone)
    total_loss, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            ys.append(y.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    m = metrics(ys, ps)
    return total_loss / len(ys), m["accuracy"], m["f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="timm name, e.g. mobilenetv3_small_100, efficientnet_b0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--images", default="dataset/images")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cpu", help="cpu (most reproducible) or cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--stage1-epochs", type=int, default=4)
    ap.add_argument("--stage2-epochs", type=int, default=12)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3, help="early-stopping patience (stage 2, on val loss)")
    ap.add_argument("--drop-rate", type=float, default=0.2)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="SMOKE TEST ONLY: random weights, no download. Never use for reported results.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    run_dir = os.path.join(args.out, f"{args.model}_seed{args.seed}")
    if os.path.exists(os.path.join(run_dir, "best.pt")) and not args.overwrite:
        sys.exit(f"{run_dir} already exists (use --overwrite to retrain)")
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    # ---- data: clean photos of train / val boxes only -------------------------------
    df = pd.read_csv(args.manifest)
    train_df = df[(df.split == "train") & (df.condition == "clean")]
    val_df = df[(df.split == "val") & (df.condition == "clean")]
    if train_df.label.nunique() < 2:
        sys.exit("training set needs both classes (intact and tampered)")
    have_val = len(val_df) > 0
    print(f"train: {len(train_df)} photos / {train_df.box.nunique()} boxes | "
          f"val: {len(val_df)} photos / {val_df.box.nunique()} boxes")
    if not have_val:
        print("no validation photos: training for a fixed number of epochs, saving the last model")

    model, data_cfg, weights = build_model(args.model, pretrained=not args.no_pretrained,
                                           drop_rate=args.drop_rate)
    mean, std = data_cfg["mean"], data_cfg["std"]
    model.to(device)

    gen = torch.Generator().manual_seed(args.seed)      # controls shuffling order
    train_loader = DataLoader(SealDataset(train_df, args.images, train_transform(mean, std)),
                              batch_size=args.batch_size, shuffle=True, generator=gen, num_workers=0)
    val_loader = (DataLoader(SealDataset(val_df, args.images, eval_transform(mean, std)),
                             batch_size=32, shuffle=False, num_workers=0) if have_val else None)

    counts = np.bincount(train_df.label_idx.values, minlength=2)
    class_w = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    history, best = [], {"val_loss": float("inf"), "state": None, "epoch": None}
    bad_epochs = 0

    def log(stage, epoch, lr, tr, va):
        row = {"stage": stage, "epoch": epoch, "lr": lr,
               "train_loss": tr[0], "train_acc": tr[1],
               "val_loss": va[0] if va else np.nan, "val_acc": va[1] if va else np.nan,
               "val_f1": va[2] if va else np.nan}
        history.append(row)
        print(f"stage{stage} ep{epoch:02d} lr={lr:.1e} train_loss={tr[0]:.4f} acc={tr[1]:.3f}"
              + (f" | val_loss={va[0]:.4f} acc={va[1]:.3f} f1={va[2]:.3f}" if va else ""))

    def maybe_save_best(va, stage, epoch):
        nonlocal bad_epochs
        if va and va[0] < best["val_loss"] - 1e-6:
            best.update(val_loss=va[0], epoch=(stage, epoch),
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad_epochs = 0
            return True
        return False

    # ---- stage 1: head only -------------------------------------------------------
    for p in backbone_params:
        p.requires_grad = False
    opt1 = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
    for ep in range(1, args.stage1_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, opt1, frozen_backbone=True)
        va = run_epoch(model, val_loader, criterion, device) if have_val else None
        log(1, ep, args.head_lr, tr, va)
        maybe_save_best(va, 1, ep)

    # ---- stage 2: fine-tune everything, cosine LR, early stopping ------------------------
    for p in backbone_params:
        p.requires_grad = True
    opt2 = torch.optim.AdamW([{"params": backbone_params, "lr": args.backbone_lr},
                              {"params": head_params, "lr": args.head_lr}],
                             weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, args.stage2_epochs))
    for ep in range(1, args.stage2_epochs + 1):
        tr = run_epoch(model, train_loader, criterion, device, opt2)
        va = run_epoch(model, val_loader, criterion, device) if have_val else None
        log(2, ep, opt2.param_groups[0]["lr"], tr, va)
        sched.step()
        if have_val:
            if not maybe_save_best(va, 2, ep):
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"early stopping (no val-loss improvement for {args.patience} epochs)")
                    break

    final_state = best["state"] if (have_val and best["state"] is not None) else \
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(final_state, os.path.join(run_dir, "best.pt"))
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    config = {
        "args": vars(args), "img_size": IMG_SIZE, "class_names": CLASS_NAMES,
        "mean": list(mean), "std": list(std), "pretrained_weights": weights,
        "manifest_sha256": sha256_file(args.manifest),
        "train_boxes": sorted(train_df.box.unique()), "val_boxes": sorted(val_df.box.unique()),
        "n_train": int(len(train_df)), "n_val": int(len(val_df)),
        "train_label_counts": {CLASS_NAMES[i]: int(c) for i, c in enumerate(counts)},
        "best_checkpoint": {"stage_epoch": best["epoch"],
                            "val_loss": None if best["state"] is None else best["val_loss"]},
        "env": env_info(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"saved {run_dir}/best.pt, history.csv, config.json")


if __name__ == "__main__":
    main()
