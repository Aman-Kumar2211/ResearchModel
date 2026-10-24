"""Evaluate one trained run on the test boxes (no retraining, no randomness).

Evaluates every test photo (a) as captured, and (b) for CLEAN test photos, with each
synthetic corruption at severities 1-3. Writes into <run>/results/:
  predictions.csv  one row per (photo, corruption, severity), with P(tampered)
  summary.csv      metrics for every breakdown used in the paper
  speed.json       parameters, model size, CPU latency

  python evaluate.py --run runs/mobilenetv3_small_100_seed0
"""
import argparse
import copy
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import CORRUPTION_SEED, SEVERITIES, metrics, seed_for, sha256_file
from corruptions import CORRUPTIONS, apply_corruption
from torch_utils import build_model, eval_transform, load_image, set_seed

META_COLS = ["relpath", "box", "label", "label_idx", "tamper", "phone", "lighting",
             "angle", "occlusion", "group", "condition", "phone_seen"]


class EvalItems(Dataset):
    """(photo, corruption, severity) items; corruption is applied deterministically on the fly."""

    def __init__(self, df, items, images_dir, size, mean, std):
        self.df, self.items, self.images_dir, self.size = df, items, images_dir, size
        self.tf = eval_transform(mean, std)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        ridx, corruption, sev = self.items[k]
        relpath = self.df.at[ridx, "relpath"]
        img = load_image(os.path.join(self.images_dir, relpath), self.size)
        if corruption != "none":
            img = apply_corruption(img, corruption, sev, seed_for(relpath, corruption, sev))
        return self.tf(img), k


def summarize(pred):
    rows = []

    def add(analysis, frame, by):
        for key, g in frame.groupby(by):
            key = key if isinstance(key, tuple) else (key,)
            rows.append({"analysis": analysis, "key": "|".join(str(k) for k in key),
                         **metrics(g.label_idx.values, g.pred.values)})

    real = pred[pred.corruption == "none"]
    syn = pred[pred.corruption != "none"]
    clean = real[real.condition == "clean"]
    add("real_set", real, ["group", "condition"])
    add("real_occlusion", real, ["group", "occlusion"])
    add("real_lighting", real, ["group", "lighting"])
    add("real_phone", real, ["group", "phone"])
    add("real_phone_seen", clean, ["group", "phone_seen"])
    add("real_tamper_type", clean[clean.label == "tampered"], ["group", "tamper"])  # recall per type
    add("synthetic", syn, ["group", "corruption", "severity"])
    return pd.DataFrame(rows)


def benchmark_speed(model, size, runs, warmup=10):
    m = copy.deepcopy(model).cpu().eval()
    torch.set_num_threads(1)
    x = torch.randn(1, 3, size, size)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        for _ in range(runs):
            t = time.perf_counter()
            m(x)
            times.append((time.perf_counter() - t) * 1000.0)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    return {"params_millions": sum(p.numel() for p in m.parameters()) / 1e6,
            "state_dict_mb": buf.tell() / 1e6,
            "cpu_latency_ms_mean": float(np.mean(times)), "cpu_latency_ms_std": float(np.std(times)),
            "note": "single image, 1 CPU thread; timings vary between machines and runs"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory created by train.py")
    ap.add_argument("--manifest", default=None, help="defaults to the manifest used for training")
    ap.add_argument("--images", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.5, help="predict tampered if P(tampered) >= this")
    ap.add_argument("--speed-runs", type=int, default=100)
    ap.add_argument("--no-pretrained", action="store_true", help="only to match a --no-pretrained smoke-test run")
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.json")) as f:
        cfg = json.load(f)
    manifest = args.manifest or cfg["args"]["manifest"]
    images_dir = args.images or cfg["args"]["images"]
    if sha256_file(manifest) != cfg["manifest_sha256"]:
        sys.exit("manifest.csv differs from the one used for training. Re-train, or pass the original manifest.")

    set_seed(0)                                   # nothing here is random, but be explicit
    device = torch.device(args.device)
    model, _, _ = build_model(cfg["args"]["model"], pretrained=False, drop_rate=cfg["args"]["drop_rate"])
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pt"), map_location="cpu"))
    model.to(device).eval()

    df = pd.read_csv(manifest)
    test = df[df.split == "test"]
    if test.empty:
        sys.exit("no test photos in manifest")
    items = [(i, "none", 0) for i in test.index]
    clean_idx = test[test.condition == "clean"].index
    for corruption in CORRUPTIONS:
        for sev in SEVERITIES:
            items += [(i, corruption, sev) for i in clean_idx]
    print(f"{len(test)} test photos -> {len(items)} evaluations "
          f"({len(CORRUPTIONS)} corruptions x {len(SEVERITIES)} severities on {len(clean_idx)} clean photos)")

    loader = DataLoader(EvalItems(df, items, images_dir, cfg["img_size"], cfg["mean"], cfg["std"]),
                        batch_size=32, shuffle=False, num_workers=0)
    probs = np.zeros(len(items), dtype=np.float64)
    with torch.no_grad():
        for x, k in loader:
            probs[k.numpy()] = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()

    pred = df.loc[[i for i, _, _ in items], META_COLS].reset_index(drop=True)
    pred["corruption"] = [c for _, c, _ in items]
    pred["severity"] = [s for _, _, s in items]
    pred["prob_tampered"] = probs
    pred["pred"] = (probs >= args.threshold).astype(int)
    pred["correct"] = (pred.pred == pred.label_idx).astype(int)

    out = os.path.join(args.run, "results")
    os.makedirs(out, exist_ok=True)
    pred.to_csv(os.path.join(out, "predictions.csv"), index=False)
    summary = summarize(pred)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    with open(os.path.join(out, "speed.json"), "w") as f:
        json.dump(benchmark_speed(model, cfg["img_size"], args.speed_runs), f, indent=2)

    show = summary[summary.analysis == "real_set"][["key", "n", "accuracy", "f1", "fnr"]]
    print("\nreal photos as captured (group|condition):\n", show.to_string(index=False))
    print(f"\nwrote {out}/predictions.csv, summary.csv, speed.json")


if __name__ == "__main__":
    main()
