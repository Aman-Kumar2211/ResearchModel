"""Aggregate a grouped k-fold experiment.

Seeds are averaged inside each fold first; then mean +/- std is taken across folds, so the
reported spread reflects "which boxes were held out" (the main source of uncertainty with so
few boxes), not just training randomness.

  python aggregate_folds.py              # reads runs_kfold/fold*/*/results, writes results_kfold/
"""
import argparse
import glob
import json
import os

import pandas as pd

from aggregate import METRICS, make_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_kfold")
    ap.add_argument("--out", default="results_kfold")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames, speeds = [], []
    for cfg_path in sorted(glob.glob(os.path.join(a.runs, "fold*", "*", "config.json"))):
        run = os.path.dirname(cfg_path)
        fold = os.path.basename(os.path.dirname(run))
        summ = os.path.join(run, "results", "summary.csv")
        if not os.path.exists(summ):
            print(f"skipping {run} (not evaluated yet)")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        s = pd.read_csv(summ)
        s["model"], s["seed"], s["fold"] = cfg["args"]["model"], cfg["args"]["seed"], fold
        frames.append(s)
        sp = os.path.join(run, "results", "speed.json")
        if os.path.exists(sp):
            with open(sp) as f:
                speeds.append({**json.load(f), "model": cfg["args"]["model"]})
    if not frames:
        raise SystemExit("no evaluated runs found")
    big = pd.concat(frames, ignore_index=True)

    # 1) average seeds inside each fold
    per_fold = (big.groupby(["model", "fold", "analysis", "key"])[METRICS + ["n"]]
                .mean().reset_index())
    per_fold.to_csv(os.path.join(a.out, "summary_per_fold.csv"), index=False)

    # 2) mean / std across folds (a fold missing a key, e.g. a tamper type, is simply not counted)
    grp = per_fold.groupby(["model", "analysis", "key"])
    agg = grp[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_folds"] = grp["fold"].nunique()
    agg["n_photos"] = grp["n"].mean()          # average photos per fold
    agg = agg.reset_index()
    agg.to_csv(os.path.join(a.out, "summary_mean_std_over_folds.csv"), index=False)
    print(f"wrote {a.out}/summary_per_fold.csv and summary_mean_std_over_folds.csv "
          f"({per_fold.fold.nunique()} folds, {big.seed.nunique()} seed(s))")

    if speeds:
        eff = pd.DataFrame(speeds).groupby("model").mean(numeric_only=True).reset_index()
        eff.to_csv(os.path.join(a.out, "efficiency.csv"), index=False)
        print(f"wrote {a.out}/efficiency.csv")
    make_figures(agg, a.out)


if __name__ == "__main__":
    main()
