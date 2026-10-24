"""Combine all runs: mean +/- std over seeds, efficiency table, accuracy-vs-severity figure.

  python aggregate.py            # reads runs/*/results, writes results/
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

METRICS = ["accuracy", "precision", "recall", "f1", "fnr"]


def load_runs(runs_dir):
    frames, speeds = [], []
    for cfg_path in sorted(glob.glob(os.path.join(runs_dir, "*", "config.json"))):
        run = os.path.dirname(cfg_path)
        summ = os.path.join(run, "results", "summary.csv")
        if not os.path.exists(summ):
            print(f"skipping {run} (not evaluated yet)")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        s = pd.read_csv(summ)
        s["model"], s["seed"] = cfg["args"]["model"], cfg["args"]["seed"]
        frames.append(s)
        sp = os.path.join(run, "results", "speed.json")
        if os.path.exists(sp):
            with open(sp) as f:
                d = json.load(f)
            d.update(model=cfg["args"]["model"], seed=cfg["args"]["seed"])
            speeds.append(d)
    if not frames:
        raise SystemExit("no evaluated runs found")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(speeds)


def make_figures(agg, out_dir):
    """Accuracy vs corruption severity, one figure per box group. agg has *_mean/*_std columns."""
    syn = agg[agg.analysis == "synthetic"].copy()
    parts = syn.key.str.split("|", expand=True)
    syn["group"], syn["corruption"], syn["severity"] = parts[0], parts[1], parts[2].astype(int)
    base = agg[(agg.analysis == "real_set") & (agg.key == "identical|clean")].set_index("model")
    for group in sorted(syn.group.unique()):
        sub = syn[syn.group == group]
        corruptions = sorted(sub.corruption.unique())
        cols = 4
        rows = -(-len(corruptions) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False, sharey=True)
        for ax, c in zip(axes.ravel(), corruptions):
            for model, mdf in sub[sub.corruption == c].groupby("model"):
                mdf = mdf.sort_values("severity")
                x, y, e = list(mdf.severity), list(mdf.accuracy_mean), list(mdf.accuracy_std.fillna(0))
                if group == "identical" and model in base.index:
                    x, y, e = [0] + x, [base.loc[model, "accuracy_mean"]] + y, [base.loc[model, "accuracy_std"]] + e
                ax.errorbar(x, y, yerr=e, marker="o", capsize=2, label=model)
            ax.set_title(c, fontsize=9)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("severity (0 = clean)")
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(corruptions):]:
            ax.axis("off")
        axes[0][0].set_ylabel("accuracy")
        axes[0][0].legend(fontsize=7)
        fig.suptitle(f"Accuracy vs corruption severity ({group} boxes)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"accuracy_vs_severity_{group}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"wrote {path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    big, speeds = load_runs(a.runs)
    grp = big.groupby(["model", "analysis", "key"])
    agg = grp[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = grp["seed"].nunique()
    agg["n_photos"] = grp["n"].first()
    agg = agg.reset_index()
    agg.to_csv(os.path.join(a.out, "summary_mean_std.csv"), index=False)
    print(f"wrote {a.out}/summary_mean_std.csv")

    if not speeds.empty:
        eff = speeds.groupby("model").mean(numeric_only=True).drop(columns=["seed"]).reset_index()
        eff.to_csv(os.path.join(a.out, "efficiency.csv"), index=False)
        print(f"wrote {a.out}/efficiency.csv")

    make_figures(agg, a.out)


if __name__ == "__main__":
    main()
