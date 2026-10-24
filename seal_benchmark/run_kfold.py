"""Cross-platform replacement for run_kfold.sh (works on Windows, no bash needed).

  python run_kfold.py                      # 5 folds x 2 models x 1 seed
  python run_kfold.py --dry-run            # print the commands only
  python run_kfold.py --models mobilenetv3_small_100 --train-args "--stage2-epochs 30 --patience 5"
"""
import argparse
import os
import shlex
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset/images")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", nargs="+", default=["mobilenetv3_small_100", "efficientnet_b0"])
    ap.add_argument("--seeds", nargs="+", default=["0"])
    ap.add_argument("--holdout-phones", default="D", help='phone(s) kept unseen in training; "" to disable')
    ap.add_argument("--train-args", default="", help='extra train.py options, e.g. "--stage2-epochs 30 --patience 5"')
    ap.add_argument("--skip-existing", action="store_true", help="skip runs that already have a trained model (resume)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    py = sys.executable

    def run(cmd):
        print("\n>>", " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True)

    run([py, "make_manifest.py", "--images", a.images, "--splits", a.splits, "--out", "manifest.csv"])
    run([py, "make_folds.py", "--manifest", "manifest.csv", "--folds", str(a.folds),
         "--holdout-phones", a.holdout_phones, "--out", "folds"])
    for f in range(a.folds):
        for m in a.models:
            for s in a.seeds:
                run_dir = os.path.join("runs_kfold", f"fold{f}", f"{m}_seed{s}")
                if a.skip_existing and os.path.exists(os.path.join(run_dir, "results", "summary.csv")):
                    print(f"\nskipping {run_dir} (already done)")
                    continue
                run([py, "train.py", "--model", m, "--seed", s, "--images", a.images,
                     "--manifest", os.path.join("folds", f"fold{f}", "manifest.csv"),
                     "--out", os.path.join("runs_kfold", f"fold{f}"), "--overwrite"] + shlex.split(a.train_args))
                run([py, "evaluate.py", "--run", run_dir])
    run([py, "aggregate_folds.py"])


if __name__ == "__main__":
    main()
