"""Create grouped k-fold manifests over the IDENTICAL boxes.

Each fold: 'test' = the fold's identical boxes + ALL different boxes (different boxes are
test-only in every fold), 'val' = boxes borrowed from the following fold(s), 'train' = the rest.
Splitting is by box, never by photo. Boxes are spread so each fold's test boxes cover
different tamper types where possible.

  python make_folds.py --manifest manifest.csv --folds 5 --out folds
  -> folds/fold0/manifest.csv ... folds/fold4/manifest.csv, folds/folds.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-boxes", type=int, default=1, help="validation boxes per fold (0 = none)")
    ap.add_argument("--seed", type=int, default=0, help="only used to order boxes before assigning folds")
    ap.add_argument("--holdout-phones", default="",
                    help="comma-separated phones that must stay unseen in training (e.g. D). Their photos are "
                         "used only for the fold's test boxes; photos of train/val boxes are dropped (split='unused') "
                         "so the unseen-phone test does not leak the same physical box.")
    ap.add_argument("--out", default="folds")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    ident = df[df.group == "identical"]
    boxes = sorted(ident.box.unique())
    k = a.folds
    if not (2 <= k <= len(boxes)):
        sys.exit(f"--folds must be between 2 and the number of identical boxes ({len(boxes)})")

    # tamper type of each box (the non-'none' type among its tampered photos)
    ttype = {}
    for b in boxes:
        t = sorted(ident[(ident.box == b) & (ident.label == "tampered")].tamper.unique())
        if len(t) > 1:
            print(f"NOTE: box {b} has several tamper types {t}; using '{t[0]}' for fold balancing")
        ttype[b] = t[0] if t else "none"

    # order boxes by tamper type (shuffled inside each type), then deal round-robin into folds
    rng = np.random.RandomState(a.seed)
    ordered = []
    for t in sorted(set(ttype.values())):
        grp = [b for b in boxes if ttype[b] == t]
        rng.shuffle(grp)
        ordered += grp
    folds = [[] for _ in range(k)]
    for i, b in enumerate(ordered):
        folds[i % k].append(b)

    summary = {}
    os.makedirs(a.out, exist_ok=True)
    for f in range(k):
        test_boxes = folds[f]
        following = [b for j in range(1, k) for b in folds[(f + j) % k]]
        val_boxes = []
        for _ in range(a.val_boxes):
            rest = [b for b in following if b not in val_boxes]
            if not rest:
                break
            # first candidate that still leaves every tamper type (present outside the test fold) in training
            need = {ttype[b] for b in boxes if b not in test_boxes and ttype[b] != "none"}
            pick = next((c for c in rest
                         if need <= {ttype[b] for b in boxes
                                     if b not in test_boxes and b not in val_boxes and b != c}), rest[0])
            val_boxes.append(pick)
        train_boxes = [b for b in boxes if b not in test_boxes and b not in val_boxes]
        if not train_boxes:
            sys.exit("no training boxes left: lower --folds or --val-boxes")

        m = df.copy()
        split_of = {**{b: "test" for b in test_boxes}, **{b: "val" for b in val_boxes},
                    **{b: "train" for b in train_boxes}}
        m["split"] = [("test" if g == "different" else split_of[b]) for b, g in zip(m.box, m.group)]
        holdout = {p.strip() for p in a.holdout_phones.split(",") if p.strip()}
        if holdout:
            drop = m.phone.isin(holdout) & (m.split != "test")
            m.loc[drop, "split"] = "unused"          # ignored by train.py and evaluate.py
        train_phones = set(m[m.split == "train"].phone)
        m["phone_seen"] = m.phone.isin(train_phones)

        tr = m[(m.split == "train") & (m.condition == "clean")]
        if tr.label.nunique() < 2:
            sys.exit(f"fold {f}: clean training photos contain only one class")
        missing = set(ttype[b] for b in boxes if ttype[b] != "none") - set(tr[tr.label == "tampered"].tamper)
        fold_dir = os.path.join(a.out, f"fold{f}")
        os.makedirs(fold_dir, exist_ok=True)
        m.to_csv(os.path.join(fold_dir, "manifest.csv"), index=False)

        summary[f"fold{f}"] = {"test_identical": test_boxes, "val": val_boxes, "train": train_boxes,
                               "test_different": sorted(m[m.group == "different"].box.unique())}
        types = ", ".join(f"{b}({ttype[b]})" for b in test_boxes)
        print(f"fold{f}: test {types} | val {val_boxes} | train {len(train_boxes)} boxes, "
              f"{len(tr)} clean photos"
              + (f" | WARNING: no training example of tamper type(s) {sorted(missing)}" if missing else ""))

    with open(os.path.join(a.out, "folds.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {a.out}/fold*/manifest.csv and {a.out}/folds.json "
          f"(different boxes are test-only in every fold)")


if __name__ == "__main__":
    main()
