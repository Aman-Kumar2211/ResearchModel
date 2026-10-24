"""

The manifest records, for every photo: parsed filename fields, SHA-256 of the file,
and the train/val/test split taken from splits.json (box ID -> split). Splitting
by BOX, never by photo, prevents near-duplicate photos leaking across splits.

  python make_manifest.py --images dataset/images --splits splits.json --out manifest.csv
  python make_manifest.py --images dataset/images --verify manifest.csv
"""
import argparse
import json
import os
import sys

import pandas as pd

from config import IMAGE_EXTS, derive_group_condition, parse_filename, sha256_file


def verify(images_dir, manifest_path):
    df = pd.read_csv(manifest_path)
    bad = 0
    for _, row in df.iterrows():
        p = os.path.join(images_dir, row.relpath)
        if not os.path.exists(p):
            print(f"MISSING  {row.relpath}")
            bad += 1
        elif sha256_file(p) != row.sha256:
            print(f"CHANGED  {row.relpath}")
            bad += 1
    extra = set()
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                extra.add(os.path.relpath(os.path.join(root, f), images_dir).replace(os.sep, "/"))
    for r in sorted(extra - set(df.relpath)):
        print(f"NOT IN MANIFEST  {r}")
        bad += 1
    print("dataset matches manifest exactly" if bad == 0 else f"{bad} problem(s) found")
    return bad == 0


def build(images_dir, splits_path, out_path, holdout_phones=()):
    with open(splits_path) as f:
        splits = json.load(f)
    bad_vals = {b: s for b, s in splits.items() if s not in ("train", "val", "test")}
    if bad_vals:
        sys.exit(f"splits.json has invalid split names (use train/val/test): {bad_vals}")

    rows, errors = [], []
    for root, _, files in os.walk(images_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, images_dir).replace(os.sep, "/")
            try:
                rec = parse_filename(fname)
            except ValueError as e:
                errors.append(f"{rel}: {e}")
                continue
            if rec["box"] not in splits:
                errors.append(f"{rel}: box {rec['box']} is not listed in {splits_path}")
                continue
            group, condition = derive_group_condition(rec)
            rows.append({
                "relpath": rel, "sha256": sha256_file(full),
                "box": rec["box"], "label": rec["label"],
                "label_idx": 1 if rec["label"] == "tampered" else 0,
                "tamper": rec["tamper"], "phone": rec["phone"], "lighting": rec["lighting"],
                "angle": rec["angle"], "occlusion": rec["occlusion"], "idx": rec["idx"],
                "group": group, "condition": condition, "split": splits[rec["box"]],
            })
    if errors:
        print("\n".join(errors))
        sys.exit(f"\n{len(errors)} file(s) could not be used. Fix names / splits.json and re-run.")
    if not rows:
        sys.exit("no images found")

    df = pd.DataFrame(rows).sort_values("relpath").reset_index(drop=True)
    if holdout_phones:
        # unseen-phone test: photos from these phones are used only for test boxes; for train/val boxes
        # they are set aside (split='unused') so the model never sees those phones or those boxes through them
        drop = df.phone.isin(set(holdout_phones)) & (df.split != "test")
        df.loc[drop, "split"] = "unused"
        print(f"held-out phone(s) {sorted(holdout_phones)}: {int(drop.sum())} photos of train/val boxes set aside (split=unused)")
    train_phones = set(df[df.split == "train"].phone)
    df["phone_seen"] = df.phone.isin(train_phones)       # False => phone never used in training
    df.to_csv(out_path, index=False)

    # ---- checks and summary --------------------------------------------
    print(f"wrote {out_path}: {len(df)} photos, {df.box.nunique()} boxes\n")
    print(pd.crosstab([df.split, df.group, df.condition], df.label), "\n")
    unused = sorted(set(splits) - set(df.box))
    if unused:
        print(f"WARNING: boxes in splits.json with no photos: {unused}")
    for sp in ("train", "val"):
        sub = df[df.split == sp]
        if len(sub) and (sub.condition != "clean").any():
            print(f"NOTE: {sp} has {int((sub.condition != 'clean').sum())} non-clean photos; "
                  "train.py will ignore them (baseline trains on clean photos only).")
        if len(sub) and sub[sub.condition == "clean"].label.nunique() < 2:
            print(f"WARNING: {sp} clean photos contain only one class")
    for sp in ("train", "test"):
        if not (df.split == sp).any():
            print(f"WARNING: no photos in split '{sp}'")
    print(f"phones used in training: {sorted(train_phones)}; "
          f"unseen test phones: {sorted(set(df.phone) - train_phones)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="dataset/images")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--holdout-phones", default="",
                    help="comma-separated phone letters kept unseen in training, e.g. D")
    ap.add_argument("--verify", metavar="MANIFEST", help="check files on disk against an existing manifest")
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.images, a.verify) else 1)
    build(a.images, a.splits, a.out, [p.strip() for p in a.holdout_phones.split(",") if p.strip()])
