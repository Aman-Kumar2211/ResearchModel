"""Turn photos sorted into folders into correctly named dataset files.

Expected input layout (one folder per condition; any camera file names inside):

  raw/<box>/<state>/<phone>/<lighting>_<angle>_<occlusion>/IMG_0001.jpg

  e.g.  raw/I03/tampered/B/indoor_straight_finger3/IMG_2041.jpg
        raw/I03/intact/A/daylight_offaxis_none/IMG_2007.jpg

Output: dataset/images/I03_tampered_cut_B_indoor_straight_finger3_01.jpg  (tamper type taken from --tampers)

  python organize_photos.py --raw raw --tampers tampers.csv --dry-run     # preview
  python organize_photos.py --raw raw --tampers tampers.csv               # copy (originals kept)

Writes rename_log.csv (original -> new name) so every dataset file can be traced back.
"""
import argparse
import csv
import os
import shutil
import sys

from PIL import Image

from config import IMAGE_EXTS, OCCLUSION_ALIASES, TAMPER_ALIASES, parse_filename


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="dataset/images")
    ap.add_argument("--tampers", default="tampers.csv", help="CSV with columns box,tamper")
    ap.add_argument("--log", default="rename_log.csv")
    ap.add_argument("--move", action="store_true", help="move instead of copy")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    with open(a.tampers, newline="") as f:
        tamper_of = {r["box"].strip(): r["tamper"].strip() for r in csv.DictReader(f)}

    plan, errors = [], []
    for box in sorted(os.listdir(a.raw)):
        bdir = os.path.join(a.raw, box)
        if not os.path.isdir(bdir):
            continue
        for state in sorted(os.listdir(bdir)):
            sdir = os.path.join(bdir, state)
            if not os.path.isdir(sdir):
                continue
            for phone in sorted(os.listdir(sdir)):
                pdir = os.path.join(sdir, phone)
                if not os.path.isdir(pdir):
                    continue
                for cond in sorted(os.listdir(pdir)):
                    cdir = os.path.join(pdir, cond)
                    if not os.path.isdir(cdir):
                        continue
                    parts = cond.split("_")
                    if len(parts) != 3:
                        errors.append(f"{cdir}: condition folder must be lighting_angle_occlusion")
                        continue
                    lighting, angle, occlusion = parts
                    occlusion = OCCLUSION_ALIASES.get(occlusion, occlusion)
                    if state == "intact":
                        tamper = "none"
                    elif box in tamper_of:
                        tamper = tamper_of[box]
                    else:
                        errors.append(f"{bdir}: box {box} not in {a.tampers}")
                        continue
                    files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(IMAGE_EXTS))
                    tamper = TAMPER_ALIASES.get(tamper, tamper)
                    for k, fname in enumerate(files, start=1):
                        ext = os.path.splitext(fname)[1].lower()
                        new = f"{box}_{state}_{tamper}_{phone}_{lighting}_{angle}_{occlusion}_{k:02d}{ext}"
                        try:
                            parse_filename(new)
                        except ValueError as e:
                            errors.append(f"{os.path.join(cdir, fname)} -> {new}: {e}")
                            continue
                        plan.append((os.path.join(cdir, fname), new))
                    other = [f for f in os.listdir(cdir) if not f.lower().endswith(IMAGE_EXTS)]
                    if other:
                        errors.append(f"{cdir}: ignoring non-JPG/PNG files {other[:3]} "
                                      "(HEIC? set the phone camera to 'Most Compatible'/JPEG, or convert first)")
    if errors:
        print("PROBLEMS:\n  " + "\n  ".join(errors))
    names = [n for _, n in plan]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        sys.exit(f"duplicate destination names: {sorted(dupes)[:5]}")
    if not plan:
        sys.exit("no photos found")

    # quick quality checks on the originals
    no_exif = landscape = portrait = small = 0
    for src, _ in plan:
        with Image.open(src) as im:
            w, h = im.size
            if not im.getexif():
                no_exif += 1
            orient = im.getexif().get(274, 1)
            if orient in (5, 6, 7, 8):
                w, h = h, w
            landscape += w > h
            portrait += h >= w
            small += min(w, h) < 600
    print(f"{len(plan)} photos | portrait {portrait}, landscape {landscape} | "
          f"{no_exif} without EXIF | {small} with short side < 600 px")
    if no_exif:
        print("  NOTE: photos without EXIF were probably sent through WhatsApp or similar (metadata stripped and "
              "re-compressed). Use original files (cable, Drive 'original quality', or send as Document).")
    if small:
        print("  NOTE: very small photos lose detail; re-export the originals if possible.")
    if landscape and portrait:
        print("  NOTE: mixed portrait/landscape. Check that every photo shows the tape vertically.")

    if a.dry_run:
        for src, new in plan[:15]:
            print(f"  {src}  ->  {new}")
        print("(dry run, nothing copied)" if len(plan) <= 15 else f"... and {len(plan) - 15} more (dry run, nothing copied)")
        return
    os.makedirs(a.out, exist_ok=True)
    with open(a.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original", "new_name"])
        for src, new in plan:
            dst = os.path.join(a.out, new)
            if os.path.exists(dst):
                sys.exit(f"{dst} already exists; refusing to overwrite")
            (shutil.move if a.move else shutil.copy2)(src, dst)
            w.writerow([src, new])
    print(f"wrote {len(plan)} files to {a.out} and {a.log}")
    if errors:
        print("Photos listed under PROBLEMS were skipped.")


if __name__ == "__main__":
    main()
