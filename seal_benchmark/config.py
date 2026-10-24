"""Shared constants and torch-free helpers (imported by every script).

Everything in the "fixed experiment settings" block is part of the experiment
definition. If you change it, you are running a different experiment.
"""
import hashlib
import os

import numpy as np

# ---- fixed experiment settings ------------------------------------------------
IMG_SIZE = 224                           # every photo is resized to IMG_SIZE x IMG_SIZE
CLASS_NAMES = ["intact", "tampered"]     # index 0 / 1; "tampered" is the positive class
CORRUPTION_SEED = 1234                   # independent of the training seed on purpose,
                                         # so every model sees identical corrupted images
SEVERITIES = (1, 2, 3)
DIFFERENT_BOX_PREFIX = "D"               # box IDs starting with "D" = "different boxes" group
CLEAN_LIGHTING = {"daylight", "indoor"}  # any other lighting counts as a robustness photo

# ---- filename convention --------------------------------------------------------
# {box}_{label}_{tamper}_{phone}_{lighting}_{angle}_{occlusion}_{idx}.jpg
# e.g.  I08_tampered_cut_B_daylight_straight_none_01.jpg
#       I08_intact_none_B_lowflash_straight_finger2_03.jpg
# tamper: none/peel/cut/restuck/disguise    occlusion: none/low/finger2/high/sticker/shadow/partialframe
FIELDS = ["box", "label", "tamper", "phone", "lighting", "angle", "occlusion", "idx"]
ALLOWED = {
    "label": {"intact", "tampered"},
    "tamper": {"none", "peel", "cut", "restuck", "disguise"},
    "lighting": {"daylight", "indoor", "lowlight", "lowflash", "glare"},
    "angle": {"straight", "offaxis"},
    "occlusion": {"none", "low", "finger2", "high", "sticker", "shadow", "partialframe"},
}
# Older spellings in existing filenames are still accepted and are converted to the new names
# inside the manifest (the files themselves are not renamed).
TAMPER_ALIASES = {"peeled": "peel", "partial": "restuck", "disguised": "disguise"}
OCCLUSION_ALIASES = {"finger1": "low", "finger3": "high"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def parse_filename(name):
    """Parse a photo filename into a dict of fields. Raises ValueError if invalid."""
    stem, ext = os.path.splitext(os.path.basename(name))
    if ext.lower() not in IMAGE_EXTS:
        raise ValueError(f"unsupported extension '{ext}'")
    parts = stem.split("_")
    if len(parts) != len(FIELDS):
        raise ValueError(f"expected {len(FIELDS)} underscore-separated tokens "
                         f"({'_'.join(FIELDS)}), got {len(parts)}")
    rec = dict(zip(FIELDS, parts))
    rec["tamper"] = TAMPER_ALIASES.get(rec["tamper"], rec["tamper"])
    rec["occlusion"] = OCCLUSION_ALIASES.get(rec["occlusion"], rec["occlusion"])
    box = rec["box"]
    if not (len(box) >= 3 and box[0].isalpha() and box[0].isupper() and box[1:].isdigit()):
        raise ValueError(f"box '{box}' must look like I08 or D01")
    if not (len(rec["phone"]) == 1 and rec["phone"].isalpha() and rec["phone"].isupper()):
        raise ValueError(f"phone '{rec['phone']}' must be a single capital letter (A, B, C, ...)")
    for key, allowed in ALLOWED.items():
        if rec[key] not in allowed:
            raise ValueError(f"{key} '{rec[key]}' not in {sorted(allowed)}")
    if rec["label"] == "intact" and rec["tamper"] != "none":
        raise ValueError("intact photos must use tamper token 'none'")
    if rec["label"] == "tampered" and rec["tamper"] == "none":
        raise ValueError("tampered photos need a tamper type (peeled/cut/partial/disguised)")
    if not rec["idx"].isdigit():
        raise ValueError(f"idx '{rec['idx']}' must be digits")
    return rec


def derive_group_condition(rec):
    group = "different" if rec["box"].startswith(DIFFERENT_BOX_PREFIX) else "identical"
    clean = rec["lighting"] in CLEAN_LIGHTING and rec["occlusion"] == "none"
    return group, ("clean" if clean else "robust")


# ---- helpers --------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(relpath, corruption, severity):
    """Deterministic per-image, per-corruption seed (does not use Python's salted hash())."""
    key = f"{CORRUPTION_SEED}|{relpath}|{corruption}|{severity}".encode()
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def metrics(y_true, y_pred):
    """Binary metrics with 'tampered' (1) as the positive class."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def div(a, b):
        return float(a) / b if b > 0 else float("nan")

    return {
        "n": int(len(y_true)),
        "n_tampered": tp + fn,
        "accuracy": div(tp + tn, len(y_true)),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
        "fnr": div(fn, tp + fn),          # share of tampered photos called intact
    }
