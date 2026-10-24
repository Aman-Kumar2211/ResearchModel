"""Deterministic synthetic corruptions (applied to test images only).

All corruptions operate on a 224x224 RGB PIL image and take a seed, so the exact
same corrupted pixels are produced on every run, for every model.

Run `python corruptions.py --demo some_photo.jpg` to write a contact sheet and
eyeball whether the corruptions look like plausible phone-photo problems.
"""
import argparse
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import uniform_filter1d

from config import IMG_SIZE, SEVERITIES, seed_for  # noqa: F401  (seed_for re-exported)


def _to_img(arr):
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))


def gaussian_noise(img, sev, rng):
    sigma = (0.04, 0.08, 0.14)[sev - 1]                 # fraction of full range
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = x + rng.normal(0.0, sigma, x.shape).astype(np.float32)
    return _to_img(x * 255.0)


def blur(img, sev, rng):                                # defocus-like (Gaussian) blur
    radius = (1.5, 3.0, 5.0)[sev - 1]
    return img.filter(ImageFilter.GaussianBlur(radius))


def motion_blur(img, sev, rng):                         # horizontal motion blur
    length = (7, 15, 25)[sev - 1]
    arr = np.asarray(img, dtype=np.float32)
    return _to_img(uniform_filter1d(arr, size=length, axis=1, mode="reflect"))


def brightness_down(img, sev, rng):                     # under-exposed / dim room
    return ImageEnhance.Brightness(img).enhance((0.6, 0.4, 0.2)[sev - 1])


def contrast_down(img, sev, rng):                       # washed-out / hazy
    return ImageEnhance.Contrast(img).enhance((0.6, 0.4, 0.2)[sev - 1])


def jpeg(img, sev, rng):                                # heavy upload compression
    quality = (50, 25, 10)[sev - 1]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def occlusion_patch(img, sev, rng):                     # gray rectangle hiding part of the photo
    frac = (0.15, 0.30, 0.50)[sev - 1]                  # fraction of image area hidden
    w, h = img.size
    area = frac * w * h
    aspect = rng.uniform(0.6, 1.6)                      # width / height
    ph = min(h, max(1, int(round(np.sqrt(area / aspect)))))
    pw = min(w, max(1, int(round(area / ph))))
    x0 = int(rng.randint(0, w - pw + 1))
    y0 = int(rng.randint(0, h - ph + 1))
    arr = np.array(img)
    arr[y0:y0 + ph, x0:x0 + pw] = 128
    return Image.fromarray(arr)


CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "blur": blur,
    "motion_blur": motion_blur,
    "brightness_down": brightness_down,
    "contrast_down": contrast_down,
    "jpeg": jpeg,
    "occlusion_patch": occlusion_patch,
}


def apply_corruption(img, name, severity, seed):
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption '{name}'")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    rng = np.random.RandomState(seed)                   # frozen stream across numpy versions
    return CORRUPTIONS[name](img.convert("RGB"), severity, rng)


def make_contact_sheet(path, out="corruption_demo.png"):
    from PIL import ImageOps
    base = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    names = list(CORRUPTIONS)
    sheet = Image.new("RGB", (IMG_SIZE * (1 + len(SEVERITIES)), IMG_SIZE * len(names)), "white")
    for r, name in enumerate(names):
        sheet.paste(base, (0, r * IMG_SIZE))
        for c, sev in enumerate(SEVERITIES, start=1):
            out_img = apply_corruption(base, name, sev, seed_for("demo", name, sev))
            sheet.paste(out_img, (c * IMG_SIZE, r * IMG_SIZE))
    sheet.save(out)
    print(f"wrote {out}  (rows: {', '.join(names)}; columns: original, severity 1, 2, 3)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="path to one photo")
    ap.add_argument("--out", default="corruption_demo.png")
    a = ap.parse_args()
    make_contact_sheet(a.demo, a.out)
