"""Torch-dependent helpers: seeding, image loading, datasets, model construction."""
import os

# Must be set before CUDA is initialised; required for deterministic cuBLAS on GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import platform  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from config import IMG_SIZE  # noqa: E402

BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def set_seed(seed):
    """Seed every RNG we use and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


_CACHE = {}


def load_image(path, size=IMG_SIZE):
    """Open a photo, apply its EXIF rotation, convert to RGB, resize to size x size.

    The (squashed) resize is intentional and identical for every split and model.
    Results are cached because the dataset is small.
    """
    key = (path, size)
    if key not in _CACHE:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        _CACHE[key] = img.resize((size, size), BICUBIC)
    return _CACHE[key]


def train_transform(mean, std, size=IMG_SIZE):
    """Light, normal-variation augmentation only (no blur/noise/occlusion/colour jitter)."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def eval_transform(mean, std):
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


class SealDataset(Dataset):
    """Clean photos from a manifest DataFrame -> (tensor, label)."""

    def __init__(self, df, images_dir, transform, size=IMG_SIZE):
        self.df = df.reset_index(drop=True)
        self.images_dir, self.transform, self.size = images_dir, transform, size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_image(os.path.join(self.images_dir, row.relpath), self.size)
        return self.transform(img), int(row.label_idx)


def build_model(name, pretrained=True, drop_rate=0.2):
    """timm model with a fresh 2-class head, plus its normalisation config."""
    import timm
    model = timm.create_model(name, pretrained=pretrained, num_classes=2, drop_rate=drop_rate)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    pc = getattr(model, "pretrained_cfg", None) or {}
    weights = {k: pc.get(k) for k in ("architecture", "tag", "hf_hub_id", "url")}
    return model, data_cfg, weights


def env_info():
    import timm
    import torchvision
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "timm": timm.__version__, "numpy": np.__version__, "pillow": Image.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = None
    return info
