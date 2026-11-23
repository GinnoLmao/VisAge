"""A single manifest drives both neural models, with recorded transforms."""

from __future__ import annotations
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_transform(*, training=False, image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    # Resize the entire aligned face for every method; avoid differing crop geometry.
    operations = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)]
    if training:
        operations += [T.RandomHorizontalFlip(), T.RandomRotation(8, interpolation=T.InterpolationMode.BILINEAR),
                       T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)]
    operations += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(operations)


class ManifestDataset(Dataset):
    def __init__(self, rows, raw_dir, transform=None):
        self.rows = list(rows)
        self.raw_dir = Path(raw_dir).resolve()
        self.transform = transform or make_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = (self.raw_dir / row["relative_path"]).resolve()
        if not path.is_relative_to(self.raw_dir):
            raise ValueError("Manifest image path escapes raw directory.")
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixels = self.transform(image)
        return pixels, int(row["label"]), index


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
