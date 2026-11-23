"""Device-side batch preparation for aligned RGB face crops.

The input loader performs only whole-face bicubic resizing and converts to
uint8 CHW tensors. This module applies independently sampled transformations
per image. Its fixed color order and bilinear tensor rotation intentionally
differ from torchvision's PIL ColorJitter/rotation policy and must be recorded
in the run configuration. No operation learns from validation/test images.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor
from torch.nn import functional as F


BATCH_AUGMENTATION_POLICY = {
    "name": "visage_tensor_batch_v1",
    "input": "uint8 RGB; whole aligned face resized bicubic by input loader",
    "sampling": "independent per image; torch global RNG on input device",
    "horizontal_flip_probability": 0.5,
    "rotation_degrees": [-8.0, 8.0],
    "rotation_interpolation": "bilinear grid_sample; zero padding; align_corners=False",
    "color_order": ["brightness", "contrast", "saturation"],
    "brightness_factor": [0.85, 1.15],
    "contrast_factor": [0.85, 1.15],
    "saturation_factor": [0.9, 1.1],
    "color_range": "clamp to [0,1] after each operation",
    "normalization": "float32 uint8 / 255; subtract configured RGB mean; divide RGB std",
}


def _normalization_values(values: Sequence[float] | Tensor, images: Tensor, *, std: bool) -> Tensor:
    name = "std" if std else "mean"
    result = torch.as_tensor(values, dtype=torch.float32, device=images.device)
    if result.shape != (3,) or not torch.all(torch.isfinite(result)):
        raise ValueError(f"{name} must contain three finite RGB values.")
    if std:
        if torch.any(result <= 0):
            raise ValueError("std values must be positive.")
    elif torch.any((result < 0) | (result > 1)):
        raise ValueError("mean values must be within [0, 1].")
    return result.view(1, 3, 1, 1)


def _rgb_grayscale(images: Tensor) -> Tensor:
    # Match torchvision's tensor RGB luminance coefficients.
    return images[:, 0:1] * 0.2989 + images[:, 1:2] * 0.587 + images[:, 2:3] * 0.114


@torch.no_grad()
def prepare_training_batch(
    images: Tensor,
    mean: Sequence[float] | Tensor,
    std: Sequence[float] | Tensor,
    *,
    augment: bool = True,
) -> Tensor:
    """Convert uint8 ``[N,3,H,W]`` images to normalized float32 on that device.

    ``augment=False`` performs only deterministic ToTensor-style conversion
    and normalization, without consuming random numbers. The caller should
    resize the full aligned face identically for all model methods before
    batching. With augmentation enabled, five fresh uniform values per image
    control flip, rotation, brightness, contrast, and saturation, in that order.
    """
    if not isinstance(images, Tensor):
        raise TypeError("images must be a torch Tensor.")
    if images.dtype != torch.uint8:
        raise TypeError("images must use torch.uint8 RGB values.")
    if images.ndim != 4 or images.shape[1] != 3 or any(size <= 0 for size in images.shape):
        raise ValueError("images must have nonempty shape [batch, 3, height, width].")
    if not isinstance(augment, bool):
        raise TypeError("augment must be a boolean.")
    mean_tensor = _normalization_values(mean, images, std=False)
    std_tensor = _normalization_values(std, images, std=True)
    result = images.to(dtype=torch.float32).div_(255.0)
    if augment:
        count, _, height, width = result.shape
        draws = torch.rand((count, 5), dtype=torch.float32, device=images.device)
        flip = (draws[:, 0] < 0.5).view(count, 1, 1, 1)
        result = torch.where(flip, result.flip(-1), result)

        angles = (draws[:, 1] * 16.0 - 8.0) * (math.pi / 180.0)
        cosine, sine = angles.cos(), angles.sin()
        theta = result.new_zeros((count, 2, 3))
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine * (height / width)
        theta[:, 1, 0] = sine * (width / height)
        theta[:, 1, 1] = cosine
        grid = F.affine_grid(theta, result.shape, align_corners=False)
        result = F.grid_sample(
            result, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        brightness = (0.85 + draws[:, 2] * 0.3).view(count, 1, 1, 1)
        result = (result * brightness).clamp_(0.0, 1.0)
        contrast = (0.85 + draws[:, 3] * 0.3).view(count, 1, 1, 1)
        luminance_mean = _rgb_grayscale(result).mean(dim=(2, 3), keepdim=True)
        result = ((result - luminance_mean) * contrast + luminance_mean).clamp_(0.0, 1.0)
        saturation = (0.9 + draws[:, 4] * 0.2).view(count, 1, 1, 1)
        grayscale = _rgb_grayscale(result)
        result = ((result - grayscale) * saturation + grayscale).clamp_(0.0, 1.0)
    return result.sub_(mean_tensor).div_(std_tensor)
