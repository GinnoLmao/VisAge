"""CPU reference checks for the device-side training augmentation policy."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from visage.augmentation import BATCH_AUGMENTATION_POLICY, prepare_training_batch


def test_unaugmented_normalization_matches_numpy_and_does_not_consume_rng() -> None:
    pixels = np.array([[[[0, 255], [128, 64]], [[255, 0], [32, 128]], [[64, 32], [255, 0]]]], dtype=np.uint8)
    images = torch.from_numpy(pixels)
    original = images.clone()
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    torch.manual_seed(83)
    rng_before = torch.get_rng_state()
    result = prepare_training_batch(images, mean, std, augment=False)
    expected = (pixels.astype(np.float32) / 255 - np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)) / np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(result.numpy(), expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
    torch.testing.assert_close(images, original, rtol=0, atol=0)
    assert result.dtype == torch.float32 and result.device == images.device
    assert not result.requires_grad


def test_augmented_batch_is_seed_reproducible_and_draws_per_image() -> None:
    images = torch.full((8, 3, 32, 32), 128, dtype=torch.uint8)
    torch.manual_seed(42)
    first = prepare_training_batch(images, (0, 0, 0), (1, 1, 1))
    torch.manual_seed(42)
    repeated = prepare_training_batch(images, (0, 0, 0), (1, 1, 1))
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert first[:, :, 16, 16].std(dim=0).min() > 0.01
    torch.manual_seed(43)
    different = prepare_training_batch(images, (0, 0, 0), (1, 1, 1))
    assert not torch.equal(first, different)
    assert torch.isfinite(first).all() and first.min() >= 0 and first.max() <= 1
    assert not first.requires_grad


def test_flip_is_independent_per_image_with_identity_color_and_rotation(monkeypatch) -> None:
    images = torch.arange(3 * 4 * 7, dtype=torch.uint8).reshape(1, 3, 4, 7).repeat(2, 1, 1, 1)
    draws = torch.full((2, 5), 0.5)
    draws[0, 0], draws[1, 0] = 0.0, 1.0
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: draws)
    result = prepare_training_batch(images, (0, 0, 0), (1, 1, 1))
    expected = images.float().div(255)
    expected[0] = expected[0].flip(-1)
    torch.testing.assert_close(result, expected, rtol=0, atol=2e-7)


def _numpy_rotate(image: np.ndarray, angle: float) -> np.ndarray:
    channels, height, width = image.shape
    result = np.zeros_like(image)
    sine, cosine = math.sin(angle), math.cos(angle)
    for y in range(height):
        for x in range(width):
            # Inverse coordinate map at pixel centers, equivalent to the
            # normalized affine grid without relying on torch's sampler.
            centered_x, centered_y = x - (width - 1) / 2, y - (height - 1) / 2
            sample_x = cosine * centered_x - sine * centered_y + (width - 1) / 2
            sample_y = sine * centered_x + cosine * centered_y + (height - 1) / 2
            x0, y0 = math.floor(sample_x), math.floor(sample_y)
            for yy in (y0, y0 + 1):
                for xx in (x0, x0 + 1):
                    if 0 <= yy < height and 0 <= xx < width:
                        weight = (1 - abs(sample_y - yy)) * (1 - abs(sample_x - xx))
                        result[:, y, x] += image[:, yy, xx] * weight
    return result


def test_non_square_rotation_matches_independent_bilinear_reference(monkeypatch) -> None:
    images = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(1, 3, 5, 7)
    draws = torch.full((1, 5), 0.5)
    draws[0, 0], draws[0, 1] = 1.0, 1.0  # No flip; rotate eight degrees.
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: draws)
    result = prepare_training_batch(images, (0, 0, 0), (1, 1, 1))
    expected = _numpy_rotate(images[0].numpy().astype(np.float32) / 255, math.radians(8))
    np.testing.assert_allclose(result[0].numpy(), expected, rtol=0, atol=3e-7)


@pytest.mark.parametrize(
    ("images", "mean", "std", "exception"),
    [
        (torch.zeros(1, 3, 4, 4), (0, 0, 0), (1, 1, 1), TypeError),
        (torch.zeros(1, 1, 4, 4, dtype=torch.uint8), (0, 0, 0), (1, 1, 1), ValueError),
        (torch.zeros(0, 3, 4, 4, dtype=torch.uint8), (0, 0, 0), (1, 1, 1), ValueError),
        (torch.zeros(1, 3, 4, 4, dtype=torch.uint8), (0, 0), (1, 1, 1), ValueError),
        (torch.zeros(1, 3, 4, 4, dtype=torch.uint8), (float("nan"), 0, 0), (1, 1, 1), ValueError),
        (torch.zeros(1, 3, 4, 4, dtype=torch.uint8), (-1, 0, 0), (1, 1, 1), ValueError),
        (torch.zeros(1, 3, 4, 4, dtype=torch.uint8), (0, 0, 0), (1, 0, 1), ValueError),
    ],
)
def test_invalid_inputs_fail_early(images, mean, std, exception) -> None:
    with pytest.raises(exception):
        prepare_training_batch(images, mean, std)


def test_named_policy_records_the_experimental_difference() -> None:
    assert BATCH_AUGMENTATION_POLICY["color_order"] == ["brightness", "contrast", "saturation"]
    assert "bilinear" in BATCH_AUGMENTATION_POLICY["rotation_interpolation"]
