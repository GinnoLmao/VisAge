import numpy as np
from PIL import Image
import pytest

from visage.robustness import CONDITIONS, perturb


def test_known_brightness_clipping_and_independent_clean_copy():
    source = Image.new("RGB", (100, 60), (100, 200, 250))
    assert perturb(source, "dim").getpixel((5, 5)) == (65, 130, 162)
    assert perturb(source, "bright").getpixel((5, 5)) == (135, 255, 255)
    clean = perturb(source, "clean")
    clean.putpixel((5, 5), (0, 0, 0))
    assert source.getpixel((5, 5)) == (100, 200, 250)


def test_crop_translation_has_black_fill_and_does_not_wrap():
    source = Image.new("RGB", (100, 60), "white")
    shifted = np.asarray(perturb(source, "crop_shift"))
    assert not shifted[:, :5].any()
    assert (shifted[:, 5:] == 255).all()


def test_blur_reduces_impulse_peak_and_jpeg_and_lowres_preserve_dimensions():
    values = np.zeros((100, 90, 3), dtype=np.uint8)
    values[50, 45] = 255
    source = Image.fromarray(values)
    assert np.asarray(perturb(source, "blur")).max() < 255
    for condition in CONDITIONS:
        first, second = perturb(source, condition), perturb(source, condition)
        assert first.size == source.size and first.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    with pytest.raises(ValueError, match="Unknown"):
        perturb(source, "unregistered")
