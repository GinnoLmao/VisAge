"""Fixed, label-independent perturbations for aligned-crop validation studies."""
from __future__ import annotations

from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter

CONDITIONS = {
    "clean": "Unchanged decoded aligned crop",
    "dim": "Multiply RGB brightness by 0.65",
    "bright": "Multiply RGB brightness by 1.35 with clipping",
    "blur": "Pillow Gaussian blur radius 1.5 pixels at source resolution",
    "jpeg": "JPEG encode/decode quality 35, chroma subsampling 2",
    "low_resolution": "Bicubic resize to 56x56 then back to original dimensions",
    "crop_shift": "Translate content right by round(0.05 * width), black fill; no wraparound",
}


def perturb(image: Image.Image, condition: str) -> Image.Image:
    image = image.convert("RGB")
    if condition == "clean":
        return image.copy()
    if condition in {"dim", "bright"}:
        return ImageEnhance.Brightness(image).enhance(0.65 if condition == "dim" else 1.35)
    if condition == "blur":
        return image.filter(ImageFilter.GaussianBlur(1.5))
    if condition == "jpeg":
        with BytesIO() as stream:
            image.save(stream, format="JPEG", quality=35, subsampling=2)
            stream.seek(0)
            with Image.open(stream) as decoded:
                return decoded.convert("RGB")
    if condition == "low_resolution":
        return image.resize((56, 56), Image.Resampling.BICUBIC).resize(image.size, Image.Resampling.BICUBIC)
    if condition == "crop_shift":
        result = Image.new("RGB", image.size)
        result.paste(image, (max(1, round(0.05 * image.width)), 0))
        return result
    raise ValueError(f"Unknown robustness condition: {condition}")
