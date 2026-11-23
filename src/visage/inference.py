"""Inference from frozen artifacts on an already selected, aligned face crop.

Use ``visage.service.InferenceService`` for full-image detection and selection.
This crop-level adapter does not verify that its input contains a face or prove
chronological age.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

from PIL import Image, ImageOps
import torch

from . import CLASS_NAMES
from .datasets import make_transform
from .methods.classical import ClassicalAgeClassifier
from .runtime import sha256_file
from .training import load_neural_artifact, logits_to_probabilities


def read_face_crop(path: str | Path) -> Image.Image:
    path = Path(path)
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("Image exceeds the 8 MiB input limit.")
    with Image.open(path) as original:
        if original.width * original.height > 20_000_000:
            raise ValueError("Image exceeds the 20 megapixel decoded limit.")
        if original.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Choose a JPEG, PNG or WebP image.")
        original.load()
        return ImageOps.exif_transpose(original).convert("RGB")


class AgeGroupPredictor:
    def __init__(self, method, artifact_dir, device="cpu"):
        self.method = method
        self.directory = Path(artifact_dir)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        if method == "classical":
            self.metadata = json.loads((self.directory / "selection.json").read_text())
            if sha256_file(self.directory / "svm.npz") != self.metadata["svm_sha256"]:
                raise ValueError("Classical artifact checksum differs from the selected model.")
            self.model = ClassicalAgeClassifier.load(self.directory)
            if self.model.extractor.config.to_dict() != self.metadata["feature_config"]:
                raise ValueError("Feature configuration differs from the selected classical model.")
            if self.model.svm._configuration() != self.metadata["selected_configuration"]:
                raise ValueError("SVM configuration differs from the selected classical model.")
            self.transform = None
        elif method in {"visagenet", "convnext_ordinal"}:
            summary = json.loads((self.directory / "training_summary.json").read_text())
            if sha256_file(self.directory / "best.pt") != summary["artifact_sha256"]:
                raise ValueError("Neural artifact checksum differs from its completed run.")
            self.model, self.metadata = load_neural_artifact(self.directory / "best.pt", self.device)
            if self.metadata["method"] != method or summary["method"] != method:
                raise ValueError("Model method does not match the requested adapter.")
            if self.metadata["manifest_sha256"] != summary["manifest_sha256"]:
                raise ValueError("Neural artifact and training summary have different dataset freezes.")
            transform = self.metadata["transform"]
            self.transform = make_transform(image_size=transform["image_size"], mean=transform["mean"], std=transform["std"])
        else:
            raise ValueError(f"Unknown method {method}")

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        if self.device.type == "cuda" and self.method != "classical":
            torch.cuda.synchronize()
        started = time.perf_counter()
        if self.method == "classical":
            label = self.model.predict_image(image)
        else:
            pixels = self.transform(image).unsqueeze(0).to(self.device)
            values = logits_to_probabilities(self.model(pixels), self.model.head_type)
            if not torch.isfinite(values).all():
                raise RuntimeError("Non-finite model output.")
            label = int(values.argmax(1).item())
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        return {"method": self.method, "class_id": int(label), "age_group": CLASS_NAMES[label],
                "inference_and_preprocessing_ms": (time.perf_counter() - started) * 1000,
                "manifest_sha256": self.metadata["manifest_sha256"],
                "device": "cpu" if self.method == "classical" else str(self.device)}
