"""One shared face crop and persistent, verified age-group predictors.

The service keeps models, never visitor image bytes. Callers serialize access to
the detector and predictors; the HTTP adapter does so with a bounded queue.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from . import CLASS_NAMES
from .faces import FaceConfig, FaceDetector, FaceError
from .image_formats import decode_upload_image as decode_image, converted_preview
from .score_inference import ScoredPretrainedPredictor
from .model_registry import METHODS, REGISTRY, ARTIFACTS, serving_default


DEFAULT_METHOD = serving_default()
ROOT = Path(__file__).resolve().parents[2]
NOTICE = "Estimated age group only; this does not verify actual age or eligibility."


class InferenceService:
    """Load each model once and use one prepared face for every requested method."""

    def __init__(self, artifacts_dir=None, *, device="cpu", detector=None,
                 face_config=None, detector_path=None, predictor_factory=ScoredPretrainedPredictor):
        artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else ARTIFACTS
        self.device = str(device)
        self.default_method = serving_default(artifacts_dir)
        self.config = face_config or getattr(detector, "config", None) or FaceConfig()
        self.detector = detector or FaceDetector(model_path=detector_path, config=self.config)
        self.predictors = {}
        try:
            for method in METHODS:
                self.predictors[method] = predictor_factory(method, Path(artifacts_dir) / method, device=device)
            freezes = {predictor.metadata["manifest_sha256"] for predictor in self.predictors.values()}
            if len(freezes) != 1:
                raise ValueError("All three models must belong to the same frozen dataset.")
            self.manifest_sha256 = freezes.pop()
            if predictor_factory is ScoredPretrainedPredictor:
                import json
                bundle = json.loads((artifacts_dir / "bundle.json").read_text(encoding="utf-8"))
                if set(bundle["methods"]) != set(METHODS):
                    raise ValueError("Incomplete serving bundle.")
                for method, predictor in self.predictors.items():
                    if predictor.artifact_sha256 != bundle["methods"][method]["artifact_sha256"] or predictor.metadata["protocol_sha256"] != bundle["protocol_sha256"]:
                        raise ValueError("Serving bundle does not match the loaded models.")
        except BaseException:
            self.close()
            raise
        self.ready = True

    def close(self):
        self.ready = False
        self.predictors.clear()

    def describe(self):
        return {
            "ready": self.ready,
            "default_method": self.default_method,
            "methods": [{**REGISTRY[method].to_dict(), "device": self.device,
                "seed": self.predictors[method].metadata.get("seed"),
                "selected_epoch": self.predictors[method].metadata.get("epoch"),
                "run_id": self.predictors[method].metadata.get("run_id"),
                "parameters": self.predictors[method].metadata.get("parameters"),
                "feature_dimension": self.predictors[method].metadata.get("pretrained", {}).get("configuration", {}).get("num_features"),
                "artifact_sha256": getattr(self.predictors[method], "artifact_sha256", None)}
                        for method in METHODS],
            "age_groups": list(CLASS_NAMES),
            "manifest_sha256": self.manifest_sha256,
            "face_processing": asdict(self.config),
            "detector": self.detector.info() if hasattr(self.detector, "info") else {"name": type(self.detector).__name__},
            "image_retention": "Images are processed in memory and are not saved or added to training data.",
            "notice": NOTICE,
        }

    def _ensure_ready(self):
        if not self.ready:
            raise RuntimeError("Inference service is not ready.")

    def faces(self, data: bytes):
        self._ensure_ready()
        started = time.perf_counter()
        image = decode_image(data, self.config)
        decoded = time.perf_counter()
        try:
            candidates = self.detector.detect(image)
            if not candidates:
                raise FaceError("NO_FACE", "No face was detected. Try a clearer, front-facing image.")
            preview = converted_preview(image)
            return {
                "image": {"width": image.width, "height": image.height},
                **({"preview": preview} if preview else {}),
                "faces": [candidate.to_dict() for candidate in candidates],
                "selection_required": len(candidates) > 1,
                "timings_ms": {"decode": (decoded - started) * 1000,
                               "detection": (time.perf_counter() - decoded) * 1000,
                               "processing_total": (time.perf_counter() - started) * 1000},
            }
        finally:
            image.close()

    def predict(self, data: bytes, *, method=None, face_index=None):
        self._ensure_ready()
        method = self.default_method if method is None else method
        if method not in (*METHODS, "all"):
            raise FaceError("INVALID_METHOD", "Choose convnextv2_nano, efficientnetv2_s, swinv2_tiny, or all.", status_code=422)
        started = time.perf_counter()
        image = decode_image(data, self.config)
        decoded = time.perf_counter()
        prepared = None
        try:
            candidates = self.detector.detect(image)
            detected = time.perf_counter()
            prepared = self.detector.prepare(image, face_index=face_index, candidates=candidates)
            cropped = time.perf_counter()
            selected_methods = METHODS if method == "all" else (method,)
            # The same PIL image object is supplied to every classifier. Each
            # artifact then applies its recorded resolution/normalization.
            predictions = [self.predictors[name].predict(prepared.crop) for name in selected_methods]
            finished = time.perf_counter()
            return {
                "default_method": self.default_method,
                "manifest_sha256": self.manifest_sha256,
                "image": {"width": image.width, "height": image.height},
                "selected_face": prepared.selected.to_dict(),
                "faces_detected": len(prepared.candidates),
                "face_processing": prepared.metadata,
                "quality": prepared.quality,
                "predictions": predictions,
                "timings_ms": {"decode": (decoded - started) * 1000,
                               "detection": (detected - decoded) * 1000,
                               "alignment_and_quality": (cropped - detected) * 1000,
                               "classifiers": (finished - cropped) * 1000,
                               "processing_total": (finished - started) * 1000},
                "notice": NOTICE,
            }
        finally:
            if prepared is not None:
                prepared.crop.close()
            image.close()
