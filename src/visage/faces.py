"""Bounded in-memory image admission and one shared YuNet face crop.

The operational quality limits are conservative engineering defaults, not a
validated face-quality or chronological-age estimator. The eye-level similarity
crop is versioned separately from the pre-aligned UTKFace training images; live
domain accuracy must be measured independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .runtime import sha256_file
from .alignment import TRAINING_TEMPLATE, similarity_transform


ROOT = Path(__file__).resolve().parents[2]
DETECTOR_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
DETECTOR_FILENAME = "face_detection_yunet_2023mar.onnx"
DETECTOR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DETECTOR_BYTES = 232589
DETECTOR_LICENSE_SHA256 = "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958"
DEFAULT_DETECTOR_DIR = ROOT / ".cache" / "face_detector"
CROP_POLICY = "yunet_training_five_point_v3"


class FaceError(ValueError):
    def __init__(self, code: str, message: str, *, details=None, status_code=422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code


@dataclass(frozen=True)
class FaceConfig:
    max_bytes: int = 8 * 1024**2
    max_pixels: int = 20_000_000
    max_detection_side: int = 1280
    max_faces: int = 10
    output_size: int = 224
    min_face_size: int = 64
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    crop_expansion: float = 1.0
    eye_height_fraction: float = (TRAINING_TEMPLATE[0][1] + TRAINING_TEMPLATE[1][1]) / 2
    eye_mid_x_fraction: float = (TRAINING_TEMPLATE[0][0] + TRAINING_TEMPLATE[1][0]) / 2
    eye_distance_fraction: float = math.dist(TRAINING_TEMPLATE[0], TRAINING_TEMPLATE[1])
    max_alignment_rmse_fraction: float = 0.1
    min_visible_fraction: float = 0.6
    max_roll_degrees: float = 40.0
    min_blur_variance: float = 1.0
    min_mean_brightness: float = 15.0
    max_mean_brightness: float = 245.0
    min_dynamic_range: float = 20.0

    def __post_init__(self):
        limits = (self.max_bytes, self.max_pixels, self.max_detection_side,
                  self.max_faces, self.output_size, self.min_face_size)
        if any(type(value) is not int for value in limits):
            raise ValueError("Face size and count limits must be integers.")
        if min(self.max_bytes, self.max_pixels, self.max_detection_side,
               self.max_faces, self.output_size, self.min_face_size) < 1:
            raise ValueError("Face limits must be positive.")
        if self.max_detection_side < 32 or self.output_size < 32:
            raise ValueError("Detection and output sizes must be at least 32 pixels.")
        if not 0 < self.score_threshold <= 1 or not 0 < self.nms_threshold <= 1:
            raise ValueError("Detector thresholds must lie in (0, 1].")
        values = asdict(self).values()
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Face configuration must be finite.")
        if not 1 <= self.crop_expansion <= 2 or not 0.2 <= self.eye_height_fraction <= 0.5:
            raise ValueError("Unsupported face crop geometry.")
        if not 0.35 <= self.eye_mid_x_fraction <= 0.65 or not 0.25 <= self.eye_distance_fraction <= 0.55:
            raise ValueError("Unsupported canonical eye geometry.")
        if not 0 < self.max_alignment_rmse_fraction <= 0.25:
            raise ValueError("Unsupported alignment residual limit.")
        if not 0 < self.min_visible_fraction <= 1 or not 0 < self.max_roll_degrees <= 90:
            raise ValueError("Invalid face visibility or roll limit.")
        if self.min_blur_variance < 0 or self.min_dynamic_range < 0:
            raise ValueError("Quality thresholds must be nonnegative.")
        if not 0 <= self.min_mean_brightness < self.max_mean_brightness <= 255:
            raise ValueError("Brightness limits must lie in [0, 255].")


@dataclass(frozen=True)
class FaceCandidate:
    index: int
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    score: float

    def to_dict(self):
        return {"index": self.index, "bbox": list(self.bbox),
                "landmarks": [list(p) for p in self.landmarks], "score": self.score}


@dataclass
class PreparedFace:
    crop: Image.Image
    selected: FaceCandidate
    candidates: tuple[FaceCandidate, ...]
    quality: dict
    metadata: dict


def decode_image(data: bytes, config: FaceConfig | None = None) -> Image.Image:
    """Fully decode an admitted image into independent EXIF-corrected RGB pixels."""
    config = config or FaceConfig()
    if not isinstance(data, bytes):
        raise TypeError("Image input must be bytes.")
    if not data:
        raise FaceError("EMPTY_IMAGE", "Supply a nonempty image.", status_code=400)
    if len(data) > config.max_bytes:
        raise FaceError("IMAGE_TOO_LARGE", "Image exceeds the 8 MiB upload limit.", status_code=413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as original:
                if original.format not in {"JPEG", "PNG", "WEBP"}:
                    raise FaceError("UNSUPPORTED_IMAGE", "Choose a JPEG, PNG or WebP image.", status_code=415)
                if original.width * original.height > config.max_pixels:
                    raise FaceError("IMAGE_DIMENSIONS_TOO_LARGE", "Image exceeds the 20 megapixel decoded limit.", status_code=413)
                if getattr(original, "n_frames", 1) != 1:
                    raise FaceError("ANIMATED_IMAGE", "Choose a still image rather than an animation.", status_code=415)
                if min(original.size) < 32:
                    raise FaceError("IMAGE_TOO_SMALL", "Image dimensions must be at least 32 pixels.")
                original.load()
                oriented = ImageOps.exif_transpose(original)
                try:
                    # Composite transparency consistently, rather than interpreting hidden RGB.
                    if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                        rgba = oriented.convert("RGBA")
                        try:
                            result = Image.new("RGB", rgba.size, (127, 127, 127))
                            result.paste(rgba, mask=rgba.getchannel("A"))
                            return result
                        finally:
                            rgba.close()
                    return oriented.convert("RGB")
                finally:
                    if oriented is not original:
                        oriented.close()
    except FaceError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise FaceError("IMAGE_DIMENSIONS_TOO_LARGE", "Image exceeds the decoded image safety limit.", status_code=413) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise FaceError("INVALID_IMAGE", "Image is unreadable or corrupt.", status_code=400) from exc


class FaceDetector:
    """One checksum-verified CPU YuNet instance; mutable network calls are locked."""

    def __init__(self, model_path=None, provenance_path=None, config=None):
        self.config = config or FaceConfig()
        model_path = Path(model_path or DEFAULT_DETECTOR_DIR / DETECTOR_FILENAME)
        provenance_path = Path(provenance_path or model_path.parent / "provenance.json")
        if not model_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError("Run scripts/download_face_detector.py before starting face inference.")
        self.provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (sha256_file(model_path) != DETECTOR_SHA256
                or self.provenance.get("sha256") != DETECTOR_SHA256
                or self.provenance.get("revision") != DETECTOR_REVISION
                or model_path.stat().st_size != DETECTOR_BYTES):
            raise ValueError("Face detector differs from the pinned official YuNet artifact.")
        license_path = model_path.parent / "LICENSE"
        if (not license_path.is_file()
                or sha256_file(license_path) != DETECTOR_LICENSE_SHA256
                or self.provenance.get("license_sha256") != DETECTOR_LICENSE_SHA256):
            raise ValueError("Face detector license evidence is missing or changed.")
        self._lock = threading.Lock()
        self._network = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                                self.config.score_threshold,
                                                self.config.nms_threshold, 5000,
                                                cv2.dnn.DNN_BACKEND_OPENCV,
                                                cv2.dnn.DNN_TARGET_CPU)

    def info(self):
        return {"name": "YuNet 2023mar", "revision": DETECTOR_REVISION,
                "sha256": DETECTOR_SHA256, "license": "MIT", "device": "cpu",
                "crop_policy": CROP_POLICY, "configuration": asdict(self.config)}

    def detect(self, image: Image.Image) -> list[FaceCandidate]:
        if image.mode != "RGB" or image.width * image.height > self.config.max_pixels:
            raise ValueError("Face detection requires admitted RGB pixels.")
        width, height = image.size
        if min(width, height) < 32:
            raise FaceError("IMAGE_TOO_SMALL", "Image dimensions must be at least 32 pixels.")
        ratio = min(1.0, self.config.max_detection_side / max(width, height))
        dw, dh = max(32, round(width * ratio)), max(32, round(height * ratio))
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if (dw, dh) != image.size:
            bgr = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        with self._lock:
            self._network.setInputSize((dw, dh))
            _, raw = self._network.detect(bgr)
            if raw is not None:
                raw = np.array(raw, copy=True)
        if raw is None:
            return []
        sx, sy = width / dw, height / dh
        records = []
        for row in np.asarray(raw):
            if row.shape != (15,) or not np.isfinite(row).all() or row[14] < self.config.score_threshold:
                continue
            x, y, w, h = map(float, row[:4])
            if w <= 0 or h <= 0:
                continue
            x0, y0 = max(0.0, x * sx), max(0.0, y * sy)
            x1, y1 = min(float(width), (x + w) * sx), min(float(height), (y + h) * sy)
            if x1 <= x0 or y1 <= y0:
                continue
            points = tuple((float(row[j]) * sx, float(row[j + 1]) * sy) for j in range(4, 14, 2))
            records.append(((x0, y0, x1 - x0, y1 - y0), points, float(row[14])))
        # Stable left-to-right, then top-to-bottom indexes refer to this exact image.
        records.sort(key=lambda r: (r[0][0], r[0][1], -r[2], r[0][2], r[0][3]))
        if len(records) > self.config.max_faces:
            raise FaceError("TOO_MANY_FACES", "Use an image containing fewer faces.",
                            details={"detected_count": len(records), "maximum": self.config.max_faces})
        return [FaceCandidate(i, *record) for i, record in enumerate(records)]

    def prepare(self, image: Image.Image, face_index=None, candidates=None) -> PreparedFace:
        candidates = tuple(self.detect(image) if candidates is None else candidates)
        details = {"faces": [face.to_dict() for face in candidates]}
        if not candidates:
            raise FaceError("NO_FACE", "No usable face was detected. Face the camera in good light.")
        if face_index is None:
            if len(candidates) != 1:
                raise FaceError("MULTIPLE_FACES", "Select a face explicitly before estimating its age group.", details=details)
            face_index = candidates[0].index
        if isinstance(face_index, bool) or not isinstance(face_index, int):
            raise FaceError("INVALID_FACE_INDEX", "Face index must be an integer.", details=details, status_code=400)
        selected = next((f for f in candidates if f.index == face_index), None)
        if selected is None:
            raise FaceError("INVALID_FACE_INDEX", "Selected face index is absent from this image.", details=details, status_code=400)
        x, y, width, height = selected.bbox
        if min(width, height) < self.config.min_face_size:
            raise FaceError("FACE_TOO_SMALL", "Move closer or use a larger face image.",
                            details={**details, "minimum_face_pixels": self.config.min_face_size})
        points = np.asarray(selected.landmarks, dtype=np.float64)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise FaceError("UNUSABLE_FACE", "Face landmarks are unusable. Try a clear frontal image.")
        if ((points[:, 0] < 0).any() or (points[:, 0] >= image.width).any()
                or (points[:, 1] < 0).any() or (points[:, 1] >= image.height).any()):
            raise FaceError("UNUSABLE_FACE", "Show the whole face, including both eyes and mouth.")
        border_margin = 0.02 * min(width, height)
        if ((points[:, 0] < border_margin).any() or (points[:, 0] > image.width - border_margin).any()
                or (points[:, 1] < border_margin).any() or (points[:, 1] > image.height - border_margin).any()):
            raise FaceError("UNUSABLE_FACE", "Keep visible face landmarks away from the image edge.",
                            details={"reasons": ["face_cut_off"], **details})
        if ((points[:, 0] < x - width * 0.25).any() or (points[:, 0] > x + width * 1.25).any()
                or (points[:, 1] < y - height * 0.25).any() or (points[:, 1] > y + height * 1.25).any()):
            raise FaceError("UNUSABLE_FACE", "Face landmarks lie outside a plausible face region.")
        eyes = points[:2][np.argsort(points[:2, 0])]
        delta = eyes[1] - eyes[0]
        eye_distance = float(np.linalg.norm(delta))
        roll = math.degrees(math.atan2(delta[1], delta[0]))
        if eye_distance < 0.12 * max(width, height) or abs(roll) > self.config.max_roll_degrees:
            raise FaceError("UNUSABLE_FACE", "Use an upright, clearly visible face with both eyes visible.")
        size = self.config.output_size
        # Five-point geometry minimizes original training-crop reconstruction
        # error better than an eye-only template; no age labels are used.
        points[:2] = eyes
        points[3:] = points[3:][np.argsort(points[3:, 0])]
        target = np.asarray(TRAINING_TEMPLATE, dtype=float)
        template_midpoint = target[:2].mean(0)
        template_distance = np.linalg.norm(target[1] - target[0])
        target = ((target - template_midpoint) * self.config.eye_distance_fraction / template_distance / self.config.crop_expansion
                  + [self.config.eye_mid_x_fraction, self.config.eye_height_fraction]) * size
        affine = similarity_transform(points, target)
        mapped_landmarks = np.c_[points, np.ones(5)] @ affine.T
        alignment_rmse = float(np.sqrt(np.mean(np.sum((mapped_landmarks - target) ** 2, axis=1))))
        if alignment_rmse > size * self.config.max_alignment_rmse_fraction:
            raise FaceError("UNUSABLE_FACE", "Use a clearly visible, front-facing face.",
                            details={"reasons": ["landmark_fit"], "alignment_rmse_pixels": alignment_rmse, **details})
        if (mapped_landmarks < 0).any() or (mapped_landmarks >= size).any():
            raise FaceError("UNUSABLE_FACE", "The aligned crop cannot contain the visible eyes, nose and mouth. Try a frontal photo.",
                            details={"reasons": ["face_cut_off"], **details})
        pixels = np.asarray(image)
        aligned = cv2.warpAffine(pixels, affine, (size, size), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(np.ones(pixels.shape[:2], dtype=np.uint8), affine, (size, size),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        context_visible_fraction = float(mask.mean())
        # Expanded background is optional (UTKFace inputs are already tightly
        # cropped). Reject missing pixels in the central face, not absent context.
        inner_mask = mask[size // 5:4 * size // 5, size // 5:4 * size // 5]
        visible_fraction = float(inner_mask.mean())
        gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
        # Measure the central face region so reflected context cannot rescue poor quality.
        inner = gray[size // 5:4 * size // 5, size // 5:4 * size // 5]
        brightness = float(inner.mean())
        dynamic_range = float(np.percentile(inner, 95) - np.percentile(inner, 5))
        blur = float(cv2.Laplacian(inner, cv2.CV_64F).var())
        quality = {"visible_fraction": visible_fraction,
                   "alignment_rmse_pixels": alignment_rmse,
                   "visible_fraction_scope": "central 60 percent of aligned crop",
                   "context_visible_fraction": context_visible_fraction,
                   "mean_brightness": brightness,
                   "dynamic_range_p95_p5": dynamic_range, "laplacian_variance": blur,
                   "roll_degrees": roll, "engineering_thresholds_only": True}
        reasons = []
        if visible_fraction < self.config.min_visible_fraction:
            reasons.append("face_cut_off")
        if not self.config.min_mean_brightness <= brightness <= self.config.max_mean_brightness:
            reasons.append("lighting")
        if dynamic_range < self.config.min_dynamic_range:
            reasons.append("low_contrast")
        if blur < self.config.min_blur_variance:
            reasons.append("blur")
        if reasons:
            raise FaceError("UNUSABLE_FACE", "Use a sharper, well-lit image showing the whole face.",
                            details={"reasons": reasons, "quality": quality, **details})
        crop = Image.fromarray(aligned)
        return PreparedFace(crop, selected, candidates, quality,
                            {"policy": CROP_POLICY, "output_size": size,
                             "geometry_fit": "Median five-point geometry from498 of500 fixed training crops; selected by training-pixel reconstruction only; no age labels or validation/test images used for fitting.",
                             "original_size": list(image.size), "affine_original_to_crop": affine.tolist(),
                             "padding": "reflect101 only outside original image; visibility measured separately",
                             "selection": "explicit_index" if len(candidates) > 1 else "single_detected_face",
                             "pixel_sha256": hashlib.sha256(aligned.tobytes()).hexdigest()})
