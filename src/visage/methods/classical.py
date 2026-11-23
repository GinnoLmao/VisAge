"""Classical age-group classifier, using only NumPy/Pillow.

The learned task is three-class image classification, not chronological-age
estimation. Shared face detection/cropping belongs to the calling application.
HOG, rotation-invariant uniform LBP, scaling and the linear SVM optimizer are
implemented here; no feature extractor or estimator from sklearn/OpenCV is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


CLASS_NAMES = ("Under 18", "18–59", "60+")
_FORMAT_VERSION = 1


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class FeatureConfig:
    """Serializable, fixed preprocessing; ``image_size`` is a square side in px.

    Default dimension: 8,100 HOG + 480 LBP = 8,580 float32 values. A 20,000
    image matrix therefore needs about 686 MB, before optional disk mapping.
    """

    image_size: int = 128
    include_hog: bool = True
    include_lbp: bool = True
    hog_cell_size: int = 8
    hog_block_size: int = 2
    hog_block_stride: int = 1
    hog_bins: int = 9
    hog_clip: float = 0.2
    lbp_radii: tuple[float, ...] = (1.0, 2.0, 3.0)
    lbp_neighbors: int = 8
    lbp_grid: int = 4
    hog_weight: float = 1.0
    lbp_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "lbp_radii", tuple(float(r) for r in self.lbp_radii))
        for name in ("image_size", "hog_cell_size", "hog_block_size", "hog_block_stride", "hog_bins", "lbp_grid"):
            _positive_integer(name, getattr(self, name))
        _positive_integer("lbp_neighbors", self.lbp_neighbors, 3)
        if self.lbp_neighbors > 32:
            raise ValueError("lbp_neighbors must be <= 32.")
        if not isinstance(self.include_hog, bool) or not isinstance(self.include_lbp, bool):
            raise ValueError("Feature inclusion flags must be booleans.")
        if not self.include_hog and not self.include_lbp:
            raise ValueError("At least one descriptor must be enabled.")
        if self.include_hog:
            if self.image_size % self.hog_cell_size:
                raise ValueError("image_size must be divisible by hog_cell_size.")
            if self.image_size // self.hog_cell_size < self.hog_block_size:
                raise ValueError("The image must contain at least one HOG block.")
        if not np.isfinite(self.hog_clip) or self.hog_clip <= 0:
            raise ValueError("hog_clip must be positive and finite.")
        if self.include_lbp:
            if not self.lbp_radii or any(not np.isfinite(r) or r <= 0 for r in self.lbp_radii):
                raise ValueError("LBP radii must be positive, finite and nonempty.")
            if self.image_size - 2 * math.ceil(max(self.lbp_radii)) < self.lbp_grid:
                raise ValueError("The LBP interior must contain every spatial grid cell.")
        for name in ("hog_weight", "lbp_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")

    @property
    def feature_dimension(self) -> int:
        dimension = 0
        if self.include_hog:
            cells = self.image_size // self.hog_cell_size
            blocks = (cells - self.hog_block_size) // self.hog_block_stride + 1
            dimension += blocks**2 * self.hog_block_size**2 * self.hog_bins
        if self.include_lbp:
            dimension += len(self.lbp_radii) * self.lbp_grid**2 * (self.lbp_neighbors + 2)
        return dimension

    def to_dict(self) -> dict:
        result = asdict(self)
        result["lbp_radii"] = list(self.lbp_radii)
        return result

    @classmethod
    def from_dict(cls, values: Mapping) -> FeatureConfig:
        try:
            return cls(**dict(values))
        except (TypeError, OverflowError) as error:
            raise ValueError("Invalid feature configuration.") from error


def _gray_array(gray: np.ndarray) -> np.ndarray:
    result = np.asarray(gray, dtype=np.float32)
    if result.ndim != 2 or min(result.shape, default=0) < 2 or not np.isfinite(result).all():
        raise ValueError("Expected a finite, two-dimensional grayscale image of at least 2 by 2.")
    return result


def hog_descriptor(
    gray: np.ndarray,
    *,
    cell_size: int = 8,
    block_size: int = 2,
    block_stride: int = 1,
    bins: int = 9,
    clip: float = 0.2,
) -> np.ndarray:
    """Unsigned HOG: central gradients, trilinear votes and L2-Hys blocks.

    Border pixels vote only into cells inside the image. Partial trailing cells
    are discarded. Orientation bins cover [0, 180) and wrap circularly.
    """
    gray = _gray_array(gray)
    for name, value in (("cell_size", cell_size), ("block_size", block_size), ("block_stride", block_stride), ("bins", bins)):
        _positive_integer(name, value)
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be positive and finite.")
    rows, cols = gray.shape[0] // cell_size, gray.shape[1] // cell_size
    if min(rows, cols) < block_size:
        raise ValueError("The grayscale image must contain at least one HOG block.")
    gray = gray[: rows * cell_size, : cols * cell_size]
    gx, gy = np.empty_like(gray), np.empty_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gx[:, 0], gx[:, -1] = gray[:, 1] - gray[:, 0], gray[:, -1] - gray[:, -2]
    gy[1:-1] = (gray[2:] - gray[:-2]) * 0.5
    gy[0], gy[-1] = gray[1] - gray[0], gray[-1] - gray[-2]
    magnitude = np.hypot(gx, gy).ravel()
    orientation = (np.arctan2(gy, gx).ravel() % np.pi) * (bins / np.pi)
    lower_bin = np.floor(orientation).astype(np.intp) % bins
    fraction_bin = orientation - np.floor(orientation)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    cell_y, cell_x = (yy.ravel() + 0.5) / cell_size - 0.5, (xx.ravel() + 0.5) / cell_size - 0.5
    lower_y, lower_x = np.floor(cell_y).astype(np.intp), np.floor(cell_x).astype(np.intp)
    fraction_y, fraction_x = cell_y - lower_y, cell_x - lower_x
    histogram = np.zeros(rows * cols * bins, dtype=np.float64)
    for dy in (0, 1):
        cy = lower_y + dy
        wy = fraction_y if dy else 1 - fraction_y
        for dx in (0, 1):
            cx = lower_x + dx
            wx = fraction_x if dx else 1 - fraction_x
            valid = (cy >= 0) & (cy < rows) & (cx >= 0) & (cx < cols)
            cell_index = (cy[valid] * cols + cx[valid]) * bins
            spatial_vote = magnitude[valid] * wy[valid] * wx[valid]
            for db in (0, 1):
                orientation_weight = fraction_bin[valid] if db else 1 - fraction_bin[valid]
                index = cell_index + (lower_bin[valid] + db) % bins
                histogram += np.bincount(index, weights=spatial_vote * orientation_weight, minlength=histogram.size)
    histogram = histogram.reshape(rows, cols, bins)
    blocks = []
    for y in range(0, rows - block_size + 1, block_stride):
        for x in range(0, cols - block_size + 1, block_stride):
            block = histogram[y : y + block_size, x : x + block_size].ravel().copy()
            block /= np.sqrt(block @ block + 1e-12)
            np.minimum(block, clip, out=block)
            block /= np.sqrt(block @ block + 1e-12)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def uniform_lbp_codes(gray: np.ndarray, radius: float = 1.0, neighbors: int = 8) -> np.ndarray:
    """Compute rotation-invariant uniform LBP (riu2) without border padding.

    Circular neighbors use bilinear sampling. Codes with <=2 cyclic bit
    transitions map to their count of ones (0..P); all others map to P+1.
    Thus P+2 bins are needed, independent of rotation.
    """
    gray = _gray_array(gray)
    _positive_integer("neighbors", neighbors, 3)
    if neighbors > 32 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Expected 3..32 neighbors and a positive finite radius.")
    margin = math.ceil(radius)
    height, width = gray.shape
    if min(height, width) <= 2 * margin:
        raise ValueError("Image is too small for this LBP radius.")
    center = gray[margin : height - margin, margin : width - margin]
    y = np.arange(margin, height - margin, dtype=np.float64)
    x = np.arange(margin, width - margin, dtype=np.float64)
    bits = np.empty((neighbors, *center.shape), dtype=bool)
    for bit in range(neighbors):
        angle = 2 * np.pi * bit / neighbors
        sample_y, sample_x = y - radius * np.sin(angle), x + radius * np.cos(angle)
        # Clipping also removes floating-point excursions at cardinal angles.
        sample_y, sample_x = np.clip(sample_y, 0, height - 1), np.clip(sample_x, 0, width - 1)
        y0, x0 = np.floor(sample_y).astype(np.intp), np.floor(sample_x).astype(np.intp)
        y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
        fy, fx = (sample_y - y0)[:, None], (sample_x - x0)[None, :]
        sampled = (
            gray[y0[:, None], x0[None, :]] * (1 - fy) * (1 - fx)
            + gray[y0[:, None], x1[None, :]] * (1 - fy) * fx
            + gray[y1[:, None], x0[None, :]] * fy * (1 - fx)
            + gray[y1[:, None], x1[None, :]] * fy * fx
        )
        # A tiny tolerance makes equal-valued regions invariant to interpolation
        # roundoff; it is relative to float32 precision, not a texture threshold.
        tolerance = 8 * np.finfo(np.float32).eps * np.maximum(1, np.abs(center))
        bits[bit] = sampled >= center - tolerance
    transitions = np.count_nonzero(bits != np.roll(bits, 1, axis=0), axis=0)
    ones = bits.sum(axis=0)
    return np.where(transitions <= 2, ones, neighbors + 1).astype(np.uint8)


def _lbp_histogram(gray: np.ndarray, config: FeatureConfig) -> np.ndarray:
    histograms = []
    bins = config.lbp_neighbors + 2
    for radius in config.lbp_radii:
        codes = uniform_lbp_codes(gray, radius, config.lbp_neighbors)
        y_edges = np.linspace(0, codes.shape[0], config.lbp_grid + 1, dtype=int)
        x_edges = np.linspace(0, codes.shape[1], config.lbp_grid + 1, dtype=int)
        for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
            for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
                histogram = np.bincount(codes[y0:y1, x0:x1].ravel(), minlength=bins).astype(np.float32)
                histogram /= histogram.sum()
                histograms.append(np.sqrt(histogram))
    return np.concatenate(histograms)


class ClassicalFeatureExtractor:
    def __init__(self, config: FeatureConfig | Mapping | None = None):
        self.config = config if isinstance(config, FeatureConfig) else FeatureConfig.from_dict(config or {})

    @property
    def feature_dimension(self) -> int:
        return self.config.feature_dimension

    def to_dict(self) -> dict:
        return self.config.to_dict()

    @classmethod
    def from_dict(cls, values: Mapping) -> ClassicalFeatureExtractor:
        return cls(FeatureConfig.from_dict(values))

    def transform_image(self, image: Image.Image) -> np.ndarray:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image containing the shared face crop.")
        config = self.config
        gray = np.asarray(image.convert("L").resize((config.image_size, config.image_size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        branches = []
        if config.include_hog:
            hog = hog_descriptor(gray, cell_size=config.hog_cell_size, block_size=config.hog_block_size, block_stride=config.hog_block_stride, bins=config.hog_bins, clip=config.hog_clip)
            branches.append(config.hog_weight * hog / max(float(np.linalg.norm(hog)), 1e-12))
        if config.include_lbp:
            lbp = _lbp_histogram(gray, config)
            branches.append(config.lbp_weight * lbp / max(float(np.linalg.norm(lbp)), 1e-12))
        result = np.concatenate(branches).astype(np.float32, copy=False)
        # Branch-wise normalization prevents dimensionality alone determining
        # the relative contribution; the explicit weights remain tunable.
        result /= max(float(np.linalg.norm(result)), 1e-12)
        return result

    def transform(self, images: Iterable[Image.Image], out: np.ndarray | None = None) -> np.ndarray:
        """Extract one image at a time, optionally into a preallocated memmap.

        A generator requires ``out`` so its size need not be discovered by
        materializing all images or all descriptor vectors in memory.
        """
        if out is None:
            if not hasattr(images, "__len__"):
                raise ValueError("Provide out for image generators, or use a sized sequence.")
            out = np.empty((len(images), self.feature_dimension), dtype=np.float32)
        if not isinstance(out, np.ndarray) or out.ndim != 2 or out.shape[1] != self.feature_dimension or out.dtype != np.float32 or not out.flags.writeable:
            raise ValueError("out must be a writable float32 matrix with the configured feature dimension.")
        count = 0
        for count, image in enumerate(images, start=1):
            if count > len(out):
                raise ValueError("More images than output rows.")
            out[count - 1] = self.transform_image(image)
        if count != len(out):
            raise ValueError("Image count does not match output rows.")
        return out


def feature_vector(image: Image.Image, config: FeatureConfig | Mapping | None = None) -> np.ndarray:
    return ClassicalFeatureExtractor(config).transform_image(image)


def _feature_matrix(X: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    # Preserve memmaps and their dtype rather than copying an entire dataset.
    result = np.asarray(X)
    if result.ndim != 2 or result.shape[1] == 0 or (not allow_empty and result.shape[0] == 0) or result.dtype.kind not in "fiu":
        raise ValueError("X must be a nonempty numeric feature matrix.")
    for start in range(0, len(result), 256):
        if not np.isfinite(result[start : start + 256]).all():
            raise ValueError("X must contain only finite values.")
    return result


def _labels(y: np.ndarray, size: int, classes: int) -> np.ndarray:
    result = np.asarray(y)
    if result.shape != (size,) or result.dtype.kind not in "iu" or np.any(result < 0) or np.any(result >= classes):
        raise ValueError(f"y must contain integer class labels 0..{classes - 1}, one per row.")
    return result.astype(np.int64, copy=False)


def multiclass_hinge_loss_and_gradient(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: np.ndarray,
    *,
    regularization: float = 0.0,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Crammer–Singer max-margin objective and a subgradient.

    L = lambda/2 ||W||^2 + mean_i weight_i max(0, 1 +
        max_{j != y_i} score_ij - score_i,y_i).
    The intercept is unregularized. Tied competitors use the first index;
    derivatives at hinge/tie boundaries are not unique.
    """
    X = _feature_matrix(X)
    coef, intercept = np.asarray(coef), np.asarray(intercept)
    coef = coef.astype(np.result_type(coef.dtype, np.float32), copy=False)
    intercept = intercept.astype(np.result_type(intercept.dtype, np.float32), copy=False)
    if coef.ndim != 2 or coef.shape[0] < 2 or coef.shape[1] != X.shape[1] or intercept.shape != (len(coef),):
        raise ValueError("Coefficient/intercept shapes do not match X.")
    if not np.isfinite(coef).all() or not np.isfinite(intercept).all() or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Model values and nonnegative regularization must be finite.")
    y = _labels(y, len(X), len(coef))
    weights = np.ones(len(X), dtype=X.dtype) if sample_weight is None else np.asarray(sample_weight)
    if weights.shape != (len(X),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weight must be a finite nonnegative vector.")
    scores = X @ coef.T + intercept
    if not np.isfinite(scores).all():
        raise ValueError("Scores overflowed; rescale input features.")
    true_score = scores[np.arange(len(X)), y].copy()
    scores[np.arange(len(X)), y] = -np.inf
    competitor = np.argmax(scores, axis=1)
    margins = 1 + scores[np.arange(len(X)), competitor] - true_score
    active = margins > 0
    loss = 0.5 * regularization * float(np.sum(coef * coef)) + float(np.mean(weights * np.maximum(margins, 0)))
    delta = np.zeros((len(X), len(coef)), dtype=np.result_type(X, coef, np.float32))
    row = np.flatnonzero(active)
    delta[row, competitor[active]] = weights[active] / len(X)
    delta[row, y[active]] = -weights[active] / len(X)
    return loss, delta.T @ X + regularization * coef, delta.sum(axis=0)


class LinearSVM:
    """Minibatch multiclass linear SVM with train-only streaming scaling.

    ``fit`` never materializes a complete scaled copy of X, so X may be a
    float32 ``np.memmap``. Balanced weights are N/(K*n_class), giving average
    training sample weight one. SGD optimizes the objective documented above.
    """

    def __init__(self, num_classes: int = 3, *, regularization: float = 1e-3, learning_rate: float = 0.05, epochs: int = 60, batch_size: int = 128, class_weight: str | None = "balanced", standardize: bool = True, seed: int = 42, tolerance: float | None = None, patience: int = 5):
        for name, value, minimum in (("num_classes", num_classes, 2), ("epochs", epochs, 1), ("batch_size", batch_size, 1), ("patience", patience, 1)):
            _positive_integer(name, value, minimum)
        if not np.isfinite(regularization) or regularization < 0 or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Require finite regularization >= 0 and learning_rate > 0.")
        if class_weight not in (None, "balanced") or not isinstance(standardize, bool):
            raise ValueError("class_weight must be 'balanced' or None; standardize must be boolean.")
        if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
            raise ValueError("tolerance must be None or finite and nonnegative.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        self.num_classes, self.regularization, self.learning_rate = int(num_classes), float(regularization), float(learning_rate)
        self.epochs, self.batch_size, self.class_weight = int(epochs), int(batch_size), class_weight
        self.standardize, self.seed, self.tolerance, self.patience = standardize, int(seed), tolerance, int(patience)
        self.coef_ = self.intercept_ = self.feature_mean_ = self.feature_scale_ = self.class_weight_ = None
        self.classes_ = np.arange(self.num_classes, dtype=np.int64)
        self.loss_history_: list[float] = []
        self.n_iter_ = 0

    def _fit_scaler(self, X: np.ndarray) -> None:
        dimension = X.shape[1]
        mean, m2, count = np.zeros(dimension), np.zeros(dimension), 0
        if self.standardize:
            # Merge batch means/second central moments in float64, avoiding
            # cancellation and a full float64 copy of the feature matrix.
            for start in range(0, len(X), 256):
                block = np.asarray(X[start : start + 256], dtype=np.float64)
                block_mean = block.mean(axis=0)
                centered = block - block_mean
                block_m2 = np.einsum("ij,ij->j", centered, centered)
                delta, new_count = block_mean - mean, count + len(block)
                m2 += block_m2 + delta * delta * (count * len(block) / new_count)
                mean += delta * (len(block) / new_count)
                count = new_count
            scale = np.sqrt(np.maximum(m2 / count, 0))
            scale[scale < 1e-6] = 1.0
        else:
            scale = np.ones(dimension)
        self.feature_mean_, self.feature_scale_ = mean.astype(np.float32), scale.astype(np.float32)

    def _scaled(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            result = (np.asarray(X, dtype=np.float32) - self.feature_mean_) / self.feature_scale_
        if not np.isfinite(result).all():
            raise ValueError("Scaled features overflowed; rescale input data.")
        return result

    def _require_fitted(self) -> None:
        if self.coef_ is None:
            raise ValueError("Fit the SVM or load a fitted artifact first.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearSVM:
        X = _feature_matrix(X)
        y = _labels(y, len(X), self.num_classes)
        counts = np.bincount(y, minlength=self.num_classes)
        if np.any(counts == 0):
            raise ValueError("Training data must contain every configured class.")
        self._fit_scaler(X)
        self.class_weight_ = (len(X) / (self.num_classes * counts) if self.class_weight == "balanced" else np.ones(self.num_classes)).astype(np.float32)
        self.coef_ = np.zeros((self.num_classes, X.shape[1]), dtype=np.float32)
        self.intercept_ = np.zeros(self.num_classes, dtype=np.float32)
        self.loss_history_, self.n_iter_ = [], 0
        rng = np.random.default_rng(self.seed)
        best_loss, stale = float("inf"), 0
        for epoch in range(self.epochs):
            order = rng.permutation(len(X))
            rate = self.learning_rate / np.sqrt(1 + epoch)
            for start in range(0, len(X), self.batch_size):
                index = order[start : start + self.batch_size]
                _, grad_w, grad_b = multiclass_hinge_loss_and_gradient(self._scaled(X[index]), y[index], self.coef_, self.intercept_, regularization=self.regularization, sample_weight=self.class_weight_[y[index]])
                self.coef_ -= rate * grad_w
                self.intercept_ -= rate * grad_b
            loss = 0.5 * self.regularization * float(np.sum(self.coef_ * self.coef_))
            for start in range(0, len(X), self.batch_size):
                block_y = y[start : start + self.batch_size]
                block_loss, _, _ = multiclass_hinge_loss_and_gradient(self._scaled(X[start : start + self.batch_size]), block_y, self.coef_, self.intercept_, sample_weight=self.class_weight_[block_y])
                loss += block_loss * len(block_y) / len(X)
            if not np.isfinite(loss):
                raise ValueError("Training diverged; lower learning_rate or rescale features.")
            self.loss_history_.append(float(loss))
            self.n_iter_ = epoch + 1
            if self.tolerance is not None:
                if not np.isfinite(best_loss) or loss < best_loss - self.tolerance * max(1, abs(best_loss)):
                    best_loss, stale = loss, 0
                else:
                    stale += 1
                if stale >= self.patience:
                    break
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = _feature_matrix(X, allow_empty=True)
        if X.shape[1] != self.coef_.shape[1]:
            raise ValueError("Feature dimension differs from the fitted model.")
        output = np.empty((len(X), self.num_classes), dtype=np.float32)
        for start in range(0, len(X), self.batch_size):
            output[start : start + self.batch_size] = self._scaled(X[start : start + self.batch_size]) @ self.coef_.T + self.intercept_
        if not np.isfinite(output).all():
            raise ValueError("Prediction scores are not finite.")
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class indices; margins are not calibrated probabilities."""
        return np.argmax(self.decision_function(X), axis=1).astype(np.int64)

    def _configuration(self) -> dict:
        return {name: getattr(self, name) for name in ("num_classes", "regularization", "learning_rate", "epochs", "batch_size", "class_weight", "standardize", "seed", "tolerance", "patience")}

    @staticmethod
    def _paths(path: str | Path) -> tuple[Path, Path]:
        path = Path(path)
        array_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        return array_path, array_path.with_suffix(".json")

    def save(self, path: str | Path) -> tuple[Path, Path]:
        self._require_fitted()
        array_path, metadata_path = self._paths(path)
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(array_path, coef=self.coef_, intercept=self.intercept_, feature_mean=self.feature_mean_, feature_scale=self.feature_scale_, class_weight=self.class_weight_)
        metadata = {"format": "visage.linear_svm", "format_version": _FORMAT_VERSION, "configuration": self._configuration(), "n_features": self.coef_.shape[1], "class_names": list(CLASS_NAMES) if self.num_classes == 3 else None, "n_iter": self.n_iter_, "loss_history": self.loss_history_}
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        return array_path, metadata_path

    @classmethod
    def load(cls, path: str | Path) -> LinearSVM:
        array_path, metadata_path = cls._paths(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "visage.linear_svm" or metadata.get("format_version") != _FORMAT_VERSION:
            raise ValueError("Unsupported SVM artifact format/version.")
        model = cls(**metadata["configuration"])
        dimension = metadata["n_features"]
        _positive_integer("n_features", dimension)
        expected = {"coef": (model.num_classes, dimension), "intercept": (model.num_classes,), "feature_mean": (dimension,), "feature_scale": (dimension,), "class_weight": (model.num_classes,)}
        with np.load(array_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError("Artifact arrays are incomplete or unexpected.")
            values = {}
            for name, shape in expected.items():
                array = arrays[name]
                if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
                    raise ValueError(f"Invalid artifact array: {name}.")
                with np.errstate(over="ignore"):
                    values[name] = array.astype(np.float32)
                if not np.isfinite(values[name]).all():
                    raise ValueError(f"Artifact array exceeds float32 range: {name}.")
            if np.any(values["feature_scale"] <= 0) or np.any(values["class_weight"] <= 0):
                raise ValueError("Artifact scales and class weights must be positive.")
        model.coef_, model.intercept_ = values["coef"], values["intercept"]
        model.feature_mean_, model.feature_scale_, model.class_weight_ = values["feature_mean"], values["feature_scale"], values["class_weight"]
        history = np.asarray(metadata.get("loss_history", []), dtype=float)
        if history.ndim != 1 or not np.isfinite(history).all() or np.any(history < 0):
            raise ValueError("Invalid loss history in artifact.")
        model.loss_history_ = history.tolist()
        model.n_iter_ = int(metadata.get("n_iter", len(history)))
        return model


class ClassicalAgeClassifier:
    """Bundle feature settings with the SVM; never persist visitor images."""

    def __init__(self, feature_config: FeatureConfig | Mapping | None = None, svm: LinearSVM | None = None):
        self.extractor = ClassicalFeatureExtractor(feature_config)
        self.svm = svm if svm is not None else LinearSVM()
        if self.svm.num_classes != 3:
            raise ValueError("Age classification requires exactly three classes.")

    def fit_features(self, X: np.ndarray, y: np.ndarray) -> ClassicalAgeClassifier:
        if np.asarray(X).ndim != 2 or np.asarray(X).shape[1] != self.extractor.feature_dimension:
            raise ValueError("X does not match the configured descriptor dimension.")
        self.svm.fit(X, y)
        return self

    def fit(self, images: Sequence[Image.Image], y: np.ndarray, *, out: np.ndarray | None = None) -> ClassicalAgeClassifier:
        return self.fit_features(self.extractor.transform(images, out=out), y)

    def decision_function(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.svm.decision_function(self.extractor.transform(images))

    def predict(self, images: Sequence[Image.Image]) -> np.ndarray:
        return np.argmax(self.decision_function(images), axis=1).astype(np.int64)

    def predict_image(self, image: Image.Image) -> int:
        return int(self.predict([image])[0])

    def save(self, directory: str | Path) -> Path:
        self.svm._require_fitted()
        if self.svm.coef_.shape[1] != self.extractor.feature_dimension:
            raise ValueError("SVM and feature settings have different dimensions.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.svm.save(directory / "svm.npz")
        (directory / "features.json").write_text(json.dumps({"format": "visage.classical_features", "format_version": _FORMAT_VERSION, "configuration": self.extractor.to_dict()}, indent=2, allow_nan=False), encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ClassicalAgeClassifier:
        directory = Path(directory)
        metadata = json.loads((directory / "features.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "visage.classical_features" or metadata.get("format_version") != _FORMAT_VERSION:
            raise ValueError("Unsupported feature artifact format/version.")
        model = cls(FeatureConfig.from_dict(metadata["configuration"]), LinearSVM.load(directory / "svm.npz"))
        if model.svm.coef_.shape[1] != model.extractor.feature_dimension:
            raise ValueError("SVM and feature artifact dimensions differ.")
        return model
