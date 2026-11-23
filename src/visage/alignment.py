"""Deterministic orientation-preserving similarity alignment, no learned weights."""
from __future__ import annotations
import numpy as np

# Coordinate-wise medians of498 unambiguous faces from500 fixed training crops.
# reports/training_alignment_comparison.json records sampling and reconstruction.
TRAINING_TEMPLATE = (
    (0.29279454231262203, 0.28809377670288083),
    (0.6810825729370118, 0.29309667587280275),
    (0.47622774124145506, 0.4994986152648926),
    (0.3146715354919434, 0.6784325408935548),
    (0.642569808959961, 0.6859370422363281),
)


def similarity_transform(source, target):
    """Least-squares uniform scale/rotation/translation; never mirror or shear."""
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape != target.shape or source.shape[1] != 2 or len(source) < 2:
        raise ValueError("Matching N-by-2 landmark arrays required.")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Finite landmarks required.")
    center, destination = source.mean(0), target.mean(0)
    x, y = source - center, target - destination
    denominator = float((x*x).sum())
    if denominator < 1e-8:
        raise ValueError("Degenerate source landmarks.")
    a = float((x*y).sum() / denominator)
    b = float((x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0]).sum() / denominator)
    matrix = np.array([[a, -b], [b, a]])
    if a*a+b*b < 1e-10:
        raise ValueError("Degenerate target alignment.")
    return np.c_[matrix, destination - matrix @ center]
