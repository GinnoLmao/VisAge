"""Fixed-vocabulary metrics, including boundary errors and explicit counts."""

from __future__ import annotations
import numpy as np
from . import CLASS_NAMES


def classification_metrics(y_true, y_pred) -> dict:
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim != 1 or pred.shape != truth.shape or not len(truth):
        raise ValueError("Expected equal, nonempty one-dimensional labels.")
    if truth.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise ValueError("Labels must be integer class indices.")
    if np.any((truth < 0) | (truth > 2) | (pred < 0) | (pred > 2)):
        raise ValueError("Labels must be 0, 1, or 2.")
    cm = np.bincount(3 * truth + pred, minlength=9).reshape(3, 3)
    tp = cm.diagonal().astype(float)
    support = cm.sum(1)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(3), where=cm.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
    rate = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    return {
        "samples": int(len(truth)), "accuracy": float(tp.sum() / len(truth)),
        "macro_f1": float(f1.mean()), "confusion_matrix": cm.tolist(),
        "row_normalized_confusion_matrix": np.divide(cm, support[:, None], out=np.zeros((3, 3)), where=support[:, None] > 0).tolist(),
        "per_class": [{"label": k, "name": CLASS_NAMES[k], "support": int(support[k]),
                       "precision": float(precision[k]), "recall": float(recall[k]), "f1": float(f1[k])} for k in range(3)],
        "under18_predicted_older_rate": rate(cm[0, 1:].sum(), support[0]),
        "senior_predicted_below60_rate": rate(cm[2, :2].sum(), support[2]),
        "non_senior_predicted_senior_rate": rate(cm[:2, 2].sum(), support[:2].sum()),
    }
