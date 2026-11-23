"""Tie-aware classification ROC and explicitly defined box-detection AP.

This independent module extends evaluation without changing the frozen training
metric implementation used by the original/research experiments.
"""
from __future__ import annotations

import numpy as np


def binary_roc(truth, scores):
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=float)
    if truth.ndim != 1 or scores.shape != truth.shape or not len(truth):
        raise ValueError("Equal nonempty one-dimensional labels and scores required.")
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Binary labels and finite scores required.")
    positives, negatives = int(truth.sum()), int((truth == 0).sum())
    if not positives or not negatives:
        raise ValueError("ROC requires both positive and negative examples.")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    true_positive = np.cumsum(truth[order])[ends]
    false_positive = 1 + ends - true_positive
    tpr, fpr = np.r_[0.0, true_positive / positives], np.r_[0.0, false_positive / negatives]
    return {"auc": float(np.trapezoid(tpr, fpr)), "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": [None, *ordered[ends].tolist()],
            "threshold_note": "None at the first point denotes a threshold above every observed score.",
            "positive_samples": positives, "negative_samples": negatives}


def multiclass_roc(truth, scores):
    truth, scores = np.asarray(truth), np.asarray(scores, dtype=float)
    if truth.ndim != 1 or scores.shape != (len(truth), 3) or not np.isin(truth, [0, 1, 2]).all():
        raise ValueError("Three-class labels and an N-by-3 score matrix required.")
    curves = {str(label): binary_roc(truth == label, scores[:, label]) for label in range(3)}
    common = np.unique(np.concatenate([curve["fpr"] for curve in curves.values()]))
    macro_tpr = np.mean([np.interp(common, curve["fpr"], curve["tpr"]) for curve in curves.values()], axis=0)
    return {"per_class": curves, "macro_auc_mean": float(np.mean([curve["auc"] for curve in curves.values()])),
            "macro_curve": {"fpr": common.tolist(), "tpr": macro_tpr.tolist()},
            "definition": "One class versus the other two. Macro AUC is the arithmetic mean of three class AUCs; scores need not be calibrated probabilities."}


def box_iou(box, others):
    box, others = np.asarray(box, dtype=float), np.asarray(others, dtype=float).reshape(-1, 4)
    if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(others).all() or (box[2:] <= 0).any() or (others[:, 2:] <= 0).any():
        raise ValueError("Finite xywh boxes with positive width/height required.")
    near = np.maximum(box[:2], others[:, :2])
    far = np.minimum(box[:2] + box[2:], others[:, :2] + others[:, 2:])
    intersection = np.maximum(far - near, 0).prod(1)
    return intersection / (box[2:].prod() + others[:, 2:].prod(1) - intersection)


def detection_ap(ground_truth: dict, predictions: dict, iou_threshold=0.5):
    """Global score-ranked greedy one-to-one matching; all-points envelope AP.

    Ground truth maps every image ID to ``boxes`` and optional ``ignore_boxes``.
    Predictions map image IDs to ``{bbox:xywh,score:float}`` lists. A duplicate
    valid-box hit is a false positive; unmatched overlap with ignored boxes is
    ignored. Images with no faces still contribute false detections. This is a
    declared general box protocol, not the official WIDER easy/medium/hard score.
    """
    if not 0 < iou_threshold <= 1 or not ground_truth or set(predictions) - set(ground_truth):
        raise ValueError("Known image IDs and an IoU threshold in (0,1] required.")
    boxes, ignored, used, ranked = {}, {}, {}, []
    total = 0
    for image_id, annotations in ground_truth.items():
        boxes[image_id] = np.asarray(annotations["boxes"], dtype=float).reshape(-1, 4)
        ignored[image_id] = np.asarray(annotations.get("ignore_boxes", []), dtype=float).reshape(-1, 4)
        for value in (boxes[image_id], ignored[image_id]):
            if not np.isfinite(value).all() or (value[:, 2:] <= 0).any():
                raise ValueError("Invalid ground-truth box.")
        used[image_id] = set()
        total += len(boxes[image_id])
        for index, pred in enumerate(predictions.get(image_id, [])):
            score = float(pred["score"])
            box_iou(pred["bbox"], [])  # Validate even a prediction with no matching GT.
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Detector confidence scores must be finite in [0,1].")
            ranked.append((score, str(image_id), index, pred["bbox"], image_id))
    if not total:
        raise ValueError("Detection AP requires at least one nonignored ground-truth face.")
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    tp, fp, thresholds, ignore_count = [], [], [], 0
    for score, _, _, box, image_id in ranked:
        overlaps = box_iou(box, boxes[image_id])
        match = int(overlaps.argmax()) if len(overlaps) else None
        if match is not None and overlaps[match] >= iou_threshold:
            fresh = match not in used[image_id]
            tp.append(int(fresh)); fp.append(int(not fresh)); used[image_id].add(match)
        elif np.any(box_iou(box, ignored[image_id]) >= iou_threshold):
            ignore_count += 1
            continue
        else:
            tp.append(0); fp.append(1)
        thresholds.append(score)
    cumulative_tp, cumulative_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cumulative_tp / total
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    extended_recall = np.r_[0.0, recall, 1.0]
    envelope = np.maximum.accumulate(np.r_[0.0, precision, 0.0][::-1])[::-1]
    changes = np.flatnonzero(extended_recall[1:] != extended_recall[:-1])
    ap = float(np.sum((extended_recall[changes + 1] - extended_recall[changes]) * envelope[changes + 1]))
    return {"ap": ap, "ap_percent": ap * 100, "iou_threshold": iou_threshold,
            "ground_truth_faces": total, "images": len(ground_truth), "predictions": len(ranked),
            "ignored_predictions": ignore_count, "true_positives": int(cumulative_tp[-1]) if len(tp) else 0,
            "false_positives": int(cumulative_fp[-1]) if len(fp) else 0,
            "precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds,
            "false_positives_per_image": (cumulative_fp / len(ground_truth)).tolist(),
            "protocol": "Continuous-coordinate xywh IoU; global descending scores; image-ID/index tie break; one-to-one valid-box matching; ignored-box exclusion; all-points interpolated AP."}
