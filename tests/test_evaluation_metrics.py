import numpy as np
import pytest
from visage.evaluation_metrics import binary_roc, box_iou, detection_ap, multiclass_roc


def test_roc_known_auc_and_ties():
    assert binary_roc([0, 0, 1, 1], [.1, .4, .35, .8])["auc"] == .75
    assert binary_roc([0, 1, 0, 1], [1, 1, 1, 1])["auc"] == .5
    assert binary_roc([0, 1], [.9, .1])["auc"] == 0
    assert binary_roc([0, 1], [.1, .9])["thresholds"][0] is None


def test_roc_requires_both_labels_and_finite_scores():
    for labels, scores in (([1, 1], [0, 1]), ([0, 1], [0, np.nan]), ([0, 2], [0, 1]), ([], [])):
        with pytest.raises(ValueError):
            binary_roc(labels, scores)


def test_tied_roc_auc_matches_independent_pairwise_rank_probability():
    rng = np.random.default_rng(91)
    truth = np.r_[np.zeros(25, dtype=int), np.ones(25, dtype=int)]
    scores = rng.integers(0, 8, size=50)
    expected = np.mean([float(p > n) + .5 * float(p == n) for p in scores[truth == 1] for n in scores[truth == 0]])
    assert binary_roc(truth, scores)["auc"] == pytest.approx(expected)


def test_multiclass_perfect_curves_and_score_scale_invariance():
    result = multiclass_roc([0, 1, 2], np.eye(3))
    assert result["macro_auc_mean"] == 1
    assert all(curve["auc"] == 1 for curve in result["per_class"].values())
    assert multiclass_roc([0, 1, 2], np.eye(3) * 5 - 3)["macro_auc_mean"] == 1


def test_iou_continuous_geometry():
    np.testing.assert_allclose(box_iou([0, 0, 2, 2], [[0, 0, 2, 2], [1, 0, 2, 2], [3, 3, 1, 1]]), [1, 1 / 3, 0])


def test_ap_duplicate_false_positive_and_missed_face():
    gt = {"a": {"boxes": [[0, 0, 10, 10], [20, 20, 10, 10]]}}
    pred = {"a": [{"bbox": [0, 0, 10, 10], "score": .9}, {"bbox": [0, 0, 10, 10], "score": .8},
                  {"bbox": [20, 20, 10, 10], "score": .7}]}
    result = detection_ap(gt, pred)
    assert result["ap"] == pytest.approx(5 / 6)
    assert result["false_positives"] == 1
    assert detection_ap(gt, {"a": pred["a"][:1]})["ap"] == .5
    assert detection_ap(gt, {})["ap"] == 0


def test_negative_images_and_ignore_boxes_have_distinct_accounting():
    gt = {"face": {"boxes": [[0, 0, 10, 10]], "ignore_boxes": [[20, 20, 5, 5]]}, "empty": {"boxes": []}}
    pred = {"empty": [{"bbox": [0, 0, 10, 10], "score": .99}], "face": [
        {"bbox": [20, 20, 5, 5], "score": .98}, {"bbox": [0, 0, 10, 10], "score": .9}]}
    result = detection_ap(gt, pred)
    assert result["ap"] == .5 and result["ignored_predictions"] == 1
    assert result["false_positives_per_image"][-1] == .5


def test_matching_is_one_to_one_and_iou_threshold_matters():
    gt = {"a": {"boxes": [[0, 0, 10, 10]]}}
    pred = {"a": [{"bbox": [2, 0, 10, 10], "score": .9}]}
    assert detection_ap(gt, pred, .5)["ap"] == 1
    assert detection_ap(gt, pred, .75)["ap"] == 0
    with pytest.raises(ValueError):
        detection_ap(gt, {"unknown": []})
    with pytest.raises(ValueError):
        detection_ap(gt, {"a": [{"bbox": [0, 0, -1, 1], "score": .1}]})
