import numpy as np
import pytest
from visage.metrics import classification_metrics


def test_hand_computed_metrics_and_boundaries():
    result = classification_metrics([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2])
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["under18_predicted_older_rate"] == 0.5
    assert result["senior_predicted_below60_rate"] == 0
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
    assert result["macro_f1"] == pytest.approx((2 / 3 + .5 + .8) / 3)


def test_missing_class_is_not_omitted_from_macro_f1():
    result = classification_metrics([1, 1], [1, 1])
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["under18_predicted_older_rate"] is None


@pytest.mark.parametrize("a,b", [([], []), ([3], [0]), ([0], [2, 1]), ([.1], [0])])
def test_invalid_labels_rejected(a, b):
    with pytest.raises(ValueError):
        classification_metrics(a, b)
