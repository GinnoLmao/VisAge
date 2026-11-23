"""Application score export preserves the frozen predictor's one-pass decision."""
import numpy as np
import pytest
import torch
from PIL import Image
from visage import CLASS_NAMES
from visage.model_registry import METHODS
from visage.score_inference import ScoredPretrainedPredictor


def predictor(method, values):
    value = object.__new__(ScoredPretrainedPredictor)
    value.method = method
    value.device = torch.device('cpu')
    value.metadata = {'manifest_sha256': 'a' * 64}
    calls = []
    def score(arrays):
        calls.append(arrays)
        return np.asarray([values], dtype=np.float32)
    value.predict_batch = score
    return value, calls


@pytest.mark.parametrize('method', METHODS)
def test_all_three_unrounded_scores_export_in_fixed_class_order_with_one_pass(method):
    adapter, calls = predictor(method, [.6, .4, 0])
    with Image.new('RGB', (200, 200), 'white') as image:
        result = adapter.predict(image)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert result['class_id'] == 0 and result['age_group'] == CLASS_NAMES[0]
    assert result['score_type'] == 'uncalibrated_softmax'
    assert [v['class_id'] for v in result['class_scores']] == [0, 1, 2]
    assert [v['age_group'] for v in result['class_scores']] == list(CLASS_NAMES)
    assert [v['score'] for v in result['class_scores']] == np.array([.6, .4, 0], dtype=np.float32).tolist()
    assert result['inference_and_preprocessing_ms'] >= 0


@pytest.mark.parametrize('scores', [[.4, .3], [.6, .6, .1], [-.1, .7, .4], [float('nan'), .4, .6], [float('inf'), 0, 0]])
def test_invalid_model_scores_are_never_exported(scores):
    adapter, _ = predictor(METHODS[0], scores)
    with Image.new('RGB', (200, 200)) as image:
        with pytest.raises(RuntimeError, match='softmax scores'):
            adapter.predict(image)


def test_equal_top_scores_preserve_the_original_argmax_tie_rule():
    adapter, _ = predictor(METHODS[0], [.5, .5, 0])
    with Image.new('RGB', (200, 200)) as image:
        assert adapter.predict(image)['class_id'] == 0
