"""Application extension exposing scores from the frozen inference transform.

The research adapter and evaluation protocol remain unchanged. This subclass
uses the inherited FP32 softmax computation once and preserves its argmax rule.
"""
import time
import numpy as np
import torch
from . import CLASS_NAMES
from .pretrained_data import pixels
from .pretrained_inference import PretrainedPredictor

SCORE_TYPE = 'uncalibrated_softmax'
SCORE_NOTICE = 'Model scores compare age groups; they are not verified probabilities of your actual age.'


class ScoredPretrainedPredictor(PretrainedPredictor):
    def predict(self, image):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        scores = np.asarray(self.predict_batch([pixels(image)])[0])
        if scores.shape != (len(CLASS_NAMES),) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1) or not np.isclose(scores.sum(), 1, atol=1e-6, rtol=0):
            raise RuntimeError('Invalid three-class softmax scores.')
        label = int(scores.argmax())
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        return {'method': self.method, 'class_id': label, 'age_group': CLASS_NAMES[label],
            'class_scores': [{'class_id': index, 'age_group': name, 'score': float(scores[index])}
                for index, name in enumerate(CLASS_NAMES)],
            'score_type': SCORE_TYPE, 'score_notice': SCORE_NOTICE,
            'inference_and_preprocessing_ms': elapsed,
            'manifest_sha256': self.metadata['manifest_sha256'], 'device': str(self.device)}
