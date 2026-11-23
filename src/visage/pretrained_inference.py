"""Strict v2 serving adapter using the identical canonical full-face transform."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from . import CLASS_NAMES
from .augmentation import prepare_training_batch
from .model_registry import REGISTRY
from .pretrained_data import GEOMETRY, pixels
from .pretrained_models import PretrainedAgeModel
from .runtime import sha256_file


class PretrainedPredictor:
    def __init__(self, method, artifact_dir, device='cpu'):
        if method not in REGISTRY:
            raise ValueError(f'Unknown active method: {method}')
        self.method, self.spec = method, REGISTRY[method]
        self.directory, self.device = Path(artifact_dir), torch.device(device)
        if self.device.type not in ('cpu', 'cuda') or (self.device.type == 'cuda' and not torch.cuda.is_available()):
            raise RuntimeError('Requested inference device is unavailable.')
        summary = json.loads((self.directory / 'training_summary.json').read_text(encoding='utf-8'))
        self.artifact_sha256 = sha256_file(self.directory / 'best.pt')
        if self.artifact_sha256 != summary['artifact_sha256'] or summary['method'] != method:
            raise ValueError('Serving checkpoint differs from the completed selected run.')
        saved = torch.load(self.directory / 'best.pt', map_location='cpu', weights_only=True)
        self.metadata = saved['metadata']
        expected_input = json.loads(json.dumps(self.spec.to_dict()))
        metadata_input = json.loads(json.dumps(self.metadata.get('input')))
        if self.metadata.get('format_version') != 2 or self.metadata['method'] != method or self.metadata['architecture'] != self.spec.checkpoint:
            raise ValueError('Artifact architecture/schema does not match the requested model.')
        if metadata_input != expected_input or self.metadata['geometry'] != GEOMETRY:
            raise ValueError('Artifact transform does not match the canonical serving transform.')
        for key in ('manifest_sha256', 'protocol_sha256', 'run_id', 'seed'):
            if summary[key] != self.metadata[key]:
                raise ValueError(f'Artifact and selected run differ: {key}')
        if self.metadata['head_type'] != 'softmax' or self.metadata['seed'] != 42:
            raise ValueError('Serving requires the declared seed42 three-class softmax checkpoint.')
        self.model = PretrainedAgeModel(method)
        self.model.load_state_dict(saved['state_dict'], strict=True)
        self.model.to(self.device)
        if self.metadata['memory_format'] == 'channels_last':
            self.model.to(memory_format=torch.channels_last)
        self.model.eval()

    @torch.inference_mode()
    def predict_batch(self, arrays):
        values = torch.as_tensor(np.stack(arrays), dtype=torch.uint8, device=self.device)
        inputs = prepare_training_batch(values, self.spec.mean, self.spec.std, augment=False)
        if self.metadata['memory_format'] == 'channels_last':
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        scores = self.model(inputs).softmax(1)
        if not torch.isfinite(scores).all():
            raise RuntimeError('Nonfinite model output.')
        return scores.cpu().numpy()

    def predict(self, image: Image.Image):
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        scores = self.predict_batch([pixels(image)])[0]
        label = int(scores.argmax())
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return {'method': self.method, 'class_id': label, 'age_group': CLASS_NAMES[label],
            'inference_and_preprocessing_ms': (time.perf_counter() - started) * 1000,
            'manifest_sha256': self.metadata['manifest_sha256'], 'device': str(self.device)}
