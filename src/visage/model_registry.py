"""The three active published pretrained age-group classifiers."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    checkpoint: str
    family: str
    classifier: str
    mean: tuple
    std: tuple
    image_size: int = 256

    def to_dict(self):
        return asdict(self)


REGISTRY = {
    'convnextv2_nano': ModelSpec('convnextv2_nano', 'ConvNeXt V2 Nano', 'convnextv2_nano.fcmae_ft_in1k',
        'Modern convolutional backbone', 'head.fc', (.485, .456, .406), (.229, .224, .225)),
    'efficientnetv2_s': ModelSpec('efficientnetv2_s', 'EfficientNetV2-S', 'tf_efficientnetv2_s.in1k',
        'Efficient convolutional backbone', 'classifier', (.5, .5, .5), (.5, .5, .5)),
    'swinv2_tiny': ModelSpec('swinv2_tiny', 'Swin Transformer V2 Tiny', 'swinv2_tiny_window8_256.ms_in1k',
        'Hierarchical windowed Transformer', 'head.fc', (.485, .456, .406), (.229, .224, .225)),
}
METHODS = tuple(REGISTRY)
ARTIFACTS = ROOT / 'artifacts/pretrained/serving'
EVALUATION = ROOT / 'reports/pretrained/evaluation'


def serving_default(directory=None):
    import json
    marker = Path(directory or ARTIFACTS) / 'bundle.json'
    if marker.is_file():
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record.get('default_method') not in REGISTRY or set(record.get('methods', {})) != set(METHODS):
            raise ValueError('Invalid three-pretrained-model serving bundle.')
        return record['default_method']
    return METHODS[0]
