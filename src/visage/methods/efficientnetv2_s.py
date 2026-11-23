"""Published EfficientNetV2-S adapted to three age groups."""
from ..pretrained_models import PretrainedAgeModel


def create_model(**kwargs):
    return PretrainedAgeModel('efficientnetv2_s', **kwargs)
