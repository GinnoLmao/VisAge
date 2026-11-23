"""Published Swin Transformer V2 Tiny adapted to three age groups."""
from ..pretrained_models import PretrainedAgeModel


def create_model(**kwargs):
    return PretrainedAgeModel('swinv2_tiny', **kwargs)
