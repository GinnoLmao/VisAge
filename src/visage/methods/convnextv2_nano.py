"""Published ConvNeXt V2 Nano adapted to three age groups."""
from ..pretrained_models import PretrainedAgeModel


def create_model(**kwargs):
    return PretrainedAgeModel('convnextv2_nano', **kwargs)
