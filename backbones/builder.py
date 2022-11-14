import tensorflow as tf
from typing import Dict

from .vit.vit_model import create_name_vit


def _build_vision_transformer(model_name: str, model_config: Dict, **kwargs):
    return create_name_vit(model_name, **model_config)


def build_backbone(name: str, params: Dict = None) -> tf.keras.Model:
    if name.startswith('vit'):
        return _build_vision_transformer(name, params)
    else:
        raise ValueError('unsupported backbone name')
