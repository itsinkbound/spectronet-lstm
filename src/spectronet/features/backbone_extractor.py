"""ImageNet-pretrained CNN backbones used as feature extractors.

Per Section 3.2: each spectrogram image is passed through ResNet-101,
VGG-16 and Inception-V3; the pooled feature vectors are concatenated
("feature fusion") before being fed to the LSTM head. During the initial
training phase all backbone layers are frozen; during fine-tuning the last
`fine_tune_last_n_layers` layers of each backbone are unfrozen.
"""
from __future__ import annotations

from typing import Dict, List

import tensorflow as tf
from tensorflow.keras import layers, models

_BACKBONE_FACTORY = {
    "resnet101": (
        tf.keras.applications.ResNet101,
        tf.keras.applications.resnet.preprocess_input,
        224,
    ),
    "vgg16": (
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
        224,
    ),
    "inception_v3": (
        tf.keras.applications.InceptionV3,
        tf.keras.applications.inception_v3.preprocess_input,
        299,
    ),
}


def build_backbone(name: str, trainable: bool = False) -> tf.keras.Model:
    if name not in _BACKBONE_FACTORY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(_BACKBONE_FACTORY)}")
    factory, _, input_size = _BACKBONE_FACTORY[name]
    base = factory(include_top=False, weights="imagenet", pooling="avg",
                    input_shape=(input_size, input_size, 3))
    base.trainable = trainable
    base._name = f"{name}_backbone"
    return base


def preprocess_for_backbone(name: str, images: tf.Tensor) -> tf.Tensor:
    _, preprocess_fn, input_size = _BACKBONE_FACTORY[name]
    resized = tf.image.resize(images, (input_size, input_size))
    return preprocess_fn(resized)


def unfreeze_last_n_layers(backbone: tf.keras.Model, n: int) -> None:
    """Fine-tuning phase: unlock only the last `n` layers of a backbone,
    matching the paper's 'last 50 layers unlocked' strategy."""
    backbone.trainable = True
    freeze_until = max(0, len(backbone.layers) - n)
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False


def build_multi_backbone_feature_extractor(
    backbone_names: List[str], trainable: bool = False
) -> Dict[str, tf.keras.Model]:
    """Instantiate one frozen (by default) backbone per name."""
    return {name: build_backbone(name, trainable=trainable) for name in backbone_names}


def fuse_backbone_outputs(
    image_input: tf.Tensor, backbones: Dict[str, tf.keras.Model]
) -> tf.Tensor:
    """Run the same spectrogram image through every backbone and concatenate
    the pooled embeddings -> the paper's "feature fusion" step."""
    features = []
    for name, backbone in backbones.items():
        prepped = preprocess_for_backbone(name, image_input)
        features.append(backbone(prepped))
    return layers.Concatenate(name="fused_features")(features)
