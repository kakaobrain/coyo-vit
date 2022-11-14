# https://github.com/tensorflow/models/blob/master/official/vision/beta/modeling/classification_model.py

"""Build classification models."""

import re
# Import libraries
import tensorflow as tf
from typing import Any, Mapping, Optional, Union


class ClassificationModel(tf.keras.Model):
    """A classification class builder."""

    def __init__(
            self,
            backbone: tf.keras.Model,
            num_classes: int,
            input_specs: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
            dropout_rate: float = 0.0,
            kernel_initializer: Union[str, tf.keras.initializers.Initializer] = 'random_uniform',
            bias_initializer: Union[str, tf.keras.initializers.Initializer] = 'zeros',
            skip_logits_layer: bool = False,
            weight_decay: float = 0.0,
            clip_grad_norm: float = 0.0,
            **kwargs):
        """Classification initialization function.

        Args:
          backbone: a backbone network.
          num_classes: `int` number of classes in classification task.
          input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
          dropout_rate: `float` rate for dropout regularization.
          kernel_initializer: kernel initializer for the dense layer.
          kernel_regularizer: tf.keras.regularizers.Regularizer object. Default to
                              None.
          bias_regularizer: tf.keras.regularizers.Regularizer object. Default to
                              None.
          skip_logits_layer: `bool`, whether to skip the prediction layer.
          **kwargs: keyword arguments to be passed.
        """
        inputs = tf.keras.Input(shape=input_specs.shape[1:], name=input_specs.name)
        outputs = backbone(inputs)

        outputs = tf.keras.layers.GlobalAveragePooling2D()(outputs)
        if not skip_logits_layer:
            if dropout_rate is not None and dropout_rate > 0:
                outputs = tf.keras.layers.Dropout(dropout_rate)(outputs)
            outputs = tf.keras.layers.Dense(
                num_classes,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer
            )(outputs)

        super(ClassificationModel, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

        self._config_dict = {
            'backbone': backbone,
            'num_classes': num_classes,
            'input_specs': input_specs,
            'dropout_rate': dropout_rate,
            'kernel_initializer': kernel_initializer,
            'weight_decay': weight_decay,
            'clip_grad_norm': clip_grad_norm,
        }
        self._input_specs = input_specs
        self._backbone = backbone
        self._weight_decay = weight_decay
        self._clip_grad_norm = clip_grad_norm

    def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
        """Return regularization l2 loss loss."""
        var_match = re.compile(regex)
        return weight_decay * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self.trainable_variables
            if var_match.match(v.name)
        ])

    def train_step(self, data):
        features, labels = data
        images, labels = features['image'], labels['label']

        with tf.GradientTape() as tape:
            pred = self(images, training=True)
            pred = tf.cast(pred, tf.float32)
            loss = self.compiled_loss(
                labels,
                pred,
                regularization_losses=[self._reg_l2_loss(self._weight_decay)])

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(labels, pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        features, labels = data
        images, labels = features['image'], labels['label']
        pred = self(images, training=False)
        pred = tf.cast(pred, tf.float32)

        self.compiled_loss(
            labels,
            pred,
            regularization_losses=[self._reg_l2_loss(self._weight_decay)])

        self.compiled_metrics.update_state(labels, pred)
        return {m.name: m.result() for m in self.metrics}

    @property
    def checkpoint_items(self) -> Mapping[str, tf.keras.Model]:
        """Returns a dictionary of items to be additionally checkpointed."""
        return dict(backbone=self.backbone)

    @property
    def backbone(self) -> tf.keras.Model:
        return self._backbone

    def get_config(self) -> Mapping[str, Any]:
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
