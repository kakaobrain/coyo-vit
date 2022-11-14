"""
Implements top K precision.
(https://github.com/google-research/scenic/blob/c077db56a64446cb3c7578a92186da31ee3ecc36/scenic/model_lib/base_models/multilabel_classification_model.py#L29)
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked
from typing import Optional


class TopKPrecision(tf.keras.metrics.Metric):
    @typechecked
    def __init__(
            self,
            k: FloatTensorLike = 1,
            name: str = "topk_precision",
            dtype: AcceptableDTypes = None,
            **kwargs,
    ):
        super().__init__(name=name, dtype=dtype)

        self.k = k
        self.axis = None
        self.init_shape = []

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        y_pred_sort = tf.argsort(y_pred, direction='DESCENDING', axis=-1)[..., :self.k]
        positives = tf.reduce_max(tf.gather(params=y_true, indices=y_pred_sort, axis=-1, batch_dims=-1), axis=-1)
        negatives = 1 - positives

        self.true_positives.assign_add(tf.cast(tf.reduce_sum(positives), dtype=self.dtype))
        self.false_positives.assign_add(tf.cast(tf.reduce_sum(negatives), dtype=self.dtype))

    def result(self):
        return tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "k": self.k,
        }

        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = tf.zeros(self.init_shape, dtype=self.dtype)
        K.batch_set_value([(v, reset_value) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()
