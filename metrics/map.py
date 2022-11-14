# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implements mean average precision."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike
from typeguard import typechecked
from typing import Optional


class MeanAveragePrecision(tf.keras.metrics.Metric):
    @typechecked
    def __init__(
            self,
            num_classes: FloatTensorLike,
            average: str = 'micro',
            name: str = "mAP",
            dtype: AcceptableDTypes = None,
            **kwargs,
    ):
        super().__init__(name=name, dtype=dtype)

        if average not in (None, "micro", "macro"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: ['micro', 'macro']"
            )

        self.num_classes = num_classes
        self.average = average
        self.axis = None
        self.init_shape = []

        if self.average == "macro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.positive_cumsum = _zero_wt_init("positive_cumsum")
        self.positive_total = _zero_wt_init("positive_total")

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        indices = tf.argsort(y_pred, direction='DESCENDING', axis=0)
        y_true_sort = tf.gather(tf.transpose(y_true),
                                tf.transpose(indices),
                                axis=-1,
                                batch_dims=-1)
        y_true_sort = tf.transpose(y_true_sort)
        pos_count = tf.math.cumsum(y_true_sort, axis=0)
        total = pos_count[-1]
        if self.average == 'micro':
            total = tf.math.reduce_sum(total)
        pos_count = tf.math.multiply(pos_count, y_true_sort)

        total_count = tf.math.cumsum(tf.ones((y_true.shape[0], self.num_classes), dtype=self.dtype))
        pp = tf.math.divide_no_nan(pos_count, total_count)
        pp = tf.math.reduce_sum(pp, axis=self.axis)

        self.positive_cumsum.assign_add(pp)
        self.positive_total.assign_add(total)

    def result(self):
        precision_at_i = tf.math.divide_no_nan(
            self.positive_cumsum, self.positive_total
        )
        return tf.math.reduce_mean(precision_at_i)

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
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
