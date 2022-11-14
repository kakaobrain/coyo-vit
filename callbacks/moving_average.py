# https://github.com/tensorflow/models/blob/master/official/vision/image_classification/callbacks.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
from typing import Any, List, MutableMapping, Optional, Text

from optimizers.ema import ExponentialMovingAverage


class MovingAverageCallback(tf.keras.callbacks.Callback):
    """A Callback to be used with a `ExponentialMovingAverage` optimizer.
    Applies moving average weights to the model during validation time to test
    and predict on the averaged weights rather than the current model weights.
    Once training is complete, the model weights will be overwritten with the
    averaged weights (by default).
    Attributes:
      overwrite_weights_on_train_end: Whether to overwrite the current model
        weights with the averaged weights from the moving average optimizer.
      **kwargs: Any additional callback arguments.
    """

    def __init__(self, overwrite_weights_on_train_end: bool = False, **kwargs):
        super(MovingAverageCallback, self).__init__(**kwargs)
        self.overwrite_weights_on_train_end = overwrite_weights_on_train_end

    def set_model(self, model: tf.keras.Model):
        super(MovingAverageCallback, self).set_model(model)
        assert isinstance(self.model.optimizer, ExponentialMovingAverage)
        self.model.optimizer.shadow_copy(self.model)

    def on_test_begin(self, logs: Optional[MutableMapping[Text, Any]] = None):
        self.model.optimizer.swap_weights()

    def on_test_end(self, logs: Optional[MutableMapping[Text, Any]] = None):
        self.model.optimizer.swap_weights()

    def on_train_end(self, logs: Optional[MutableMapping[Text, Any]] = None):
        if self.overwrite_weights_on_train_end:
            self.model.optimizer.assign_average_vars(self.model.variables)


class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Saves and, optionally, assigns the averaged weights.
    Taken from tfa.callbacks.AverageModelCheckpoint.
    Attributes:
      update_weights: If True, assign the moving average weights to the model, and
        save them. If False, keep the old non-averaged weights, but the saved
        model uses the average weights. See `tf.keras.callbacks.ModelCheckpoint`
        for the other args.
    """

    def __init__(self,
                 update_weights: bool,
                 filepath: str,
                 monitor: str = 'val_loss',
                 verbose: int = 0,
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 mode: str = 'auto',
                 save_freq: str = 'epoch',
                 **kwargs):
        self.update_weights = update_weights
        super().__init__(filepath, monitor, verbose, save_best_only,
                         save_weights_only, mode, save_freq, **kwargs)

    def set_model(self, model):
        if not isinstance(model.optimizer, ExponentialMovingAverage):
            raise TypeError('AverageModelCheckpoint is only used when training'
                            'with MovingAverage')
        return super().set_model(model)

    def _save_model(self, epoch, logs):
        assert isinstance(self.model.optimizer, ExponentialMovingAverage)

        if self.update_weights:
            self.model.optimizer.assign_average_vars(self.model.variables)
            return super()._save_model(epoch, logs)
        else:
            # Note: `model.get_weights()` gives us the weights (non-ref)
            # whereas `model.variables` returns references to the variables.
            non_avg_weights = self.model.get_weights()
            self.model.optimizer.assign_average_vars(self.model.variables)
            # result is currently None, since `super._save_model` doesn't
            # return anything, but this may change in the future.
            result = super()._save_model(epoch, logs)
            self.model.set_weights(non_avg_weights)
            return result

    def on_epoch_end(self, epoch, logs=None):
        logging.info('checkpoint started..')
        super().on_epoch_end(epoch, logs)
        logging.info('checkpoint finished..')
