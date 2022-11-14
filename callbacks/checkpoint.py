import logging
import os

import tensorflow as tf


class ReusableBackupAndRestore(tf.keras.callbacks.experimental.BackupAndRestore):
    """A BackupAndRestore callback that can be used across multiple model.fit()s."""

    def on_train_end(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)


class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super().__init__()
        self._checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, epoch, logs=None):
        # https://keras.io/guides/serialization_and_saving/#tf-checkpoint-format
        self._checkpoint_manager.save()
