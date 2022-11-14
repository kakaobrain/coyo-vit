import tensorflow as tf
from typing import Optional, Mapping, Any


# https://github.com/tensorflow/models/blob/master/official/vision/image_classification/learning_rate.py
class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A wrapper for LearningRateSchedule that includes warmup steps."""

    def __init__(self,
                 lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
                 warmup_steps: int,
                 init_lr: Optional[float] = 0.,
                 warmup_lr: Optional[float] = None):
        """Add warmup decay to a learning rate schedule.

        Args:
          lr_schedule: base learning rate scheduler
          warmup_steps: number of warmup steps
          init_lr: an optional field for the initial warmup learning rate. if it
          is not specified, lr starts from 0.
          warmup_lr: an optional field for the final warmup learning rate. This
            should be provided if the base `lr_schedule` does not contain this
            field.
        """
        super(WarmupDecaySchedule, self).__init__()
        self._lr_schedule = lr_schedule
        self._warmup_steps = warmup_steps
        self._init_lr = init_lr
        self._warmup_lr = warmup_lr

    def __call__(self, step: int):
        lr = self._lr_schedule(step)
        if self._warmup_steps:
            base_lr = self._warmup_lr - self._init_lr
            global_step_recomp = tf.cast(step, tf.float32)
            warmup_steps = tf.cast(self._warmup_steps, tf.float32)
            warmup_lr = base_lr * global_step_recomp / warmup_steps + self._init_lr
            lr = tf.cond(global_step_recomp < warmup_steps, lambda: warmup_lr,
                         lambda: lr)
        return lr

    def get_config(self) -> Mapping[str, Any]:
        config = self._lr_schedule.get_config()
        config.update({
            "warmup_steps": self._warmup_steps,
            "warmup_lr": self._warmup_lr,
        })
        return config
