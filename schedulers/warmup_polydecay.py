import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from typing import Optional, Mapping, Any


class WarmupPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses a warmup and polynomial decay schedule."""

    def __init__(self,
                 base_lr: float,
                 decay_steps: int,
                 end_learning_rate: Optional[float] = 0.,
                 power: Optional[float] = 1.0,
                 warmup_steps: Optional[int] = 0,
                 init_lr: Optional[float] = 0.,
                 name: Optional[str] = None):
        """Add warmup decay to a learning rate schedule.

        Args:
          base_lr: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The initial learning rate.
          decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
            Must be positive.  The learning rate reaches an `end_learning_rate`
            in the given `decay_steps`.
          end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The minimal end learning rate.
          power: A scalar `float32` or `float64` `Tensor` or a
            Python number.  The power of the polynomial. Defaults to linear, 1.0.
          name: String.  Optional name of the operation. Defaults to
            'PolynomialDecay'.
          warmup_steps: number of warmup steps
          init_lr: an optional field for the initial warmup learning rate. if it
            is not specified, lr starts from 0.
        """
        super(WarmupPolynomialDecay, self).__init__()
        self.base_lr = base_lr
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.name = name

    def __call__(self, step: int):
        with ops.name_scope_v2(self.name or "WarmupPolynomialDecay") as name:
            base_lr = ops.convert_to_tensor_v2_with_dispatch(
                self.base_lr, name="initial_learning_rate")
            dtype = base_lr.dtype
            end_learning_rate = math_ops.cast(self.end_learning_rate, dtype)
            power = math_ops.cast(self.power, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            decay_steps_recomp = math_ops.cast(self.decay_steps, dtype)

            warmup_steps = tf.cast(self.warmup_steps, dtype)
            base_lr = base_lr - self.init_lr
            base_lr = math_ops.maximum(base_lr, 0)
            warmup_lr = base_lr * global_step_recomp / warmup_steps + self.init_lr

            # Make sure that the global_step used is not bigger than decay_steps.
            global_step_recomp = math_ops.minimum(global_step_recomp,
                                                  decay_steps_recomp)
            lr = math_ops.divide(global_step_recomp - warmup_steps,
                                 decay_steps_recomp - warmup_steps)
            lr = math_ops.add(
                math_ops.multiply(base_lr - end_learning_rate,
                                  math_ops.pow(1 - lr, power)),
                end_learning_rate,
                name=name)

            lr = tf.cond(global_step_recomp < warmup_steps, lambda: warmup_lr,
                         lambda: lr)
            return lr

    def get_config(self) -> Mapping[str, Any]:
        return {
            "base_lr": self.base_lr,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "warmup_steps": self.warmup_steps,
            "init_lr": self.init_lr,
            "name": self.name
        }
