import tensorflow as tf
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from typeguard import typechecked
from typing import Optional, Union, Callable, List


class BCELoss(LossFunctionWrapper):
    """Implements the asymmetric loss function.
    (https://arxiv.org/pdf/2009.14119.pdf). Asymmetric loss is useful for
    multi-label classification when you have highly imbalanced classes.
    """

    @typechecked
    def __init__(
            self,
            from_logits: bool = True,
            label_smoothing: FloatTensorLike = 0.,
            reduction: str = tf.keras.losses.Reduction.AUTO,
            name: str = "bce_loss",
    ):
        """Initializes `AsymmetricLoss` instance.
            Args:
              from_logits: Whether to interpret `y_pred` as a tensor of
                [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
                  assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
              label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
                we compute the loss between the predicted labels and a smoothed version
                of the true labels, where the smoothing squeezes the labels towards 0.5.
                Larger values of `label_smoothing` correspond to heavier smoothing.
              axis: The axis along which to compute crossentropy (the features axis).
                Defaults to -1.
              reduction: Type of `tf.keras.losses.Reduction` to apply to
                loss. Default value is `AUTO`. `AUTO` indicates that the reduction
                option will be determined by the usage context. For almost all cases
                this defaults to `SUM_OVER_BATCH_SIZE`. When used with
                `tf.distribute.Strategy`, outside of built-in training loops such as
                `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
                will raise an error. Please see this custom training [tutorial](
                  https://www.tensorflow.org/tutorials/distribute/custom_training) for
                    more details.
              name: Name for the op.
            """

        super().__init__(
            bce_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            eps=tf.keras.backend.epsilon()
        )


@tf.function
def bce_loss(
        y_true: TensorLike,
        y_pred: TensorLike,
        from_logits: bool = True,
        label_smoothing: FloatTensorLike = 0.,
        eps: FloatTensorLike = 1e-8
) -> tf.Tensor:
    """Implements the asymmetric loss function.
    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if label_smoothing and label_smoothing < 0:
        raise ValueError("Value of label_smoothing should be greater than or equal to zero.")
    if eps and eps < 0:
        raise ValueError("Value of eps should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    if from_logits:
        y_pred = tf.sigmoid(y_pred)

    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred

    # Label Smoothing
    if label_smoothing:
        y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    # Basic CE Calculation
    los_pos = y_true * tf.math.log(tf.clip_by_value(y_pred_pos, clip_value_min=eps, clip_value_max=1.))
    los_neg = (1 - y_true) * tf.math.log(
        tf.clip_by_value(y_pred_neg, clip_value_min=eps, clip_value_max=1.))
    loss = los_pos + los_neg

    return -tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
