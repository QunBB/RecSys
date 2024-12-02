import tensorflow as tf
from tensorflow.keras import backend as K

from keras.src.losses import LossFunctionWrapper, losses_utils


def ranking_loss(y_true, y_pred):
    """Only compute pairs loss of positive v.s. negative samples.
    """

    z_ij = tf.reshape(y_pred, [-1, 1]) - tf.reshape(y_pred, [1, -1])

    y_true = tf.convert_to_tensor(y_true, dtype="int64")
    mask = tf.logical_and(tf.equal(tf.reshape(y_true, [-1, 1]), 1),
                          tf.equal(tf.reshape(y_true, [1, -1]), 0))
    mask = tf.cast(mask, z_ij.dtype)

    per_pair_loss = K.binary_crossentropy(tf.ones_like(z_ij, z_ij.dtype), z_ij, from_logits=True)

    num_pairs = tf.reduce_sum(mask)

    batch_size = tf.cast(tf.shape(y_pred)[0], z_ij.dtype)

    return tf.reduce_sum(per_pair_loss * mask, axis=-1) / (num_pairs + K.epsilon()) * batch_size


class RankingLoss(LossFunctionWrapper):
    def __init__(
        self,
        reduction=losses_utils.ReductionV2.AUTO,
        name="ranking_loss",
    ):
        super().__init__(
            ranking_loss,
            name=name,
            reduction=reduction,
        )
