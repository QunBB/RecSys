import tensorflow as tf
from tensorflow.keras.initializers import Zeros


class Dice(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.bn = tf.keras.layers.BatchNormalization(center=False, scale=False)
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.alphas = self.add_weight(
            "alphas",
            shape=[input_dim],
            initializer=Zeros(),
            dtype=self.dtype,
            trainable=True,
        )
        self.beta = self.add_weight(
            "beta",
            shape=[input_dim],
            initializer=Zeros(),
            dtype=self.dtype,
            trainable=True,
        )

    def call(self, inputs, training=None):
        x_normed = self.bn(inputs, training=training)
        x_p = tf.nn.sigmoid(self.beta * x_normed)

        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs


class PRelu(tf.keras.layers.Layer):
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.alphas = self.add_weight(
            "alphas",
            shape=[input_dim],
            initializer=Zeros(),
            dtype=self.dtype,
            trainable=True,
        )

    def call(self, inputs, **kwargs):
        pos = tf.nn.relu(inputs)
        neg = self.alphas * (inputs - tf.abs(inputs)) * 0.5

        return pos + neg


class Activation(tf.keras.layers.Activation):

    def call(self, inputs, **kwargs):

        return self.activation(inputs)


def get_activation(name):
    if not name:
        return Activation("linear")

    if name == "dice":
        return Dice()

    if name == "prelu":
        return PRelu()

    return Activation(name)
