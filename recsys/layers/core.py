from typing import List, Callable, Union

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from recsys.layers.activation import get_activation


class PredictLayer(tf.keras.layers.Layer):
    def __init__(self,
                 task: str,
                 num_classes: int = 1,
                 use_bias: bool = True,
                 **kwargs):
        assert task in ["binary", "regression", "multiclass"], f"Invalid task: {task}"

        if task != "multiclass":
            output_dim = 1
        else:
            assert num_classes > 1
            output_dim = num_classes

        self.task = task
        self.dense = tf.keras.layers.Dense(output_dim, use_bias=use_bias)
        super(PredictLayer, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        output = self.dense(inputs)

        if self.task == "binary":
            output = tf.nn.sigmoid(output)
        elif self.task == "multiclass":
            output = tf.nn.softmax(output, axis=-1)

        return output


class Identity(tf.keras.layers.Layer):

    def call(self, inputs, *args, **kwargs):
        return inputs


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_units: List[int],
                 activation: Union[str, Callable] = "relu",
                 l2_reg: float = 0.,
                 dropout_rate: float = 0.,
                 use_bn: bool = False,
                 **kwargs
                 ):

        self.dense_layers = [tf.keras.layers.Dense(i, kernel_regularizer=l2(l2_reg)) for i in hidden_units]

        self.activations = [get_activation(activation) for _ in hidden_units]

        self.dropout_layers = [tf.keras.layers.Dropout(dropout_rate) for _ in hidden_units]
        if use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in hidden_units]

        self.use_bn = use_bn
        super(FeedForwardLayer, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        output = inputs

        for i in range(len(self.dense_layers)):
            fc = self.dense_layers[i](output)

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activations[i](fc, training=training)

            fc = self.dropout_layers[i](fc, training=training)

            output = fc

        return output
