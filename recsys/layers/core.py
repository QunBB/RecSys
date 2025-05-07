from typing import List, Callable, Union, Optional

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Zeros

from .activation import get_activation


class PredictLayer(tf.keras.layers.Layer):
    def __init__(self,
                 task: str,
                 num_classes: int = 1,
                 as_logit: bool = False,
                 use_bias: bool = True,
                 **kwargs):
        """
        Args:
            num_classes: Number of classes when task is "multiclass"
            as_logit: Whether to return origin logit, otherwise probability
            use_bias: Whether to add bias
        """
        assert task in ["binary", "regression", "multiclass"], f"Invalid task: \"{task}\""

        if task != "multiclass":
            output_dim = 1
        else:
            assert num_classes > 1
            output_dim = num_classes

        self.output_dim = output_dim
        self.use_bias = use_bias

        self.activation = None
        if not as_logit:
            if task == "binary":
                self.activation = tf.nn.sigmoid
            elif task == "multiclass":
                self.activation = tf.nn.softmax

        self.dense = None

        super(PredictLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[-1] != self.output_dim:
            self.dense = tf.keras.layers.Dense(self.output_dim, use_bias=self.use_bias)
        elif self.use_bias:
            self.bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")
            self.dense = lambda x: x + self.bias

    def call(self, inputs, *args, **kwargs):
        output = inputs

        if self.dense is not None:
            output = self.dense(output)

        if self.activation is not None:
            output = self.activation(output)

        return output


class Identity(tf.keras.layers.Layer):

    def call(self, inputs, *args, **kwargs):
        return inputs


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self,
                 hidden_units: Union[int, List[int]],
                 activation: Optional[Union[str, Callable]] = "relu",
                 l2_reg: float = 0.,
                 dropout_rate: float = 0.,
                 use_bn: bool = False,
                 **kwargs
                 ):

        if not isinstance(hidden_units, list):
            hidden_units = [hidden_units]
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


def predict(task, inputs):
    if task.return_logit:
        logit = PredictLayer(task.belong, task.num_classes, as_logit=True, name=f"dnn/{task.name}_logit_layer")(inputs)
        prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/{task.name}_predict_layer")(logit)
        outputs = [Identity(name=task.name)(prediction), Identity(name="logit")(logit)]
    else:
        outputs = PredictLayer(task.belong, task.num_classes, name=f"dnn/{task.name}_predict_layer")(inputs)
        outputs = Identity(name=task.name)(outputs)

    return outputs