import tensorflow as tf
from tensorflow.keras.initializers import Zeros, Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2

from recsys.layers.activation import get_activation


class GateNU(Layer):
    """Gate Neural Unit

    Reference:
        PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information
    """
    def __init__(self,
                 hidden_units,
                 gamma=2.,
                 l2_reg=0.):
        assert len(hidden_units) == 2
        self.gamma = gamma

        self.dense_layers = [
            tf.keras.layers.Dense(hidden_units[0], activation="relu", kernel_regularizer=l2(l2_reg)),
            tf.keras.layers.Dense(hidden_units[1], activation="sigmoid", kernel_regularizer=l2(l2_reg))
        ]

        super(GateNU, self).__init__()

    def call(self, inputs):
        output = self.dense_layers[0](inputs)

        output = self.gamma * self.dense_layers[1](output)

        return output


class EPNet(Layer):
    """Embedding Personalized Network(EPNet)

    Reference:
        PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information
    """
    def __init__(self,
                 l2_reg=0.,
                 **kwargs):
        self.l2_reg = l2_reg

        self.gate_nu = None

        super(EPNet, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        shape1, shape2 = input_shape

        self.gate_nu = GateNU(hidden_units=[shape2[-1], shape2[-1]], l2_reg=self.l2_reg)

    def call(self, inputs, *args, **kwargs):
        domain, emb = inputs

        return self.gate_nu(tf.concat([domain, tf.stop_gradient(emb)], axis=-1)) * emb


class PPNet(Layer):
    """Parameter Personalized Network(PPNet)

    Reference:
        PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information
    """
    def __init__(self,
                 multiples,
                 hidden_units,
                 activation,
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        self.hidden_units = hidden_units
        self.l2_reg = l2_reg

        self.multiples = multiples

        self.dense_layers = []
        self.dropout_layers = []
        for i in range(multiples):
            self.dense_layers.append(
                [tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)) for units in hidden_units]
            )
            self.dropout_layers.append(
                [tf.keras.layers.Dropout(dropout) for _ in hidden_units]
            )
        self.gate_nu = []

        super(PPNet, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gate_nu = [GateNU([i*self.multiples, i*self.multiples], l2_reg=self.l2_reg
                               ) for i in self.hidden_units]

    def call(self, inputs, training=None, **kwargs):
        inputs, persona = inputs

        gate_list = []
        for i in range(len(self.hidden_units)):
            gate = self.gate_nu[i](tf.concat([persona, tf.stop_gradient(inputs)], axis=-1))
            gate = tf.split(gate, self.multiples, axis=1)
            gate_list.append(gate)

        output_list = []

        for n in range(self.multiples):
            output = inputs

            for i in range(len(self.hidden_units)):
                fc = self.dense_layers[n][i](output)

                output = gate_list[i][n] * fc

                output = self.dropout_layers[n][i](output, training=training)

            output_list.append(output)

        return output_list