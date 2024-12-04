import tensorflow as tf
from tensorflow.keras.initializers import Zeros, Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

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


class PartitionedNormalization(Layer):
    """
    Partitioned Normalization for multi-domains

    Reference:
        One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
    """
    def __init__(self,
                 num_domain,
                 name=None,
                 **kwargs):

        self.bn_list = [tf.keras.layers.BatchNormalization(center=False, scale=False, name=f"bn_{i}", **kwargs) for i in range(num_domain)]

        super(PartitionedNormalization, self).__init__(name=name)

    def build(self, input_shape):
        assert len(input_shape) == 2 and len(input_shape[1]) <= 2
        dim = input_shape[0][-1]

        self.global_gamma = self.add_weight(
            name="global_gamma",
            shape=[dim],
            initializer=Constant(0.5),
            trainable=True
        )
        self.global_beta = self.add_weight(
            name="global_beta",
            shape=[dim],
            initializer=Zeros(),
            trainable=True
        )
        self.domain_gamma = self.add_weight(
                name="domain_gamma",
                shape=[len(self.bn_list), dim],
                initializer=Constant(0.5),
                trainable=True
            )
        self.domain_beta = self.add_weight(
                name="domain_beta",
                shape=[len(self.bn_list), dim],
                initializer=Zeros(),
                trainable=True
            )

    def compute_bn(self, idx, x, training):
        return tf.case([
            (tf.equal(idx, i), lambda: self.bn_list[i](x, training=training)) for i in range(len(self.bn_list))
        ])

    def call(self, inputs, training=None):
        inputs, domain_index = inputs

        # take the first sample's domain index as current batch's domain index
        domain_index = tf.reshape(domain_index, [-1])[0]

        output = self.compute_bn(domain_index, inputs, training=training)

        output = (self.global_gamma + self.domain_gamma[domain_index]) * output + (self.global_beta + self.domain_beta[domain_index])

        return output


class StarTopologyFCN(Layer):
    """
    Reference:
        One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
    """
    def __init__(self,
                 num_domain,
                 hidden_units,
                 activation="relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        self.num_domain = num_domain
        self.hidden_units = hidden_units
        self.activation_list = [get_activation(activation) for _ in hidden_units]
        self.dropout_list = [tf.keras.layers.Dropout(dropout) for _ in hidden_units]
        self.l2_reg = l2_reg
        super(StarTopologyFCN, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_shape = input_shape[0]

        self.shared_bias = [
            self.add_weight(
                name=f"shared_bias_{i}",
                shape=[i],
                initializer=Zeros(),
                trainable=True
            ) for i in self.hidden_units
        ]
        self.domain_bias = [
            self.add_weight(
                name=f"domain_bias_{i}",
                shape=[self.num_domain, i],
                initializer=Zeros(),
                trainable=True
            ) for i in self.hidden_units
        ]

        hidden_units = self.hidden_units.copy()
        hidden_units.insert(0, input_shape[-1])
        self.shared_weights = [
            self.add_weight(
                name=f"shared_weight_{i}",
                shape=[hidden_units[i], hidden_units[i+1]],
                initializer="glorot_uniform",
                regularizer=l2(self.l2_reg),
                trainable=True
            ) for i in range(len(hidden_units) - 1)
        ]
        self.domain_weights = [
            self.add_weight(
                name=f"domain_weight_{i}",
                shape=[self.num_domain, hidden_units[i], hidden_units[i + 1]],
                initializer="glorot_uniform",
                regularizer=l2(self.l2_reg),
                trainable=True
            ) for i in range(len(hidden_units) - 1)
        ]

    def call(self, inputs, training=None, **kwargs):
        inputs, domain_index = inputs

        # take the first sample's domain index as current batch's domain index
        domain_index = int(tf.reshape(domain_index, [-1])[0])

        output = inputs
        for i in range(len(self.hidden_units)):
            weight = self.shared_weights[i] * self.domain_weights[i][domain_index]
            bias = self.shared_bias[i] + self.domain_bias[i][domain_index]

            fc = tf.matmul(output, weight) + bias
            output = self.activation_list[i](fc, training=training)
            output = self.dropout_list[i](output, training=training)

        return output


class MetaUnit(Layer):
    """
    Reference:
        Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
    """
    def __init__(self,
                 num_layer,
                 activation="leaky_relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        self.num_layer = num_layer
        self.l2_reg = l2_reg

        self.weights_dense = []
        self.bias_dense = []
        self.activation_list = [get_activation(activation) for _ in range(num_layer)]
        self.dropout_list = [tf.keras.layers.Dropout(dropout) for _ in range(num_layer)]

        super(MetaUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_size = input_shape[0][-1]
        self.input_size = input_size

        for i in range(self.num_layer):
            self.weights_dense.append(
                tf.keras.layers.Dense(input_size*input_size, kernel_regularizer=l2(self.l2_reg))
            )
            self.bias_dense.append(
                tf.keras.layers.Dense(input_size, kernel_regularizer=l2(self.l2_reg))
            )

    def call(self, inputs, training=None, **kwargs):
        inputs, scenario_views = inputs

        # [bs, 1, dim]
        squeeze = False
        if K.ndim(inputs) == 2:
            squeeze = True
            inputs = tf.expand_dims(inputs, axis=1)

        output = inputs
        for i in range(self.num_layer):
            # [bs, dim*dim]
            w = self.weights_dense[i](scenario_views)
            b = self.bias_dense[i](scenario_views)

            # [bs, dim, dim]
            w = tf.reshape(w, [-1, self.input_size, self.input_size])
            b = tf.expand_dims(b, axis=1)

            # [bs, 1, dim] * [bs, dim, dim] = [bs, 1, dim]
            fc = tf.matmul(output, w) + b

            output = self.activation_list[i](fc, training=training)

            output = self.dropout_list[i](output, training=training)

        # [bs, dim]
        if squeeze:
            return tf.squeeze(output, axis=1)
        else:
            return output


class MetaAttention(Layer):
    """
    Reference:
        Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
    """
    def __init__(self,
                 meta_unit=None,
                 num_layer=3,
                 activation="leaky_relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        if meta_unit is not None:
            self.meta_unit = meta_unit
        else:
            self.meta_unit = MetaUnit(num_layer, activation, dropout, l2_reg)
        self.dense = tf.keras.layers.Dense(1)

        super(MetaAttention, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        expert_views, task_views, scenario_views = inputs
        task_views = tf.repeat(tf.expand_dims(task_views, axis=1), tf.shape(expert_views)[1], axis=1)
        # [bs, num_experts, dim]
        meta_unit_output = self.meta_unit([tf.concat([expert_views, task_views], axis=-1), scenario_views], training=training)
        # [bs, num_experts, 1]
        score = self.dense(meta_unit_output)
        # [bs, dim]
        output = tf.reduce_sum(expert_views * score, axis=1)

        return output


class MetaTower(Layer):
    """
    Reference:
        Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling
    """
    def __init__(self,
                 meta_unit=None,
                 num_layer=3,
                 meta_unit_depth=3,
                 activation="leaky_relu",
                 dropout=0.,
                 l2_reg=0.,
                 **kwargs):
        if meta_unit is not None:
            self.layers = [meta_unit] * num_layer  # all `meta_unit` in the lise will be the same object
        else:
            self.layers = [MetaUnit(meta_unit_depth, activation, dropout, l2_reg) for _ in range(num_layer)]
        self.activation_list = [get_activation(activation) for _ in range(num_layer)]
        self.dropout_list = [tf.keras.layers.Dropout(dropout) for _ in range(num_layer)]

        super(MetaTower, self).__init__(**kwargs)

    def call(self, inputs, training=None, **kwargs):
        inputs, scenario_views = inputs

        output = inputs
        for i in range(len(self.layers)):
            output = self.layers[i]([output, scenario_views], training=training)
            output = self.activation_list[i](output, training=training)
            output = self.dropout_list[i](output, training=training)

        return output