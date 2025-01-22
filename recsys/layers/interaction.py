from itertools import combinations

import tensorflow as tf
from tensorflow.keras.initializers import Zeros, Constant, Ones
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

from .activation import get_activation
from .core import FeedForwardLayer


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
    Partitioned Normalization for multi-domains.
    And, the implement has supported different domains samples in a mini-batch.

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

    def generate_grid_tensor(self, indices, dim):
        y = tf.range(dim)
        x_grid, y_grid = tf.meshgrid(indices, y)
        return tf.transpose(tf.stack([x_grid, y_grid], axis=-1), [1, 0, 2])

    def call(self, inputs, training=None):
        inputs, domain_index = inputs
        domain_index = tf.cast(tf.reshape(domain_index, [-1]), "int32")
        dim = inputs.shape.as_list()[-1]

        output = inputs
        # compute each domain's BN individually
        for i, bn in enumerate(self.bn_list):
            mask = tf.equal(domain_index, i)
            single_bn = self.bn_list[i](tf.boolean_mask(inputs, mask), training=training)
            single_bn = (self.global_gamma + self.domain_gamma[i]) * single_bn + (self.global_beta + self.domain_beta[i])

            # get current domain samples' indices
            indices = tf.boolean_mask(tf.range(tf.shape(inputs)[0]), mask)
            indices = self.generate_grid_tensor(indices, dim)
            output = tf.cond(
                tf.reduce_any(mask),
                lambda: tf.reshape(tf.tensor_scatter_nd_update(output, indices, single_bn), [-1, dim]),
                lambda: output
            )

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
                shape=[1, i],
                initializer=Zeros(),
                trainable=True
            ) for i in self.hidden_units
        ]
        self.domain_bias_list = [
            tf.keras.layers.Embedding(
                self.num_domain,
                output_dim=i,
                embeddings_initializer=Zeros()
            ) for i in self.hidden_units
        ]

        hidden_units = self.hidden_units.copy()
        hidden_units.insert(0, input_shape[-1])
        self.shared_weights = [
            self.add_weight(
                name=f"shared_weight_{i}",
                shape=[1, hidden_units[i], hidden_units[i+1]],
                initializer="glorot_uniform",
                regularizer=l2(self.l2_reg),
                trainable=True
            ) for i in range(len(hidden_units) - 1)
        ]
        self.domain_weights_list = [
            tf.keras.layers.Embedding(
                self.num_domain,
                hidden_units[i] * hidden_units[i + 1],
                embeddings_initializer="glorot_uniform",
                embeddings_regularizer=l2(self.l2_reg)
            ) for i in range(len(hidden_units) - 1)
        ]

    def call(self, inputs, training=None, **kwargs):
        inputs, domain_index = inputs

        output = tf.expand_dims(inputs, axis=1)
        for i in range(len(self.hidden_units)):
            domain_weight = tf.reshape(self.domain_weights_list[i](domain_index),
                                       [-1] + self.shared_weights[i].shape.as_list()[1:])
            weight = self.shared_weights[i] * domain_weight
            domain_bias = tf.reshape(self.domain_bias_list[i](domain_index), [-1] + self.shared_bias[i].shape.as_list()[1:])
            bias = self.shared_bias[i] + domain_bias

            fc = tf.matmul(output, weight) + tf.expand_dims(bias, 1)
            output = self.activation_list[i](fc, training=training)
            output = self.dropout_list[i](output, training=training)

        return tf.squeeze(output, axis=1)


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


class GatingNetwork(Layer):
    """Multi-gate Mixture-of-Experts.
    """
    def __init__(self,
                 num_experts,
                 name=None,
                 l2_reg=0.,
                 dropout=0.,
                 use_bn=False
                 ):
        self.num_experts = num_experts

        self.ffn = FeedForwardLayer([num_experts], None, l2_reg, dropout, use_bn,)

        super(GatingNetwork, self).__init__(name=name)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 3 and input_shape[1][1] == self.num_experts

    def call(self, inputs, training=None):
        # B = batch_size, N = num_experts, D = experts_dim
        # experts = [B, N, D]
        inputs, experts = inputs
        # [B, N]
        gate = self.ffn(inputs, training=training)
        gate = tf.nn.softmax(gate, axis=-1)
        # [B, 1, N]
        gate = tf.expand_dims(gate, axis=1)
        # [B, D]
        output = tf.squeeze(tf.matmul(gate, experts), axis=1)

        return output


class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model.

    Reference:
        cross_type="vector" -> Deep & Cross Network for Ad Click Predictions
        cross_type="matrix" -> DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems
    """

    def __init__(self,
                 layer_num=2,
                 cross_type="matrix",
                 low_rank_dim=None,
                 l2_reg=0,
                 **kwargs):
        assert cross_type in ("vector", "matrix")
        self.layer_num = layer_num
        self.cross_type = cross_type
        self.low_rank_dim = low_rank_dim
        self.l2_reg = l2_reg
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            dim = int(input_shape[0][-1]) * len(input_shape)
        else:
            dim = int(input_shape[-1])

        if self.cross_type == 'vector':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, 1),
                                            initializer="glorot_normal",
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        elif self.cross_type == 'matrix':
            if self.low_rank_dim is not None:
                self.kernels_1 = [self.add_weight(name='kernel_1' + str(i),
                                                  shape=(dim, self.low_rank_dim),
                                                  initializer="glorot_normal",
                                                  regularizer=l2(self.l2_reg),
                                                  trainable=True) for i in range(self.layer_num)]
                self.kernels_2 = [self.add_weight(name='kernel_2' + str(i),
                                                  shape=(self.low_rank_dim, dim),
                                                  initializer="glorot_normal",
                                                  regularizer=l2(self.l2_reg),
                                                  trainable=True) for i in range(self.layer_num)]
            else:
                self.kernels = [self.add_weight(name='kernel' + str(i),
                                                shape=(dim, dim),
                                                initializer="glorot_normal",
                                                regularizer=l2(self.l2_reg),
                                                trainable=True) for i in range(self.layer_num)]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(1, dim),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]

    def call(self, inputs, **kwargs):
        if isinstance(inputs, list):
            inputs = tf.concat(inputs, axis=-1)
        assert K.ndim(inputs) == 2

        x_0 = inputs
        x_l = inputs

        for i in range(self.layer_num):
            if self.cross_type == "vector":
                xw = tf.matmul(x_l, self.kernels[i])
                x_l = x_0 * xw + self.bias[i] + x_l
            else:
                if self.low_rank_dim is not None:
                    xw_lora = tf.matmul(x_l, self.kernels_1[i])
                    xw = tf.matmul(xw_lora, self.kernels_2[i])
                else:
                    xw = tf.matmul(x_l, self.kernels[i])
                x_l = x_0 * (xw + self.bias[i]) + x_l

        return x_l


class ProductLayer(Layer):
    """Product-based Neural Networks.

    Reference:
        Product-based Neural Networks for User Response Prediction over Multi-field Categorical Data
    """
    def __init__(self,
                 inner=True,
                 outer=False,
                 kernel_type="mat",
                 activation=None,
                 net_size=None,
                 **kwargs):
        assert inner or outer, "you must apply at least one of `inner` or `outer` product"
        if outer:
            assert kernel_type in ("net", "mat", "vec", "num"), f"Not support such kernel `{kernel_type}` in outer product"

        self.inner = inner
        self.outer = outer
        self.kernel_type = kernel_type
        self.activation = activation
        self.net_size = net_size
        super(ProductLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) > 1 and all(len(shape) == 2 for shape in input_shape), \
            "`ProductLayer` inputs must be a list of at least 2 inputs with dimensions 2"

        num_fields = len(input_shape)
        dim = input_shape[0][-1]
        num_pairs = int(num_fields * (num_fields - 1) / 2)

        if self.outer:
            # PIN
            if self.kernel_type == "net":
                if self.activation is not None:
                    self.activation = get_activation(self.activation)

                self.kernel_w = [self.add_weight("micro_net_w1", shape=[1, num_pairs, dim * 3, self.net_size],
                                                 initializer="glorot_normal"),
                                 self.add_weight("micro_net_w2", shape=[1, num_pairs, self.net_size, 1],
                                                 initializer="glorot_normal")]
                self.kernel_b = [self.add_weight("micro_net_b1", shape=[1, num_pairs, self.net_size],
                                                 initializer="glorot_normal"),
                                 self.add_weight("micro_net_b2", shape=[1, num_pairs, 1],
                                                 initializer="glorot_normal")]
            # KPNN
            elif self.kernel_type == "mat":
                self.kernel = self.add_weight("product_kernel", shape=[dim, num_pairs, dim],
                                              initializer="glorot_normal")
            elif self.kernel_type == "vec":
                self.kernel = self.add_weight("product_kernel", shape=[num_pairs, dim],
                                              initializer="glorot_normal")
            else:  # kernel_type == "num"
                self.kernel = self.add_weight("product_kernel", shape=[num_pairs, 1],
                                              initializer="glorot_normal")

    def call(self, inputs):
        outputs = []

        if self.inner:
            outputs.append(self._inner_product(inputs))
        if self.outer:
            outputs.append(self._outer_product(inputs))

        return tf.concat(outputs, axis=-1)


    def _inner_product(self, embeddings_list):
        p, q = self._get_embedding_pairs(embeddings_list)

        ip = tf.reduce_sum(p * q, [-1])

        return ip

    def _outer_product(self, embeddings_list):
        # PIN: micro network kernel
        if self.kernel_type == "net":
            # [bs, paris, 3*dim]
            p = self._get_mirco_embedding_paris(embeddings_list)
            # [bs, paris, 3*dim, 1]
            p = tf.expand_dims(p, axis=-1)

            # [bs, paris, size]
            sub_net_1 = tf.reduce_sum(
                # [bs, paris, 3*dim, size]
                tf.multiply(p, self.kernel_w[0]),
                axis=2
            ) + self.kernel_b[0]

            if self.activation is not None:
                sub_net_1 = self.activation(sub_net_1)

            # [bs, paris, size, 1]
            sub_net_1 = tf.expand_dims(sub_net_1, axis=-1)
            # [bs, paris, 1]
            sub_net_2 = tf.reduce_sum(
                # [bs, paris, size, 1]
                tf.multiply(sub_net_1, self.kernel_w[1]),
                axis=2
            ) + self.kernel_b[1]

            return tf.squeeze(sub_net_2, axis=-1)

        # KPNN: linear kernel
        p, q = self._get_embedding_pairs(embeddings_list)

        if self.kernel_type == "mat":
            # batch * 1 * pair * k
            p = tf.expand_dims(p, 1)
            # batch * pair
            kp = tf.reduce_sum(
                # batch * pair * k
                tf.multiply(
                    # batch * pair * k
                    tf.transpose(
                        # batch * k * pair
                        tf.reduce_sum(
                            # batch * k * pair * k
                            tf.multiply(
                                p, self.kernel),
                            -1),
                        [0, 2, 1]),
                    q),
                -1)
        else:
            # 1 * pair * (k or 1)
            k = tf.expand_dims(self.kernel, 0)
            # batch * pair
            kp = tf.reduce_sum(p * q * k, -1)

        return kp

    def _get_embedding_pairs(self, embeddings_list):
        num_fields = len(embeddings_list)

        p = []
        q = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                p.append(embeddings_list[i])
                q.append(embeddings_list[j])
        # [bs, paris, dim]
        p = tf.stack(p, axis=1)
        q = tf.stack(q, axis=1)

        return p, q

    def _get_mirco_embedding_paris(self, embeddings_list):
        num_fields = len(embeddings_list)
        p = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                p.append(tf.concat([embeddings_list[i], embeddings_list[j], embeddings_list[i] * embeddings_list[j]], axis=-1))

        # [bs, paris, 3*dim]
        p = tf.stack(p, axis=1)
        return p
