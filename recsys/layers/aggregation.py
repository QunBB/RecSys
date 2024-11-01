import tensorflow as tf
from tensorflow.keras import backend as K

from recsys.layers.core import FeedForwardLayer
from recsys.layers.transformer import Encoder
from recsys.layers.embedding import PositionalEmbedding


def get_agg_layer(pooling,
                  **kwargs):
    layer = {
        "mean_pooling": MeanPoolingLayer,
        "sum_pooling": SumPoolingLayer,
        "attention": AttentionLayer,
        "transformer": TransformerLayer
    }[pooling]

    return layer(**kwargs)


class MeanPoolingLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        if isinstance(input_shape[0], list):
            for i in range(len(input_shape) - 1):
                assert input_shape[i][-1] == input_shape[i+1][-1] and K.ndim(input_shape[i][-1]) == K.ndim(input_shape[i+1][-1])

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            inputs = inputs[-1]
            mask = mask[-1]

        if K.ndim(inputs) == 2:
            return inputs

        assert K.ndim(inputs) == 3

        mask = tf.cast(tf.expand_dims(mask, 2), inputs.dtype)

        return tf.reduce_sum(inputs * mask, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-9)


class SumPoolingLayer(tf.keras.layers.Layer):

    def build(self, input_shape):
        if isinstance(input_shape[0], list):
            for i in range(len(input_shape) - 1):
                assert K.ndim(input_shape[i][-1]) == K.ndim(input_shape[i+1][-1])

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            inputs = inputs[-1]
            mask = mask[-1]

        if K.ndim(inputs) == 2:
            return inputs

        assert K.ndim(inputs) == 3

        mask = tf.cast(tf.expand_dims(mask, 2), inputs.dtype)

        return tf.reduce_sum(inputs * mask, axis=1)


class AttentionLayer(tf.keras.layers.Layer):
    """ DIN Attention:

    Reference:
        Deep Interest Network for Click-Through Rate Prediction
    """
    def __init__(self,
                 ffn_hidden_units=[80, 40],
                 ffn_activation="dice",
                 query_ffn=False,
                 query_activation="prelu",
                 **kwargs):
        self.query_ffn = query_ffn
        self.query_activation = query_activation
        self.query_ffn_layer = None
        
        self.ffn_layer = FeedForwardLayer(ffn_hidden_units, ffn_activation)
        self.dense = tf.keras.layers.Dense(1)
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 2 and len(input_shape[0]) == 2 and len(input_shape[1]) == 3
        if self.query_ffn:
            self.query_ffn_layer = tf.keras.layers.Dense(input_shape[1][-1], self.query_activation)

    def call(self, inputs, mask=None, **kwargs):
        # query: [B, H]
        # keys: [B, L, H]
        query, keys = inputs
        if self.query_ffn_layer is not None:
            query = self.query_ffn_layer(query)

        length = tf.shape(keys)[-2]
        query = tf.expand_dims(query, axis=1)
        att_inputs = tf.concat([tf.tile(query, [1, length, 1]),
                                keys, query - keys, query * keys], axis=-1)
        hidden_layer = self.ffn_layer(att_inputs)
        scores = self.dense(hidden_layer)
        scores = tf.reshape(scores, [-1, 1, length])

        if mask is not None:
            mask = mask[1]

            mask = tf.expand_dims(mask, axis=1)

            scores += (1.0 - tf.cast(mask, keys.dtype)) * (-1e9)

        scores /= keys.get_shape().as_list()[-1] ** 0.5
        scores = tf.nn.softmax(scores)

        att_outputs = tf.matmul(scores, keys)

        return tf.squeeze(att_outputs, axis=1)


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, num_layers=1, d_model=None, num_heads=2, dff=None, dropout_rate=0.1,
                 max_len=None, pos_emb="sum", **kwargs):
        self.pos_emb_layer = PositionalEmbedding(merge=pos_emb, max_length=max_len)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, dropout_rate)
        super(TransformerLayer, self).__init__(**kwargs)

    def call(self, inputs):
        query, keys = inputs

        if K.ndim(query) != K.ndim(keys):
            query = tf.expand_dims(query, axis=-2)

        keys = self.pos_emb_layer(keys)

        outputs = self.encoder([query, keys])

        return outputs
