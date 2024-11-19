import math

import tensorflow as tf
from tensorflow.keras import backend as K


class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """
        MultiHeadAttention supporting Target-aware Attention and Target-aware Representation.

    Reference:
        Temporal Interest Network for User Response Prediction

    Args:
        target_aware: Whether to use Target-aware Representation for values.
        attention_ffn: Whether to use linear projection for attention_output.
    """
    def __init__(self,
                 target_aware=False,
                 attention_ffn=True,
                 **kwargs):
        self.target_aware = target_aware
        self.attention_ffn = attention_ffn
        super(MultiHeadAttention, self).__init__(**kwargs)

    def _compute_attention(
        self, query, key, value, attention_mask=None, training=None
    ):
        """Applies Target-aware Attention and Target-aware Representation with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
            query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
            key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
            value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          attention_output: Multi-head outputs of attention computation.
          attention_scores: Multi-head attention weights.
        """

        if self.target_aware:
            # Target-aware Representation
            value = tf.einsum("...i,...i->...i", value, query)

        # Note: Applying scalar multiply at the smaller end of einsum improves
        # XLA performance, but may introduce slight numeric differences in
        # the Transformer attention head.
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )

        # `context_layer` = [B, T, N, H]
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
    ):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        # Convert RaggedTensor to Tensor.
        query_is_ragged = isinstance(query, tf.RaggedTensor)
        if query_is_ragged:
            query_lengths = query.nested_row_lengths()
            query = query.to_tensor()
        key_is_ragged = isinstance(key, tf.RaggedTensor)
        value_is_ragged = isinstance(value, tf.RaggedTensor)
        if key_is_ragged and value_is_ragged:
            # Ensure they have the same shape.
            bounding_shape = tf.math.maximum(
                key.bounding_shape(), value.bounding_shape()
            )
            key = key.to_tensor(shape=bounding_shape)
            value = value.to_tensor(shape=bounding_shape)
        elif key_is_ragged:
            key = key.to_tensor(shape=tf.shape(value))
        elif value_is_ragged:
            value = value.to_tensor(shape=tf.shape(key))

        attention_mask = self._compute_attention_mask(
            query,
            value,
            key=key,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )

        if self.attention_ffn:
            attention_output = self._output_dense(attention_output)
        else:
            attention_output = tf.reshape(attention_output, [-1, self._query_shape[1], self._num_heads * self._key_dim])

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths
            )

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, attention_ffn=True, residual=True, target_aware=False, **kwargs):
        self.mha = MultiHeadAttention(attention_ffn=attention_ffn, target_aware=target_aware, **kwargs)

        self.residual = residual
        if residual:
            self.layernorm = tf.keras.layers.LayerNormalization()
            self.add = tf.keras.layers.Add()

        super(BaseAttention, self).__init__()


class CrossAttention(BaseAttention):
    def call(self, x, context=None, training=None):
        if context is None:
            context = x
        attn_output = self.mha(
            query=x,
            key=context,
            value=context,
            training=training)

        if self.residual:
            x = self.add([x, attn_output])
            x = self.layernorm(x)
            return x
        else:
            return attn_output


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=None):
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff=None, dropout_rate=0.1,
                 attention_ffn=False, residual=True, target_aware=False):
        super().__init__()

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            attention_ffn=attention_ffn,
            residual=residual,
            target_aware=target_aware)

        self.ffn = None
        if dff is not None:
            self.ffn = FeedForward(d_model, dff)

    def call(self, x, context=None, training=None):
        x = self.cross_attention(x, context, training=training)

        if self.ffn is not None:
            x = self.ffn(x, training=training)
        return x


class Encoder(tf.keras.layers.Layer):
    """
    Args:
        d_model: Size of each attention head.
        num_layers: Number of multi-heads attention layer.
        num_heads: Number of attention heads.
        dff: Size of feed forward layer ("intermediate" layer).
        dropout_rate: Dropout probability.
        attention_ffn: Whether to use linear projection for attention_output.
        residual: Whether to add a residual for the linear projection of attention_output.
        target_aware: Whether to use Target-aware Representation for values.
    """
    def __init__(self, d_model, num_layers, num_heads, dff, dropout_rate=0.1,
                 attention_ffn=True, residual=True, target_aware=False):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.enc_layers = None

        self.kwargs = {"num_heads": num_heads, "dff": dff, "dropout_rate": dropout_rate,
                       "attention_ffn": attention_ffn, "residual": residual, "target_aware": target_aware}
        if d_model is not None:
            self.enc_layers = [
                EncoderLayer(d_model=d_model, **self.kwargs)
                for _ in range(num_layers)]

    def build(self, input_shape):
        if self.enc_layers is None:
            if isinstance(input_shape, list):
                d_model = input_shape[0][-1] // self.num_heads
            else:
                d_model = input_shape[-1] // self.num_heads
            self.enc_layers = [
                EncoderLayer(d_model=d_model, **self.kwargs)
                for _ in range(self.num_layers)]

    def call(self, x, training=None):
        if len(x) == 2:
            x, context = x
        else:
            context = x

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, context=context, training=training)

        return tf.squeeze(x, axis=-2)  # Shape `(batch_size, seq_len, d_model*num_heads)`.
