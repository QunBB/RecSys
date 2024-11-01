import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context=None):
        if context is None:
            context = x
        attn_output = self.mha(
            query=x,
            key=context,
            value=context)

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


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

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff=None, dropout_rate=0.1):
        super().__init__()

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = None
        if dff is not None:
            self.ffn = FeedForward(d_model, dff)

    def call(self, x, context=None):
        x = self.cross_attention(x, context)
        if self.ffn is not None:
            x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.enc_layers = None
        if d_model is not None:
            self.enc_layers = [
                EncoderLayer(d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             dropout_rate=dropout_rate)
                for _ in range(num_layers)]

    def build(self, input_shape):
        if self.enc_layers is None:
            if isinstance(input_shape, list):
                d_model = input_shape[0][-1]
            else:
                d_model = input_shape[-1]
            self.enc_layers = [
                EncoderLayer(d_model=d_model,
                             num_heads=self.num_heads,
                             dff=self.dff,
                             dropout_rate=self.dropout_rate)
                for _ in range(self.num_layers)]

    def call(self, x):
        if len(x) == 2:
            x, context = x
        else:
            context = x

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, context=context)

        return tf.squeeze(x, axis=-2)  # Shape `(batch_size, seq_len, d_model)`.
