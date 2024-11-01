import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, constraints, regularizers


class DenseEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 dim,
                 vocab_size=1,
                 embeddings_initializer="uniform",
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs,
                 ):
        assert vocab_size in [0, 1]

        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = None

        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        super(DenseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.vocab_size == 1:
            self.embedding = self.add_weight(
                shape=(1, self.dim),
                initializer=self.embeddings_initializer,
                name="embeddings",
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                experimental_autocast=False,
            )
        self.built = True

    def call(self, inputs, *args, **kwargs):
        if self.embedding is None:
            return inputs

        if int(inputs.get_shape()[-1]) != 1:
            inputs = tf.expand_dims(inputs, axis=-1)

        return tf.matmul(inputs, self.embedding)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, merge="sum", dim=None, max_length=None, **kwargs):
        super().__init__(**kwargs)

        assert merge in ("sum", "concat")
        self.merge = merge
        self.dim = dim
        self.max_length = max_length

    def build(self, input_shape):
        if self.dim is not None:
            self.pos_embedding = tf.keras.layers.Embedding(
                input_shape[-2] if self.max_length is None else self.max_length,
                self.dim
            )
        else:
            self.pos_embedding = tf.keras.layers.Embedding(
                input_shape[-2] if self.max_length is None else self.max_length,
                input_shape[-1]
            )

    def compute_mask(self, inputs, mask=None):
        return mask


    def call(self, x):
        length = tf.shape(x)[1]
        pos_emb = self.pos_embedding(tf.range(length))[tf.newaxis, :, :]
        if self.merge == "sum":
            x += pos_emb
        else:
            x = tf.concat([x, pos_emb], axis=-1)
        return x