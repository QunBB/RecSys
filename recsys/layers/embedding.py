from typing import Optional

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
            if K.ndim(inputs) == 1:
                inputs = tf.expand_dims(inputs, axis=1)
            return inputs

        if inputs.get_shape()[-1] != 1:
            inputs = tf.expand_dims(inputs, axis=-1)

        return tf.matmul(inputs, self.embedding)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 merge: str = "sum",
                 dim: Optional[int] = None,
                 max_length: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)

        assert merge in ("sum", "concat")
        self.merge = merge
        self.dim = dim
        self.max_length = max_length

    def build(self, input_shape):
        assert len(input_shape) == 3

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


    def call(self, x, start=0):
        length = tf.shape(x)[1]
        pos_emb = self.pos_embedding(tf.ones_like(x[:, :, 0], dtype="int32") * tf.range(start, length+start))
        if self.merge == "sum":
            x += pos_emb
        else:
            x = tf.concat([x, pos_emb], axis=-1)
        return x


class MaskEmbedding(tf.keras.layers.Layer):
    """Feature Mask.

    Reference:
        AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System
    """
    def __init__(self,
                 num_sample: int,
                 min_probability: float = 0.1,
                 max_probability: float = 0.5,
                 emd_dim: Optional[int] = None,
                 embeddings_initializer="uniform",
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        self.num_sample = num_sample
        self.min_probability = min_probability
        self.max_probability = max_probability

        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.embeddings_constraint = embeddings_constraint

        super(MaskEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            assert len(input_shape) == 3
            _, num_features, dim = input_shape.shape.as_list()
        else:
            num_features = len(input_shape)
            dim = input_shape[0][-1]

        self.mask_embeddings = self.add_weight(
                shape=(1, 1, num_features, dim),
                initializer=self.embeddings_initializer,
                name="embeddings",
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
                experimental_autocast=False,
            )

    def call(self, inputs, *args, **kwargs):
        is_list = False
        if isinstance(inputs, list):
            inputs = tf.stack(inputs, axis=1)
            is_list = True

        num_features = inputs.shape[1]
        dim = inputs.shape[2]
        probability = tf.random.uniform([1, self.num_sample, num_features, 1], minval=self.min_probability, maxval=self.max_probability)
        mask = tf.cast(tf.random.uniform(probability.shape) < probability, inputs.dtype)

        inputs = tf.expand_dims(inputs, axis=1)  # [batch_size, 1, num_features, dim]

        augments = inputs * (1 - mask) + self.mask_embeddings * mask  # [batch_size, num_sample, num_features, dim]
        augments = tf.reshape(augments, [-1, num_features, dim])  # [batch_size*num_sample, num_features, dim]

        if is_list:
            tensor_list = tf.split(augments, num_or_size_splits=num_features, axis=1)
            augments = [tf.squeeze(tensor, axis=1) for tensor in tensor_list]

        return augments
