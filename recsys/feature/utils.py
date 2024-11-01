from itertools import chain
from collections import OrderedDict
from typing import List, Tuple, Dict, Union

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2

from recsys.feature import Field
from recsys.layers.embedding import DenseEmbedding


def build_feature_embeddings(
        fields: List[Field],
        prefix: str = "embedding_",
        return_list: bool = False
) -> Tuple[Dict[str, Input], Union[Dict[str, Dict[str, tf.Tensor]], Dict[str, tf.Tensor]]]:
    emb_table_dict = {}

    history_emb = {f.emb or f.name for f in fields if f.belong == "history"}

    for field in fields:
        if not field.emb:
            field.emb = field.name

        if field.emb not in emb_table_dict:
            if field.vocabulary_size > 1:
                mask_zero = False
                if field.emb in history_emb:
                    mask_zero = True
                emb_table_dict[field.emb] = tf.keras.layers.Embedding(
                    field.vocabulary_size, field.dim, name=prefix + field.emb, mask_zero=mask_zero,
                    embeddings_initializer=field.initializer, embeddings_regularizer=l2(field.l2_reg)
                )
            else:
                emb_table_dict[field.emb] = DenseEmbedding(
                    field.dim, field.vocabulary_size, name=prefix + field.emb,
                    embeddings_initializer=field.initializer, embeddings_regularizer=l2(field.l2_reg)
                )

    embeddings_dict = OrderedDict()
    inputs_dict = OrderedDict()
    for field in fields:
        name = field.name
        emb_name = field.emb
        dtype = field.belong
        group = field.group

        if field.belong == "history" or field.vocabulary_size == 0:
            inputs_dict[name] = Input(shape=(field.length,), name=name, dtype=field.dtype)
        else:
            inputs_dict[name] = Input(shape=(), name=name, dtype=field.dtype)

        embeddings_dict.setdefault(dtype, {})
        embeddings_dict[dtype].setdefault(group, [])

        embeddings_dict[dtype][group].append(emb_table_dict[emb_name](inputs_dict[name]))

    if not return_list:
        for dtype in embeddings_dict:
            if len(embeddings_dict[dtype]) <= 1:
                embeddings_dict[dtype] = tf.keras.layers.Concatenate(axis=-1)(list(chain.from_iterable(embeddings_dict[dtype].values())))
            else:
                # each group's embeddings
                for group in embeddings_dict[dtype]:
                    embeddings_dict[dtype][group] = tf.keras.layers.Concatenate(axis=-1)(embeddings_dict[dtype][group])

    return inputs_dict, embeddings_dict
