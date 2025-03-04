from itertools import chain
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.regularizers import l2

from recsys.feature import Field
from recsys.layers.embedding import DenseEmbedding


def build_feature_embeddings(
        fields: List[Field],
        inputs_dict: Optional[dict] = None,
        disable: Optional[Union[List[str], str]] = "task",
        prefix: str = "embedding/",
        return_list: bool = False,
        global_emb_table_dict: Optional[Dict[str, tf.keras.layers.Layer]] = None,
) -> Tuple[Dict[str, Input], Dict[str, Union[Dict[str, tf.Tensor], tf.Tensor]]]:
    if global_emb_table_dict is not None:
        emb_table_dict = global_emb_table_dict
    else:
        emb_table_dict = {}

    history_emb = {f.emb for f in fields if f.belong == "history"}

    # create embedding table
    for field in fields:

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

    # for the same fields with multiple embedding tables
    if inputs_dict is None:
        inputs_dict = OrderedDict()
    embeddings_dict = OrderedDict()
    for field in fields:
        # create fields input and compute their embeddings
        name = field.name
        emb_name = field.emb
        dtype = field.belong
        group = field.group

        if name not in inputs_dict:
            if field.belong == "history" or field.vocabulary_size == 0:
                inputs_dict[name] = Input(shape=(field.length,), name=name, dtype=field.dtype)
            else:
                inputs_dict[name] = Input(shape=(), name=name, dtype=field.dtype)

        embeddings_dict.setdefault(dtype, {})
        embeddings_dict[dtype].setdefault(group, [])

        embeddings_dict[dtype][group].append(emb_table_dict[emb_name](inputs_dict[name]))

    # prohibit some special fields in the other model
    if disable:
        if not isinstance(disable, list):
            disable = [disable]
        for belong in disable:
            assert belong not in embeddings_dict, f"Current model doesn't support such features that belong to \"{belong}\""

    if not return_list:
        for dtype in embeddings_dict:
            if len(embeddings_dict[dtype]) <= 1:
                embeddings_dict[dtype] = tf.keras.layers.Concatenate(axis=-1)(list(chain.from_iterable(embeddings_dict[dtype].values())))
            else:
                # each group's embeddings
                for group in embeddings_dict[dtype]:
                    embeddings_dict[dtype][group] = tf.keras.layers.Concatenate(axis=-1)(embeddings_dict[dtype][group])

    return inputs_dict, embeddings_dict


def concatenate(
        embeddings_dict,
        names: Optional[List[str]] = None
) -> tf.Tensor:
    assert len(embeddings_dict) > 0, "embeddings_dict is empty"

    if names is None:
        return tf.concat(list(embeddings_dict.values()), axis=-1)

    embeddings = []
    for name in names:
        embeddings.append(embeddings_dict[name])

    assert len(embeddings) > 0, f"At least one must exist: {names}"

    return tf.concat(embeddings, axis=-1)
