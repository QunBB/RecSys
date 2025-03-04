"""
DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

IJCAI'2017：https://arxiv.org/abs/1703.04247
"""
from typing import List, Union, Callable
from itertools import chain

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.interaction import FM
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def deepfm(
        dnn_fields: List[Field],
        fm_fields: List[Field],
        task: Task = Task(name="ctr", belong="binary"),
        hidden_units: List[int] = [100, 64],
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        use_bn: bool = False,
        history_agg: str = "attention",
        agg_kwargs: dict = {}
) -> tf.keras.Model:
    """

    :param dnn_fields: the fields of dnn part
    :param fm_fields: the fields of fm part
    :param task: which task
    :param hidden_units: hidden units in MLPs
    :param activation: activation in MLPs
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    inputs_dict = None
    global_emb_table_dict = {}

    # FM Layer
    inputs_dict, fm_embeddings_dict = build_feature_embeddings(fm_fields, inputs_dict=inputs_dict, global_emb_table_dict=global_emb_table_dict, return_list=True, disable=["task", "domain"])
    fm_inputs = []
    for dtype in fm_embeddings_dict:
        fm_inputs.extend(list(chain.from_iterable(fm_embeddings_dict[dtype].values())))
    fm_logit = FM()(fm_inputs)

    inputs_dict, dnn_embeddings_dict = build_feature_embeddings(dnn_fields, inputs_dict=inputs_dict, global_emb_table_dict=global_emb_table_dict, disable=["task", "domain"])
    dnn_inputs = []
    # history embeddings sequence aggregation with target embedding
    if "history" in dnn_embeddings_dict:
        # For regression task, item_embeddings may don't exist. We could take user embeddings as query.
        if "item" in dnn_embeddings_dict:
            target_embeddings = dnn_embeddings_dict["item"]
        else:
            target_embeddings = dnn_embeddings_dict["user"]
        seq_inputs = dnn_embeddings_dict.pop("history")
        history_embeddings = history_embedding_aggregation(seq_inputs,
                                                           target_embeddings,
                                                           history_agg, **agg_kwargs)
        dnn_inputs.append(history_embeddings)

    # Deep Hidden Layer
    dnn_inputs.append(concatenate(dnn_embeddings_dict))
    if hidden_units[-1] != 1:
        hidden_units.append(1)
    dnn_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name="dnn/MLPs")
    dnn_logit = dnn_layer(tf.concat(dnn_inputs, axis=-1))

    outputs = predict(task, fm_logit + dnn_logit)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model
