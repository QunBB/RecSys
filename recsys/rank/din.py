"""
Deep Interest Network for Click-Through Rate Prediction

KDD'2018：https://arxiv.org/abs/1706.06978
"""
from typing import List, Union, Callable

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def din(
        fields: List[Field],
        task: Task = Task(name="ctr", belong="binary"),
        attention_hidden_units: List[int] = [80, 40],
        attention_activation: Union[str, Callable] = "dice",
        query_ffn: bool = False,
        query_activation: Union[str, Callable] = "prelu",
        dnn_hidden_units: List[int] = [100, 64],
        dnn_activation: Union[str, Callable] = "dice",
        dropout: float = 0.,
        l2_reg: float = 0.,
        use_bn: bool = False
) -> tf.keras.Model:
    """

    :param fields: the list of all fields
    :param task: which task
    :param attention_hidden_units: hidden units in attention layer
    :param attention_activation: activation in dnn of attention layer
    :param query_ffn: whether to apply a feed-forward layer for query. It is necessary when dimension of query is not equal to key's
    :param query_activation: activation in feed-forward layer for query
    :param attention_activation: activation in dnn of attention layer
    :param dnn_hidden_units: hidden units in MLPs
    :param dnn_activation: activation in MLPs
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :return:
    """
    inputs_dict, embeddings_dict = build_feature_embeddings(fields, disable=["task", "domain"])

    embeddings_dict["history"] = history_embedding_aggregation(embeddings_dict["history"],
                                                               embeddings_dict["item"],
                                                               "attention",
                                                               ffn_hidden_units=attention_hidden_units,
                                                               ffn_activation=attention_activation,
                                                               query_ffn=query_ffn,
                                                               query_activation=query_activation)

    embeddings = concatenate(embeddings_dict)

    mlp_layer = FeedForwardLayer(dnn_hidden_units, dnn_activation, l2_reg, dropout, use_bn, name="dnn/MLPs")
    mlp_output = mlp_layer(embeddings)

    outputs = predict(task, mlp_output)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model
