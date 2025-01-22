"""
Temporal Interest Network for User Response Prediction

WWW'2024：https://arxiv.org/abs/2308.08487
"""
from typing import List, Union, Callable

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def tin(
        fields: List[Field],
        task: Task = Task(name="ctr", belong="binary"),
        hidden_units: List[int] = [100, 64],
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        use_bn: bool = False,
        attention_kwargs: dict = {}
) -> tf.keras.Model:
    """

    :param fields: the list of all fields
    :param task: which task
    :param hidden_units: hidden units in MLPs
    :param activation: activation
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :param attention_kwargs: arguments about attention in TIM
    :return:
    """
    inputs_dict, embeddings_dict = build_feature_embeddings(fields, disable=["task", "domain"])

    embeddings_dict["history"] = history_embedding_aggregation(embeddings_dict["history"],
                                                               embeddings_dict["item"],
                                                               "tim",
                                                               **attention_kwargs)

    embeddings = concatenate(embeddings_dict)

    mlp_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name="dnn/MLPs")
    mlp_output = mlp_layer(embeddings)

    outputs = predict(task, mlp_output)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model
