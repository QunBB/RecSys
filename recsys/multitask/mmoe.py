"""
Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts

KDD'2018：https://arxiv.org/pdf/2305.16360
"""
from typing import List, Union, Callable

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import FeedForwardLayer, PredictLayer, Identity
from recsys.layers.utils import history_embedding_aggregation
from recsys.layers.interaction import GatingNetwork
from recsys.train.multi_opt_model import Model


def mmoe(
        fields: List[Field],
        task_list: List[Task],
        num_experts: int = 3,
        experts_dim: int = 64,
        hidden_units: List[int] = [100, 64],
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        use_bn: bool = False,
        history_agg: str = "attention",
        agg_kwargs: dict = {}
) -> tf.keras.Model:
    """

    :param fields: the list of all fields
    :param task_list: the list of multiple task
    :param num_experts: the number of shared experts
    :param experts_dim: the dimension of shared experts
    :param hidden_units: hidden units in MLPs
    :param activation: activation
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    inputs_dict, embeddings_dict = build_feature_embeddings(fields, disable=["task", "domain"])

    # history embeddings sequence aggregation with target embedding
    if "history" in embeddings_dict:
        # For regression task, item_embeddings may don't exist. We could take user embeddings as query.
        if "item" in embeddings_dict:
            target_embeddings = embeddings_dict["item"]
        else:
            target_embeddings = embeddings_dict["user"]
        embeddings_dict["history"] = history_embedding_aggregation(embeddings_dict["history"],
                                                                   target_embeddings,
                                                                   history_agg, **agg_kwargs)

    embeddings = concatenate(embeddings_dict)

    experts_list = []
    for i in range(num_experts):
        experts_list.append(
            FeedForwardLayer([experts_dim], activation, l2_reg, dropout, use_bn, name=f"dnn/expert_{i}")(embeddings)
        )

    experts_output = tf.stack(experts_list, axis=1)

    output_list = []
    for task in task_list:

        gating_network = GatingNetwork(num_experts=num_experts,
                                       name=f"dnn/mmoe_{task.name}",
                                       l2_reg=l2_reg,
                                       dropout=dropout,
                                       use_bn=use_bn)
        tower_inputs = gating_network([embeddings, experts_output])

        tower_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name=f"dnn/tower_{task.name}")
        tower_output = tower_layer(tower_inputs)

        prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/predict_layer_{task.name}")(tower_output)
        output_list.append(Identity(name=task.name)(prediction))

    model = Model(inputs=inputs_dict, outputs=output_list)

    return model
