"""
Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

RecSys'2020：https://dl.acm.org/doi/10.1145/3383313.3412236
"""
from typing import List, Union, Callable

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import FeedForwardLayer, PredictLayer, Identity
from recsys.layers.interaction import GatingNetwork
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def ple(
        fields: List[Field],
        task_list: List[Task],
        num_layer: int = 1,
        num_experts: int = 2,
        num_shared_experts: int = 2,
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
    :param num_layer: the number of layers
    :param num_experts: the number of task experts
    :param num_shared_experts: the number of shared experts
    :param experts_dim: the dimension of experts
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

    output_list = []
    # inputs for each task in every layer
    layer_inputs = {task.name: embeddings for task in task_list}
    # the shared experts' inputs
    shared_inputs = embeddings
    # Multi-level Extraction Networks
    for layer_id in range(num_layer):

        shared_experts_list = []
        for i in range(num_shared_experts):
            shared_experts_list.append(
                FeedForwardLayer([experts_dim], activation, l2_reg, dropout, use_bn, name=f"dnn/layer_{layer_id}_shared_expert_{i}")(shared_inputs)
            )

        task_experts = []
        for task in task_list:
            # extraction for each task experts
            current_inputs = layer_inputs[task.name]
            for i in range(num_experts):
                task_experts.append(
                    FeedForwardLayer([experts_dim], activation, l2_reg, dropout, use_bn, name=f"dnn/layer_{layer_id}_{task.name}_expert_{i}")(current_inputs)
                )

            gating_network = GatingNetwork(num_experts=num_experts + num_shared_experts,
                                           name=f"dnn/extraction_{layer_id}_{task.name}",
                                           l2_reg=l2_reg,
                                           dropout=dropout,
                                           use_bn=use_bn)
            extraction = gating_network([current_inputs,
                                         tf.stack(shared_experts_list + task_experts[-num_experts:], axis=1)])
            layer_inputs[task.name] = extraction

            # task tower in the last layer
            if layer_id == num_layer - 1:

                tower_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name=f"dnn/tower_{task.name}")
                tower_output = tower_layer(extraction)

                prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/predict_layer_{task.name}")(tower_output)
                output_list.append(Identity(name=task.name)(prediction))

        # shared experts don't compute extraction in the last layer
        if not output_list:
            # extraction for shared experts
            gating_network = GatingNetwork(num_experts=num_shared_experts + num_experts * len(task_list),
                                           name=f"dnn/extraction_shared_{layer_id}",
                                           l2_reg=l2_reg,
                                           dropout=dropout,
                                           use_bn=use_bn)
            extraction = gating_network([shared_inputs,
                                         tf.stack(shared_experts_list + task_experts, axis=1)])
            shared_inputs = extraction

    model = Model(inputs=inputs_dict, outputs=output_list)

    return model