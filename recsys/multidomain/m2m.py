"""
Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling

CIKM'2022：https://arxiv.org/abs/2201.06814
"""
from typing import List, Union, Callable

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import PredictLayer, Identity, FeedForwardLayer
from recsys.layers.interaction import MetaAttention, MetaTower, MetaUnit
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def m2m(
        fields: List[Field],
        task_list: List[Task],
        num_experts: int,
        view_dim: int = 256,
        scenario_dim: int = 64,
        meta_tower_depth: int = 3,
        meta_unit_depth: int = 3,
        meta_unit_shared: bool = True,
        activation: Union[str, Callable] = "leaky_relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        history_agg: str = "transformer",
        agg_kwargs: dict = {"position_merge": "concat"}
) -> tf.keras.Model:
    """

    :param fields: the list of all fields
    :param task_list: the list of multiple task
    :param num_experts: the number of experts
    :param view_dim: the dimension of experts view and tasks view
    :param scenario_dim: the dimension of scenario view
    :param meta_tower_depth: the number of meta tower layer
    :param meta_unit_depth: the number of meta unit layer
    :param meta_unit_shared: whether to share meta unit
    :param activation: activation
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of parameters
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    inputs_dict, embeddings_dict = build_feature_embeddings(fields, disable=None)

    # Scenario Knowledge Representation
    scenario_mlp = FeedForwardLayer([scenario_dim], activation, l2_reg, dropout, name="dnn/scenario_mlp")
    scenario_inputs = [embeddings_dict.pop("domain"), embeddings_dict["user"]]
    if "item" in embeddings_dict:
        scenario_inputs.append(embeddings_dict["item"])
    scenario_views = scenario_mlp(tf.concat(scenario_inputs, axis=-1))

    # history embeddings sequence aggregation with target embedding
    if "history" in embeddings_dict:
        embeddings_dict["history"] = history_embedding_aggregation(embeddings_dict["history"],
                                                                   target_embeddings=scenario_views,
                                                                   aggregation=history_agg, **agg_kwargs)

    task_embeddings = embeddings_dict.pop("task")

    # Expert View Representation
    expert_inputs = concatenate(embeddings_dict)
    expert_views = []
    for i in range(num_experts):
        expert_views.append(
            FeedForwardLayer([view_dim], activation, l2_reg, dropout, name=f"dnn/expert_{i}_mlp")(expert_inputs)
        )
    expert_views = tf.stack(expert_views, axis=1)

    # Whether to share meta unit
    attention_shared_meta_unit = None
    tower_shared_meta_unit = None
    if meta_unit_shared:
        attention_shared_meta_unit = MetaUnit(meta_unit_depth, activation, dropout, l2_reg, name="dnn/attention_shared_meta_unit")
        tower_shared_meta_unit = MetaUnit(meta_unit_depth, activation, dropout, l2_reg, name="dnn/tower_shared_meta_unit")

    output_list = []
    # group name of task fields must be equal to task name
    for task in task_list:
        # Task View Representation
        assert task.name in task_embeddings, f"Missing field group for input \"{task.name}\""
        task_views = FeedForwardLayer([view_dim], activation, l2_reg, dropout, name=f"dnn/{task.name}_mlp")(task_embeddings[task.name])

        meta_attention = MetaAttention(meta_unit=attention_shared_meta_unit, num_layer=meta_unit_depth, activation=activation, dropout=dropout, l2_reg=l2_reg, name=f"dnn/meta_attention_{task.name}")
        attention_output = meta_attention([expert_views, task_views, scenario_views])

        meta_tower = MetaTower(meta_unit=tower_shared_meta_unit, num_layer=meta_tower_depth, meta_unit_depth=meta_unit_depth, activation=activation, dropout=dropout,l2_reg=l2_reg, name=f"dnn/meta_tower_{task.name}")
        tower_output = meta_tower([attention_output, scenario_views])

        prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/predict_layer_{task.name}")(tower_output)
        output_list.append(Identity(name=task.name)(prediction))

    # each prediction's name is "{task.name}"
    model = Model(inputs=inputs_dict, outputs=output_list)

    return model