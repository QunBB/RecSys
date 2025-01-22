"""
Ads Recommendation in a Collapsed and Entangled World

KDD'2024：https://arxiv.org/abs/2403.00793
"""
from typing import List, Union, Callable, Optional, Dict, Tuple
from itertools import chain
from collections import defaultdict

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import FeedForwardLayer, PredictLayer, Identity
from recsys.layers.interaction import GatingNetwork
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def flexible_group(
        group: Dict[Union[Task, Tuple[Task,...]], List[Field]]
) -> Dict[Tuple[Task,...], List[Field]]:
    if not group:
        return group

    return {k if isinstance(k, tuple) else (k,): v for k, v in group.items()}


def assign_field_for_shared(
        fields: List[List[Field]]
) -> List[Field]:
    """assign the embedding table whose dimension is medium to the shared expert
    """
    if len(fields) == 1:
        return fields[0]

    fields.sort(key=lambda x: max([f.dim for f in x]))

    return fields[len(fields) // 2]


def build_embeddings(fields, inputs_dict, prefix, history_agg, **agg_kwargs):
    inputs_dict, embeddings_dict = build_feature_embeddings(fields,
                                                            inputs_dict=inputs_dict,
                                                            disable=["task", "domain"],
                                                            prefix=f"embedding/{prefix}_")
    # history embeddings sequence aggregation with target embedding
    if "history" in embeddings_dict:
        # For regression task, item_embeddings may don't exist. We could take user embeddings as query.
        if "item" in embeddings_dict:
            target_embeddings = embeddings_dict["item"]
        else:
            target_embeddings = embeddings_dict["user"]
        embeddings_dict["history"] = history_embedding_aggregation(embeddings_dict["history"],
                                                                   target_embeddings,
                                                                   history_agg, prefix=f"dnn/his_{prefix}", **agg_kwargs)

    embeddings = concatenate(embeddings_dict)

    return inputs_dict, embeddings


def stem(
    task_group: Dict[Union[Task, Tuple[Task,...]], List[Field]],
    auxiliary_task_group: Optional[Dict[Union[Task, Tuple[Task,...]], List[Field]]] = None,
    shared_group: List[Field] = None,
    stop_gradients: bool = False,
    experts_dim: int = 64,
    hidden_units: List[int] = [100, 64],
    activation: Union[str, Callable] = "relu",
    dropout: float = 0.,
    l2_reg: float = 0.,
    use_bn: bool = False,
    history_agg: str = "attention",
    agg_kwargs: dict = {}
) -> tf.keras.Model:
    """Shared and Task-specific Embedding (STEM) and Asymmetric Multi-Embedding (AME)

    :param task_group: a dict. The key is a tuple with the tasks
                    which belong to the same group and share the same embedding table.
                    The value is the corresponding fields that with the same name are same in different groups,
                    mainly for assigning special embedding dimension.
    :param auxiliary_task_group: same as `task_group`, but as for auxiliary tasks learning.
    :param shared_group: for Asymmetric Multi-Embedding (AME), you could assign special dimension for shared embedding table and expert.
    :param stop_gradients: whether to stop gradients in Shared and Task-specific Embedding (STEM).
    :param experts_dim: the dimension of experts
    :param hidden_units: DNN hidden units in PPNet
    :param activation: DNN activation in PPNet
    :param dropout: dropout rate of DNN
    :param l2_reg: l2 regularizer of DNN parameters
    :param use_bn: whether to use batch
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:

    Example:

    ```python
    # stem
    model = stem(task_group={(task_a, task_b): fields, task_c: fields}, stop_gradients=True)

    # ame
    fields_a = [Field('uid', dim=4), Field('item_id', dim=4), ...]
    fields_b = [Field('uid', dim=8), Field('item_id', dim=8), ...]
    fields_shared = [Field('uid', dim=6), Field('item_id', dim=6), ...]
    model = stem(task_group={
        task_a: fields_a #
        task_b: fields_b #
    }, shared_group=fields_shared
    )

    # stem for auxiliary learning
    model = stem(task_group={task_a: fields}, auxiliary_task_group={task_b: fields})
    ```
    """
    task_group = flexible_group(task_group)
    auxiliary_task_group = flexible_group(auxiliary_task_group)

    inputs_dict = None
    task_embedding_dict = {}
    # assign embedding table to each main task
    for i, group in enumerate(task_group):
        inputs_dict, embeddings = build_embeddings(task_group[group], inputs_dict, f"main_group_{i}", history_agg, **agg_kwargs)
        for task in group:
            task_embedding_dict[task] = embeddings

    # assign embedding table to each auxiliary task
    if auxiliary_task_group:
        for i, group in enumerate(auxiliary_task_group):
            inputs_dict, embeddings = build_embeddings(auxiliary_task_group[group], inputs_dict, f"auxiliary_group_{i}", history_agg, **agg_kwargs)
            for task in group:
                task_embedding_dict[task] = embeddings
    # we need shared expert when there are no auxiliary tasks
    else:
        if shared_group is None:
            shared_group = assign_field_for_shared(list(task_group.values()))

        inputs_dict, embeddings = build_embeddings(shared_group, inputs_dict, "shared", history_agg, **agg_kwargs)
        task_embedding_dict["shared"] = embeddings

    task_experts = {}
    # each task's expert
    for task, embeddings in task_embedding_dict.items():
        task_experts[task] = FeedForwardLayer([experts_dim], activation, l2_reg, dropout, use_bn, name=f"dnn/expert_{task}")(embeddings)

    # assign experts for each task's tower inputs
    used_experts = defaultdict(list)
    for task_i in chain(*task_group.keys()):
        for task_j in task_experts:
            if stop_gradients and task_i is not task_j:
                used_experts[task_i].append(tf.stop_gradient(task_experts[task_j]))
            else:
                used_experts[task_i].append(task_experts[task_j])

    if auxiliary_task_group:
        for task in chain(*auxiliary_task_group.keys()):
            used_experts[task] = [task_experts[task]]

    if "shared" in task_experts:
        for task in used_experts:
            used_experts[task].append(task_experts["shared"])

    output_list = []
    # Multi-gate Mixture-of-Experts & Tower Layer
    for task in used_experts:
        gating_network = GatingNetwork(num_experts=len(used_experts[task]),
                                       name=f"dnn/gate_{task}",
                                       l2_reg=l2_reg,
                                       dropout=dropout,
                                       use_bn=use_bn)
        tower_inputs = gating_network([task_embedding_dict[task],
                                       tf.stack(used_experts[task], axis=1)])

        tower_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name=f"dnn/tower_{task}")
        tower_output = tower_layer(tower_inputs)

        prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/predict_layer_{task}")(tower_output)
        output_list.append(Identity(name=task.name)(prediction))

    model = Model(inputs=inputs_dict, outputs=output_list)

    return model
