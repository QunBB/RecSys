"""
Ads Recommendation in a Collapsed and Entangled World

KDD'2024：https://arxiv.org/abs/2403.00793
"""
from typing import List, Union, Callable, Optional, Dict, Tuple, Any, Sequence
from collections import defaultdict
from itertools import chain

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings
from recsys.layers.core import FeedForwardLayer, PredictLayer, Identity
from recsys.layers.interaction import WeightedSum, InteractionExpert
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def count_num_groups(
        fields: List[Field],
        experts: Union[Dict[InteractionExpert, Any], Sequence[InteractionExpert]]
) -> Optional[int]:
    """Count number of embeddings for field-aware model, e.g., FFM, GwPFM
    """
    for e in experts:
        if e == InteractionExpert.GwPFM:
            num_groups = len(set(f.group for f in fields if f.belong != "history"))
            return num_groups

    return None


def get_embedding_table(fields, inputs_dict, num_groups, name, history_agg, **agg_kwargs):
    if num_groups is not None:
        fields = fields.copy()
        for f in fields:
            f.dim *= num_groups

    inputs_dict, embeddings_dict = build_feature_embeddings(fields,
                                                            inputs_dict=inputs_dict,
                                                            prefix=f"embedding/{name}_",
                                                            disable=["task", "domain"],
                                                            return_list=True)

    history_embeddings = None
    # history embeddings sequence aggregation with target embedding
    if "history" in embeddings_dict:
        # For regression task, item_embeddings may don't exist. We could take user embeddings as query.
        if "item" in embeddings_dict:
            target_embeddings = embeddings_dict["item"]
        else:
            target_embeddings = embeddings_dict["user"]
        target_embeddings = tf.concat(list(chain.from_iterable(target_embeddings.values())), axis=-1)
        seq_inputs = tf.keras.layers.Concatenate(axis=-1)(
            list(chain.from_iterable(embeddings_dict.pop("history").values())))
        history_embeddings = history_embedding_aggregation(seq_inputs,
                                                           target_embeddings,
                                                           history_agg,
                                                           prefix=f"dnn/his_{name}_",
                                                           **agg_kwargs)

    # get each field groups' embeddings
    group_dict = defaultdict(list)
    for dtype in embeddings_dict:
        for group in embeddings_dict[dtype]:
            group_dict[group].extend(embeddings_dict[dtype][group])

    # record each field's group index, specially for GwPFM model
    field_group_idx = []
    embeddings = []
    for i, group in enumerate(group_dict):
        field_group_idx.extend([i] * len(group_dict[group]))
        embeddings.extend(group_dict[group])

    return embeddings, history_embeddings, inputs_dict, field_group_idx


def hmoe(
        fields: List[Field],
        expert_group: Tuple[Dict[InteractionExpert, Dict[str, Any]],...],
        task: Task = Task(name="ctr", belong="binary"),
        gate_weighted: bool = False,
        sum_weighted: bool = False,
        hidden_units: List[int] = [128, 64],
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        use_bn: bool = False,
        history_agg: str = "attention",
        agg_kwargs: dict = {}
) -> tf.keras.Model:
    """Heterogeneous Mixture-of-Experts with Multi-Embedding

    :param fields: the list of all fields, use the `field.group` to group fields
    :param expert_group: interaction expert groups. Experts in the same dict are the same group that will share same embedding table
    :param task: which task
    :param gate_weighted: whether to apply gate-weighted sum of the outputs from the experts
    :param sum_weighted: whether to apply simple weighted sum of the outputs from the experts
    :param hidden_units: hidden units in MLPs, can be `None` when not apply DNN after features interaction
    :param activation: activation in MLPs
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:

    Example:

    ```python
    # `GwPFM` and `ProductLayer` expert shared the first embedding table and `CrossNet` use the second.
    experts = (
        # first experts group
        {
            InteractionExpert.GwPFM: {},
            InteractionExpert.ProductLayer: {"outer": True},
        },
        # second experts group
        {
            InteractionExpert.CrossNet: {}
        }
    )
    model = hmoe(fields, task=task, expert_group=experts)
    ```
    """
    experts_embedding = {}
    experts_parameter = {}
    inputs_dict = None
    # assign embedding table and initialization parameters to each expert
    for i, group in enumerate(expert_group):
        num_groups = count_num_groups(fields, group)
        embeddings, history_embeddings, inputs_dict, field_group_idx = get_embedding_table(
            fields, inputs_dict, num_groups, f"group_{i}", history_agg, **agg_kwargs
        )
        for expert in group:
            experts_embedding[expert] = (embeddings, history_embeddings)

            experts_parameter[expert] = group[expert]
            experts_parameter[expert]["name"] = str(expert)
            if expert == InteractionExpert.GwPFM:
                experts_parameter[expert]["num_groups"] = num_groups
                experts_parameter[expert]["field_group_idx"] = field_group_idx

    # expert interaction & MLPs layer
    experts_output = {}
    for expert, (embeddings, history_embeddings) in experts_embedding.items():
        interaction = expert.init_layer(**experts_parameter[expert])(embeddings)
        if history_embeddings is not None:
            interaction = tf.concat([interaction, history_embeddings], axis=-1)
        # Deep Network
        mlp_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name=f"dnn/{expert}_MLPs")
        experts_output[expert] = mlp_layer(interaction)

    if gate_weighted:
        # gate-weighted sum of the outputs from the experts
        scores = []
        final_interaction = []
        for i, group in enumerate(expert_group):
            gate_ffn = FeedForwardLayer([len(group)], name=f"dnn/gate_{i}")
            # use the embeddings corresponding to current group to compute gate weight
            for n, expert in enumerate(group):
                final_interaction.append(experts_output[expert])
                if n == 0:
                    scores.append(
                        gate_ffn(tf.concat(experts_embedding[expert][0], axis=-1))
                    )

        scores = tf.nn.softmax(tf.concat(scores, axis=-1), axis=-1)
        final_interaction = tf.squeeze(
            tf.matmul(tf.expand_dims(scores, axis=1), tf.stack(final_interaction, axis=1)),
            axis=1)
    elif sum_weighted:
        final_interaction = WeightedSum()(experts_output.values())
    else:
        # element-wise summation
        final_interaction = sum(experts_output.values())

    # classification tower
    if task.return_logit:
        logit = PredictLayer(task.belong, task.num_classes, as_logit=True, name=f"dnn/logit_layer")(final_interaction)
        prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/predict_layer")(logit)
        outputs = [Identity(name=task.name)(prediction), Identity(name="logit")(logit)]
    else:
        outputs = PredictLayer(task.belong, task.num_classes, name=f"dnn/predict_layer")(final_interaction)
        outputs = Identity(name=task.name)(outputs)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model
