"""
Ads Recommendation in a Collapsed and Entangled World

KDD'2024：https://arxiv.org/abs/2403.00793
"""
from typing import List, Union, Callable, Optional
from collections import defaultdict
from itertools import chain

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.interaction import GwPFM
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def gwpfm(
        fields: List[Field],
        task: Task = Task(name="ctr", belong="binary"),
        hidden_units: Optional[List[int]] = None,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        use_bn: bool = False,
        history_agg: str = "attention",
        agg_kwargs: dict = {}
) -> tf.keras.Model:
    """GwPFM: Group-weighted Part-aware Factorization Machines

    :param fields: the list of all fields, use the `field.group` to group fields
    :param task: which task
    :param hidden_units: hidden units in MLPs, can be `None` when not apply DNN after features interaction
    :param activation: activation in MLPs
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    # count the number of field groups
    num_groups = len(set(f.group for f in fields if f.belong != "history"))
    for f in fields:
        f.dim *= num_groups

    inputs_dict, embeddings_dict = build_feature_embeddings(fields, disable=["task", "domain"], return_list=True)

    history_embeddings = None
    # history embeddings sequence aggregation with target embedding
    if "history" in embeddings_dict:
        # For regression task, item_embeddings may don't exist. We could take user embeddings as query.
        if "item" in embeddings_dict:
            target_embeddings = embeddings_dict["item"]
        else:
            target_embeddings = embeddings_dict["user"]
        target_embeddings = tf.concat(list(chain.from_iterable(target_embeddings.values())), axis=-1)
        seq_inputs = tf.keras.layers.Concatenate(axis=-1)(list(chain.from_iterable(embeddings_dict.pop("history").values())))
        history_embeddings = history_embedding_aggregation(seq_inputs,
                                                           target_embeddings,
                                                           history_agg, **agg_kwargs)

    # get each field groups' embeddings
    group_dict = defaultdict(list)
    for dtype in embeddings_dict:
        for group in embeddings_dict[dtype]:
            group_dict[group].extend(embeddings_dict[dtype][group])

    # record each field's group index
    field_group_idx = []
    # GwPFM model's inputs: [batch_size, num_fields, num_groups, dim]
    inputs = []
    for i, group in enumerate(group_dict):
        field_group_idx.extend([i] * len(group_dict[group]))
        inputs.extend(group_dict[group])
    inputs = tf.stack(inputs, axis=1)
    inputs = tf.reshape(inputs, [-1, len(field_group_idx), num_groups, inputs.shape[-1] // num_groups])

    # GwPFM interaction
    interaction = GwPFM(num_groups=num_groups, field_group_idx=field_group_idx, name="dnn/gwpfm")(inputs)

    tower_inputs = interaction if history_embeddings is None else tf.concat([interaction, history_embeddings], axis=-1)

    # Deep Network
    if hidden_units:
        mlp_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name="dnn/MLPs")
        final_interaction = mlp_layer(tower_inputs)
    else:
        final_interaction = tower_inputs

    outputs = predict(task, final_interaction)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model