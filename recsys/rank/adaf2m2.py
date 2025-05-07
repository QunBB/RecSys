"""
AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System

DASFAA'2025：https://arxiv.org/abs/2501.15816
"""
from typing import List, Union, Callable, Optional, Dict, Any, Tuple
from itertools import chain
from copy import deepcopy

import tensorflow as tf
from tensorflow.keras import backend as K

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.embedding import MaskEmbedding
from recsys.layers.interaction import InteractionExpert
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def merge_state_aware_embeddings(
        id_fields: List[Field],
        non_id_fields: List[Field],
        inputs_dict: dict,
        global_emb_table_dict: dict
) -> tf.Tensor:
    """Compute and concatenate state_aware features
    """
    # state-aware non-ID features
    inputs_dict, state_embeddings_dict = build_feature_embeddings(
        non_id_fields,
        disable=["task", "domain"],
        return_list=True,
        inputs_dict=inputs_dict,
        global_emb_table_dict=global_emb_table_dict
    )

    assert len(set([f"{f.dtype}/{f.group}/{f.name}" for f in id_fields])) == len(id_fields), \
        "There are duplicate state_aware ID features present."
    # state-aware ID features. `global_emb_table_dict` to ensure unique ID embedding
    inputs_dict, state_id_embeddings_dict = build_feature_embeddings(
        id_fields,
        disable=["task", "domain"],
        return_list=True,
        inputs_dict=inputs_dict,
        global_emb_table_dict=global_emb_table_dict
    )

    # state-aware non-ID embeddings
    state_id_embeddings = []
    for dtype in state_id_embeddings_dict:
        state_id_embeddings.extend(list(chain.from_iterable(state_id_embeddings_dict[dtype].values())))

    index_dict = {}
    # compute ID embeddings' norm
    norm_dict = {}
    norm_fields = []
    for field in id_fields:
        # get the corresponding index of current feature embedding in state_id_embeddings_dict
        index = index_dict.setdefault(field.belong, {}).setdefault(field.group, 0)
        emb = state_id_embeddings_dict[field.belong][field.group][index]
        dim = emb.shape.as_list()[-1]
        for n, func in zip(["log", "sqrt", "square"], [tf.math.log, tf.math.sqrt, tf.math.square]):
            # clip norm to avoid nan
            norm = tf.clip_by_value(tf.norm(emb, axis=-1), clip_value_min=1e-7, clip_value_max=1e7)
            norm_dict[f"{field.name}_norm_{n}"] = func(norm)
            norm_fields.append(Field(f"{field.name}_norm_{n}", dim=dim, vocabulary_size=field.vocabulary_size,
                                     dtype=emb.dtype, belong=field.belong, group=field.group))

        index_dict[field.belong][field.group] += 1

    # embeddings' norm features
    _, norm_id_embeddings_dict = build_feature_embeddings(
        norm_fields,
        disable=["task", "domain"],
        return_list=True,
        inputs_dict=norm_dict,
        global_emb_table_dict=global_emb_table_dict
    )

    # concatenate all state-aware embeddings
    final_state_embeddings = []
    for belong in state_embeddings_dict:
        final_state_embeddings.extend(list(chain.from_iterable(state_embeddings_dict[belong].values())))
    for belong in norm_id_embeddings_dict:
        final_state_embeddings.extend(list(chain.from_iterable(norm_id_embeddings_dict[belong].values())))
    final_state_embeddings.extend(state_id_embeddings)

    return tf.concat(final_state_embeddings, axis=-1)


def interaction_deep_layer(
        embeddings: list,
        interaction_layer: tf.keras.layers.Layer,
        mlp_layer: tf.keras.layers.Layer,
        history_embeddings: tf.Tensor = None,
) -> tf.Tensor:
    """Interaction layer + Deep network
    """
    # Interaction Network
    interaction_output = interaction_layer(embeddings)

    dnn_inputs = [interaction_output]
    if history_embeddings is not None:
        dnn_inputs.append(history_embeddings)

    # Deep Network
    mlp_output = mlp_layer(tf.concat(dnn_inputs, axis=-1))

    return mlp_output


def adaf2m2(
        fields: List[Field],
        state_non_id_fields: List[Field],
        state_id_fields: List[Field],
        num_sample: int,
        interaction: InteractionExpert,
        interaction_params: Optional[Dict[str, Any]] = None,
        task: Task = Task(name="ctr", belong="binary"),
        min_probability: float = 0.1,
        max_probability: float = 0.5,
        mask_history: bool = False,
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
    :param task: which task
    :param state_non_id_fields: the list of all state-aware non-ID fields.
    :param state_id_fields: the list of all state-aware ID-based fields. Noted that:
                1. each one muse be in above `fields`.
                2. `vocabulary_size` here is for id embedding norm, not for origin id embedding.
                3. will concatenate their origin embeddings and embedding norn.
    :param num_sample: the number of augmented samples in feature mask
    :param interaction: interaction layer
    :param interaction_params: the parameters of interaction layer
    :param min_probability: the minimum probability of mask embedding
    :param max_probability: the maximum probability of mask embedding
    :param mask_history: whether to mask history embedding
    :param hidden_units: hidden units in MLPs
    :param activation: activation in MLPs
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: whether to use batch normalization in MLPs
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    global_emb_table_dict = {}
    inputs_dict = {}
    inputs_dict, embeddings_dict = build_feature_embeddings(fields,
                                                            disable=["task", "domain"],
                                                            return_list=True,
                                                            inputs_dict=inputs_dict,
                                                            global_emb_table_dict=global_emb_table_dict)

    for f in state_id_fields:
        assert f.name in global_emb_table_dict, f"The state-aware ID-based field: {f.name} not exists"

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
                                                           history_agg, **agg_kwargs)

    embeddings_list = []
    for dtype in embeddings_dict:
        embeddings_list.extend(list(chain.from_iterable(embeddings_dict[dtype].values())))

    # Only one instance for shared parameters
    interaction_layer = interaction.init_layer(**(interaction_params or {}))
    mlp_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name="dnn/MLPs")

    # Feature Mask
    feature_mask = MaskEmbedding(num_sample, min_probability=min_probability, max_probability=max_probability)
    augmented_embeddings = feature_mask(embeddings_list)

    if history_embeddings is not None:
        if mask_history:
            history_emb_mask = MaskEmbedding(num_sample, min_probability=min_probability, max_probability=max_probability)
            augmented_history_embeddings = history_emb_mask([history_embeddings])
        else:
            augmented_history_embeddings = tf.repeat(history_embeddings, num_sample, axis=0)
    else:
        augmented_history_embeddings = None

    augmented_outputs = interaction_deep_layer(augmented_embeddings, interaction_layer, mlp_layer, augmented_history_embeddings)

    # Adaptive Feature Modeling
    num_features = len(embeddings_list)
    if history_embeddings is not None:
        num_features += 1
    state_aware_embeddings = merge_state_aware_embeddings(state_id_fields, state_non_id_fields, inputs_dict, global_emb_table_dict)
    adaptive_weights = FeedForwardLayer(hidden_units=[num_features], activation="sigmoid", name="dnn/weight_generator")(state_aware_embeddings)
    adaptive_weights = tf.split(adaptive_weights, num_features, axis=1)
    adaptive_embeddings_list = [emb * adaptive_weights[i] for i, emb in enumerate(embeddings_list)]
    if history_embeddings is not None:
        adaptive_history_embeddings = history_embeddings * adaptive_weights[-1]
    else:
        adaptive_history_embeddings = None

    main_outputs = interaction_deep_layer(adaptive_embeddings_list, interaction_layer, mlp_layer, adaptive_history_embeddings)
    main_prediction = predict(task, main_outputs)

    aux_task = deepcopy(task)
    aux_task.return_logit = False
    aux_task.name = "aux_" + aux_task.name
    aux_prediction = predict(aux_task, augmented_outputs)

    final_outputs = []
    if isinstance(main_prediction, list):
        final_outputs.extend(main_prediction)
    else:
        final_outputs.append(main_prediction)
    final_outputs.append(aux_prediction)

    model = Model(inputs=inputs_dict, outputs=final_outputs)

    return model
