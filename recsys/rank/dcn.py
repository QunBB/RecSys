"""
DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

WWW'2021：https://arxiv.org/abs/2008.13535

Deep & Cross Network for Ad Click Predictions

ADKDD'2017：https://arxiv.org/pdf/1708.05123.pdf
"""
from typing import List, Union, Callable, Optional

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.interaction import CrossNet
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def dcn(
        fields: List[Field],
        task: Task = Task(name="ctr", belong="binary"),
        layer_num: int = 2,
        cross_type: str = "matrix",
        low_rank_dim: Optional[int] = None,
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
    :param layer_num: the number of cross network layer
    :param cross_type: feature cross method, can be "vector" or "matrix"
    :param low_rank_dim: Low-Rank dimension in DCN-V2
    :param hidden_units: hidden units in MLPs
    :param activation: activation in MLPs
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    inputs_dict, embeddings_dict = build_feature_embeddings(fields, disable=["task", "domain"])

    history_embeddings = None
    # history embeddings sequence aggregation with target embedding
    if "history" in embeddings_dict:
        # For regression task, item_embeddings may don't exist. We could take user embeddings as query.
        if "item" in embeddings_dict:
            target_embeddings = embeddings_dict["item"]
        else:
            target_embeddings = embeddings_dict["user"]
        seq_inputs = embeddings_dict.pop("history")
        history_embeddings = history_embedding_aggregation(seq_inputs,
                                                           target_embeddings,
                                                           history_agg, **agg_kwargs)

    embeddings = concatenate(embeddings_dict)

    # Cross Network
    cross_output = CrossNet(layer_num, cross_type, low_rank_dim, l2_reg, name="dnn/cross_net")(embeddings)

    # Deep Network
    mlp_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name="dnn/MLPs")
    mlp_output = mlp_layer(embeddings if history_embeddings is None else tf.concat([embeddings, history_embeddings], axis=-1))

    stack = tf.concat([cross_output, mlp_output], axis=-1)

    outputs = predict(task, stack)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model
