"""
DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems

WWW'2021：https://arxiv.org/abs/2008.13535

Deep & Cross Network for Ad Click Predictions

ADKDD'2017：https://arxiv.org/pdf/1708.05123.pdf
"""
from typing import List, Union, Callable, Optional
from itertools import chain

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.interaction import ProductLayer
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def pnn(
        fields: List[Field],
        task: Task = Task(name="ctr", belong="binary"),
        inner_product: bool = True,
        outer_product: bool = False,
        kernel_type: str = "mat",
        net_activation: Optional[Union[str, Callable]] = None,
        net_size: Optional[int] = None,
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
    :param inner_product: whether to apply inner product
    :param outer_product: whether to apply outer product
    :param kernel_type: which kernel to use
    :param net_activation: activation in micro network, when use `net` kernel
    :param net_size: product output size in micro network
    :param hidden_units: hidden units in MLPs
    :param activation: activation in MLPs
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of MLPs parameters
    :param use_bn: Whether to use batch normalization in MLPs
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
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
        seq_inputs = tf.keras.layers.Concatenate(axis=-1)(
            list(chain.from_iterable(embeddings_dict.pop("history").values())))
        history_embeddings = history_embedding_aggregation(seq_inputs,
                                                           target_embeddings,
                                                           history_agg, **agg_kwargs)

    embeddings_list = []
    for dtype in embeddings_dict:
        embeddings_list.extend(list(chain.from_iterable(embeddings_dict[dtype].values())))

    # Product Network
    product_layer = ProductLayer(inner=inner_product, outer=outer_product, kernel_type=kernel_type,
                                 activation=net_activation, net_size=net_size, name="dnn/product_layer")
    product_output = product_layer(embeddings_list)

    dnn_inputs = embeddings_list + [product_output]
    if history_embeddings is not None:
        dnn_inputs.append(history_embeddings)

    # Deep Network
    mlp_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name="dnn/MLPs")
    mlp_output = mlp_layer(tf.concat(dnn_inputs, axis=-1))

    outputs = predict(task, mlp_output)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model
