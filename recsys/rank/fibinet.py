"""
FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction

RecSys'2019：https://arxiv.org/abs/1905.09433

FiBiNet++:Improving FiBiNet by Greatly Reducing Model Size for CTR Prediction

CIKM'2023：https://arxiv.org/pdf/2209.05016.pdf
"""
from typing import List, Union, Callable
from itertools import chain

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings
from recsys.layers.core import FeedForwardLayer, predict
from recsys.layers.interaction import SENet, BiLinearLayer
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def fibinet(
        fields: List[Field],
        task: Task = Task(name="ctr", belong="binary"),
        reduction_ratio: int = 3,
        num_groups: int = 2,
        bilinear_output_size: int = 50,
        bilinear_type: str = "all",
        bilinear_1bit: bool = True,
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
    :param reduction_ratio: reduction ratio in SENet
    :param num_groups: the groups' number of feature embeddings in SENet
    :param bilinear_output_size: the output size of Bi-Linear+ Module
    :param bilinear_type: the feature interaction type in Bi-Linear+ Module
    :param bilinear_1bit: whether to apply 1bit inner product after interaction in Bi-Linear+ Module
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

    # SENet+ Module
    senet = SENet(reduction_ratio, num_groups, name="dnn/senet")
    senet_output = senet(embeddings_list)

    # Bi-Linear+ Module
    bilinear = BiLinearLayer(bilinear_output_size, bilinear_type, product_1bit=bilinear_1bit, name="dnn/bilinear")
    bilinear_output = bilinear(embeddings_list)

    dnn_inputs = [senet_output, bilinear_output]
    if history_embeddings is not None:
        dnn_inputs.append(history_embeddings)

    # Deep Network
    mlp_layer = FeedForwardLayer(hidden_units, activation, l2_reg, dropout, use_bn, name="dnn/MLPs")
    mlp_output = mlp_layer(tf.concat(dnn_inputs, axis=-1))

    outputs = predict(task, mlp_output)

    model = Model(inputs=inputs_dict, outputs=outputs)

    return model
