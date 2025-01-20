"""
PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information

KDD‘2023：https://arxiv.org/pdf/2302.01115
"""
from typing import List, Union, Callable

import tensorflow as tf

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import PredictLayer, Identity
from recsys.layers.interaction import EPNet, PPNet
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def pepnet(
        fields: List[Field],
        task_list: List[Task],
        hidden_units: List[int] = [100, 64],
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        history_agg: str = "mean_pooling",
        agg_kwargs: dict = {}
) -> tf.keras.Model:
    """

    :param fields: the list of all fields
    :param task_list: the list of multiple task
    :param hidden_units: DNN hidden units in PPNet
    :param activation: DNN activation in PPNet
    :param dropout: dropout rate of DNN
    :param l2_reg: l2 regularizer of DNN parameters
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    inputs_dict, embeddings_dict = build_feature_embeddings(fields)

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
        embeddings_dict["context"] = concatenate(embeddings_dict, ["history", "context"])

    epnet = EPNet(l2_reg, name="dnn/epnet")
    ppnet = PPNet(len(task_list), hidden_units, activation, dropout, l2_reg, name="dnn/ppnet")

    output_list = []

    ep_emb = epnet([embeddings_dict["domain"], embeddings_dict["context"]])

    pp_output = ppnet([ep_emb, concatenate(embeddings_dict, ["user", "item"])])

    # compute each task's prediction in corresponding domain
    for i, task in enumerate(task_list):

        prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/{task}")(pp_output[i])
        output_list.append(Identity(name=task.name)(prediction))

    model = Model(inputs=inputs_dict, outputs=output_list)

    return model
