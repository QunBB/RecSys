"""
PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information

KDD‘2023：https://arxiv.org/pdf/2302.01115
"""
from typing import List, Union, Callable

import tensorflow as tf
from tensorflow.keras.initializers import Zeros

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings
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
        embeddings_dict["history"] = history_embedding_aggregation(embeddings_dict["history"], embeddings_dict["item"],
                                                                   history_agg, **agg_kwargs)
        embeddings_dict["context"] = tf.concat([embeddings_dict["history"], embeddings_dict["context"]], axis=-1)

    epnet = EPNet(l2_reg, name="dnn/epnet")
    ppnet = PPNet(len(task_list), hidden_units, activation, l2_reg, name="dnn/ppnet")

    output_list = []
    # compute each domain's prediction
    domain_embeddings = embeddings_dict["domain"]
    for group in domain_embeddings:

        ep_emb = epnet([domain_embeddings[group], embeddings_dict["context"]])

        pp_output = ppnet([ep_emb, tf.concat([embeddings_dict["user"], embeddings_dict["item"]], axis=-1)])

        # compute each task's prediction in special domain
        for i, task in enumerate(task_list):
            if len(domain_embeddings) == 1:
                output_name = task.name
            else:
                output_name = f"{group}_{task.name}"

            prediction = PredictLayer(task.belong, task.num_classes, name=f"dnn/{output_name}")(pp_output[i])
            output_list.append(Identity(name=output_name)(prediction))

    # each prediction's name is "{domain.group}_{task.name}" when there are more than one domain
    # and is "{task.name}" when there is only one domain
    # model = tf.keras.Model(inputs=inputs_dict, outputs=output_list)
    model = Model(inputs=inputs_dict, outputs=output_list)

    return model
