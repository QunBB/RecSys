"""
One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction

CIKM'2021：https://arxiv.org/abs/2101.11427
"""
from typing import List, Union, Callable

import tensorflow as tf
from tensorflow.keras.initializers import Zeros

from recsys.feature import Field, Task
from recsys.feature.utils import build_feature_embeddings, concatenate
from recsys.layers.core import PredictLayer, FeedForwardLayer
from recsys.layers.interaction import PartitionedNormalization, StarTopologyFCN
from recsys.layers.utils import history_embedding_aggregation
from recsys.train.multi_opt_model import Model


def star(
        fields: List[Field],
        num_domain: int,
        task: Task = Task(name="ctr", belong="binary"),
        fcn_units: List[int] = [100, 64],
        aux_units: List[int] = [100, 64],
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.,
        l2_reg: float = 0.,
        history_agg: str = "attention",
        agg_kwargs: dict = {}
) -> tf.keras.Model:
    """

    :param fields: the list of all fields
    :param num_domain: the number of domains
    :param task: which task
    :param fcn_units: hidden units in Star Topology FCN
    :param aux_units: hidden units in Auxiliary Network
    :param activation: activation
    :param dropout: dropout rate
    :param l2_reg: l2 regularizer of DNN parameters
    :param history_agg: the method of aggregation about historical behavior features
    :param agg_kwargs: arguments about aggregation
    :return:
    """
    # domain field must only have one: `domain indicator`
    index = [i for i, f in enumerate(fields) if f.belong == "domain"]
    assert len(index) == 1 and fields[index[0]].dtype in ("int32", "int64"), "domain field must only have one: `domain indicator`"
    domain_indicator_name = fields[index[0]].name

    inputs_dict, embeddings_dict = build_feature_embeddings(fields)

    domain_embedding = embeddings_dict.pop("domain")
    domain_indicator = inputs_dict[domain_indicator_name]

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

    embeddings = concatenate(embeddings_dict)

    # Star Topology FCN
    # Partitioned Normalization
    fcn_pn = PartitionedNormalization(num_domain=num_domain, name="dnn/fcn/partitioned_normalization")
    fcn_inputs = fcn_pn([embeddings, domain_indicator])

    fcn = StarTopologyFCN(num_domain, fcn_units, activation, dropout, l2_reg, name="dnn/fcn")
    fcn_prediction_layer = PredictLayer(task.belong, task.num_classes, as_logit=True, name=f"dnn/fcn_{task.name}")

    fcn_output = fcn([fcn_inputs, domain_indicator])
    fcn_logit = fcn_prediction_layer(fcn_output)

    # Auxiliary Network
    # Partitioned Normalization
    aux_pn = PartitionedNormalization(num_domain=num_domain, name="dnn/aux/partitioned_normalization")
    aux_inputs = aux_pn([tf.concat([domain_embedding, embeddings], axis=-1),
                         domain_indicator])

    aux_layer = FeedForwardLayer(aux_units, activation, l2_reg, dropout_rate=dropout)
    aux_prediction_layer = PredictLayer(task.belong, task.num_classes, as_logit=True, name=f"dnn/aux_{task.name}")

    aux_output = aux_layer(aux_inputs)
    aux_logit = aux_prediction_layer(aux_output)

    # Final Prediction Layer
    final_prediction_layer = PredictLayer(task.belong, task.num_classes, name=f"dnn/final_{task.name}")
    final_prediction = final_prediction_layer(fcn_logit + aux_logit)

    model = Model(inputs=inputs_dict, outputs=final_prediction)

    return model
