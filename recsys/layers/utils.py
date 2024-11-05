from typing import Dict, Union

import tensorflow as tf

from recsys.layers.aggregation import get_agg_layer


def history_embedding_aggregation(
        history_embeddings: Union[Dict[str, tf.Tensor], tf.Tensor],
        target_embeddings: tf.Tensor,
        aggregation: str,
        **kwargs
):
    outputs = []

    if isinstance(history_embeddings, dict) and len(history_embeddings) == 1:
        history_embeddings = list(history_embeddings.values())[0]

    if isinstance(history_embeddings, dict):
        for group in history_embeddings:
            kwargs["name"] = f"dnn/his-{aggregation}-{group}"
            layer = get_agg_layer(aggregation, **kwargs)
            outputs.append(
                layer([target_embeddings, history_embeddings[group]])
            )
    else:
        kwargs["name"] = f"dnn/his-{aggregation}"
        layer = get_agg_layer(aggregation, **kwargs)
        outputs.append(
            layer([target_embeddings, history_embeddings])
        )

    return tf.concat(outputs, axis=-1)