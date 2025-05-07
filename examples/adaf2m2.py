import numpy as np
import tensorflow as tf

from recsys.feature import Field, Task
from recsys.layers.interaction import InteractionExpert
from recsys.rank.adaf2m2 import adaf2m2
from recsys.train.losses import AugmentedBinaryCrossentropy


def create_model(task):
    model = adaf2m2([
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=100, belong='item'),
            Field('cate_id', vocabulary_size=20, belong='item'),
            Field('context_id', vocabulary_size=100, belong='context'),
            # history sequence
            Field('his_item_id', vocabulary_size=100, emb='item_id', length=20, belong='history'),
            Field('his_cate_id', vocabulary_size=20, emb='cate_id', length=20, belong='history'),
        ],
        state_id_fields=[
            # `vocabulary_size` here is for id embedding norm, not for origin id embedding
            Field('uid', vocabulary_size=0),
            Field('item_id', vocabulary_size=0, belong='item'),
        ],
        state_non_id_fields=[
            # you can set `vocabulary_size=1` to use embeddings
            Field('activate_days', vocabulary_size=0, dtype="float32"),
            Field('interaction_count', vocabulary_size=0, dtype="float32"),
        ],
        num_sample=3, interaction=InteractionExpert.ProductLayer, interaction_params={"outer": True}, task=task
    )

    print(model.summary())

    return model


def create_dataset():
    n_samples = 2000
    np.random.seed(2024)
    data = {
        'uid': np.random.randint(0, 100, [n_samples]),
        'item_id': np.random.randint(0, 100, [n_samples]),
        'cate_id': np.random.randint(0, 20, [n_samples]),
        'his_item_id': np.random.randint(0, 100, [n_samples, 20]),
        'his_cate_id': np.random.randint(0, 20, [n_samples, 20]),
        'context_id': np.random.randint(0, 100, [n_samples]),
        'activate_days': np.random.randint(0, 30, [n_samples]),
        'interaction_count': np.random.randint(0, 100, [n_samples]),
    }
    labels = np.random.randint(0, 2, [n_samples])

    return data, labels


if __name__ == '__main__':
    task = Task(name="ctr", belong="binary")
    model = create_model(task)
    data, labels = create_dataset()

    model.compile(optimizer='adam',
                  loss=[tf.keras.losses.BinaryCrossentropy(), AugmentedBinaryCrossentropy()],  # `AugmentedBinaryCrossentropy` for aux loss
                  loss_weights=[1.0, 0.2],
                  metrics={task.name: tf.keras.metrics.AUC()}  # use a dictionary to avoid computing aux output metric
                  )
    model.fit(data, labels, batch_size=128, epochs=10)
