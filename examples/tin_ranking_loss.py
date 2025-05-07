import random

import numpy as np
import tensorflow as tf

from recsys.feature import Field, Task
from recsys.rank.tin import tin
from recsys.train.losses import RankingLoss

random.seed(2024)
np.random.seed(2024)
tf.random.set_seed(2024)


def create_model(task=Task(name="ctr", belong="binary")):
    model = tin([
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=100, belong='item'),
            Field('cate_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=100, emb='item_id', length=20, belong='history'),
            Field('his_cate_id', vocabulary_size=20, emb='cate_id', length=20, belong='history'),
            Field('context_id', vocabulary_size=100, belong='context'),
        ], task=task
    )

    print(model.summary())

    return model


def create_dataset(n_samples=20000, seed=2024):
    np.random.seed(seed)
    data = {
        'uid': np.random.randint(0, 100, [n_samples]),
        'item_id': np.random.randint(0, 100, [n_samples]),
        'cate_id': np.random.randint(0, 20, [n_samples]),
        'his_item_id': np.random.randint(0, 100, [n_samples, 20]),
        'his_cate_id': np.random.randint(0, 20, [n_samples, 20]),
        'context_id': np.random.randint(0, 100, [n_samples]),
    }
    labels = np.where(np.random.random([n_samples]) <= 0.2, 1, 0)

    return data, labels


if __name__ == '__main__':
    model = create_model(task=Task(name="ctr", belong="binary", return_logit=True))
    data, labels = create_dataset()

    model.compile(optimizer='adam',
                  loss=[tf.keras.losses.BinaryCrossentropy(), RankingLoss()],
                  loss_weights=[0.7, 0.3],
                  metrics={"ctr": tf.keras.metrics.AUC()})
    # ctr_loss: 0.4846 - ctr_auc: 0.6149
    model.fit(data, labels, batch_size=128, epochs=10)

    # ctr_loss: 0.4817 - ctr_auc: 0.4923
    valid_data, labels = create_dataset(n_samples=1000, seed=2025)
    model.evaluate(valid_data, labels)