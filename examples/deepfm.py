import numpy as np
import tensorflow as tf

from recsys.rank.deepfm import deepfm, Field, Task


def create_model(task=Task(name="ctr", belong="binary")):
    common_fields = [Field('uid', vocabulary_size=100),
                     Field('item_id', vocabulary_size=100, belong='item'),
                     Field('cate_id', vocabulary_size=20, belong='item')]
    dnn_fields = [Field('his_item_id', vocabulary_size=100, emb='item_id', length=20, belong='history'),
                  Field('his_cate_id', vocabulary_size=20, emb='cate_id', length=20, belong='history'), ]
    fm_fields = [Field('age', vocabulary_size=100, belong='context')]
    model = deepfm(
        dnn_fields=common_fields + dnn_fields,
        fm_fields=common_fields + fm_fields,
        task=task
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
        'age': np.random.randint(0, 100, [n_samples]),
    }
    labels = np.where(np.random.random([n_samples]) <= 0.2, 1, 0)

    return data, labels


if __name__ == '__main__':
    model = create_model()
    data, labels = create_dataset()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=tf.keras.metrics.AUC())
    model.fit(data, labels, batch_size=128, epochs=10)
