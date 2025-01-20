import numpy as np
import tensorflow as tf

from recsys.multitask.mmoe import mmoe, Field, Task

task_list = [
    Task(name='click'),
    Task(name='like'),
    Task(name='buy')
]


def create_model():
    fields = [
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item_id', length=20, belong='history'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ]

    model = mmoe(fields, task_list, 3)

    print(model.summary())

    return model


def create_dataset():
    n_samples = 2000
    np.random.seed(2024)
    data = {
        'uid': np.random.randint(0, 100, [n_samples]),
        'item_id': np.random.randint(0, 20, [n_samples]),
        'his_item_id': np.random.randint(0, 20, [n_samples, 20]),
        'context_id': np.random.randint(0, 20, [n_samples]),
    }
    labels = {t.name: np.random.randint(0, 2, [n_samples]) for t in task_list}

    return data, labels


if __name__ == '__main__':
    model = create_model()
    data, labels = create_dataset()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=tf.keras.metrics.AUC())
    model.fit(data, labels, batch_size=32, epochs=10)
