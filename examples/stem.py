import numpy as np
import tensorflow as tf

from recsys.feature import Field, Task
from recsys.multitask.stem import stem


task_list = [
    Task(name='click'),
    Task(name='like'),
    Task(name='buy')
]


def create_stem_model():
    fields = [
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item_id', length=20, belong='history'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ]

    model = stem(task_group={(task_list[0], task_list[1]): fields,
                             task_list[2]: fields},
                 stop_gradients=True)

    print(model.summary())

    return model


def create_ame_model():
    fields = [
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item_id', length=20, belong='history'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ]
    task_group = {}
    dims = [4, 8, 16, 8]
    for task, dim in zip(task_list + ["shared"], dims):
        task_fields = fields.copy()
        for f in task_fields:
            f.dim = dim
        task_group[task] = task_fields
    shared_group = task_group.pop("shared")

    model = stem(task_group=task_group, shared_group=shared_group)

    print(model.summary())

    return model


def create_stem_al_model():
    fields = [
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item_id', length=20, belong='history'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ]

    model = stem(task_group={task_list[0]: fields},
                 auxiliary_task_group={(task_list[1], task_list[2],): fields})

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
    data, labels = create_dataset()

    for func in [create_stem_model, create_ame_model, create_stem_al_model]:
        model = func()

        model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=tf.keras.metrics.AUC())
        model.fit(data, labels, batch_size=32, epochs=10)
