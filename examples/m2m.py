import numpy as np
import tensorflow as tf

from recsys.multidomain.m2m import m2m, Field, Task

task_list = [
    Task(name='pv', belong="regression"),
    Task(name='click', belong="regression"),
    Task(name='active_rate', belong="regression")
]

num_domain = 3


def create_model():
    fields = [
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_seq_1_id', vocabulary_size=20, emb='item_id', length=20, belong='history', group='seq_1'),
            Field('his_seq_2_id', vocabulary_size=20, emb='cate_id', length=20, belong='history', group='seq_2'),
            Field('domain_id', vocabulary_size=num_domain, belong='domain'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ]
    # each task's fields
    for task in task_list:
        fields.append(Field(f'{task.name}_id', vocabulary_size=len(task_list), emb='task_id', belong='task', group=task.name))
        fields.append(Field(f'{task.name}_count', vocabulary_size=1, emb='task_count', belong='task', group=task.name, dtype="float32"))

    model = m2m(fields, task_list, 3)

    print(model.summary())

    return model


def create_dataset():
    n_samples = 2000
    np.random.seed(2024)
    data = {
        'uid': np.random.randint(0, 100, [n_samples]),
        'item_id': np.random.randint(0, 20, [n_samples]),
        'his_seq_1_id': np.random.randint(0, 20, [n_samples, 20]),
        'his_seq_2_id': np.random.randint(0, 20, [n_samples, 20]),
        'domain_id': np.random.randint(0, num_domain, [n_samples]),
        'context_id': np.random.randint(0, 20, [n_samples]),
    }
    for i, task in enumerate(task_list):
        data[f'{task.name}_id'] = np.ones([n_samples]) * i
        data[f'{task.name}_count'] = np.random.random([n_samples])
    labels = {t.name: np.random.random([n_samples]) for t in task_list}

    return data, labels


if __name__ == '__main__':
    model = create_model()
    data, labels = create_dataset()

    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    model.fit(data, labels, batch_size=32, epochs=10)
