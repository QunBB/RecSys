import numpy as np
import tensorflow as tf

from recsys.feature import Field, Task
from recsys.multidomain.pepnet import pepnet

task_list = [
    Task(name='click'),
    Task(name='like'),
    Task(name='fav')
]

num_domain = 3


def create_model():
    fields = [
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item_id', length=20, belong='history'),
            Field('context_id', vocabulary_size=20, belong='context'),
            # domain's fields
            Field(f'domain_id', vocabulary_size=num_domain, belong='domain'),
            Field(f'domain_impression', vocabulary_size=1, belong='domain', dtype="float32")
        ]

    model = pepnet(fields, task_list, [64, 32],
                   history_agg='attention', agg_kwargs={}
                   # history_agg='transformer', agg_kwargs={'num_layers': 1, 'd_model': 4, 'num_heads': 2, 'dff': 64}
                   )

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
        'domain_id': np.random.randint(0, num_domain, [n_samples]),
        'domain_impression': np.random.random([n_samples])
    }
    labels = {t.name: np.random.randint(0, 2, [n_samples]) for t in task_list}

    return data, labels


def train(data, labels):
    model = create_model()

    # Those layers with prefix "dnn" will use the adam optimizer, and adagrad for prefix "embedding".
    # Also, you must have the default optimizer for legacy layers.
    model.compile(optimizer={'dnn': 'adam', 'embedding': 'Adagrad', 'default': 'adam'},
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(data, labels, batch_size=32, epochs=10)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.save('./pepnet-saved/model.ckpt')

    print(model({k: v[:10] for k, v in data.items()}))

    print(model.optimizer['embedding'].variables())


def restore(data):
    model = create_model()

    model.compile(optimizer={'dnn': 'adam', 'embedding': 'Adagrad', 'default': 'adam'},
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore('./pepnet-saved/model.ckpt-1')

    print(model({k: v[:10] for k, v in data.items()}))

    for layer in model.optimizer:
        model.optimizer[layer].build(model.special_layer_variables[layer])
    print(model.optimizer['embedding'].variables())


if __name__ == '__main__':
    data, labels = create_dataset()

    train(data, labels)

    restore(data)
