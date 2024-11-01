import numpy as np
import tensorflow as tf

from recsys.multidomain.pepnet import pepnet, Field, Task

task_list = [
    Task(name='click'),
    Task(name='like'),
    Task(name='fav')
]


def create_model():
    model = pepnet([
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item', length=20, belong='history'),
            Field('domain_1_id', vocabulary_size=2, emb="domain_id", belong='domain', group='domain_1'),
            Field('domain_2_id', vocabulary_size=2, emb="domain_id", belong='domain', group='domain_2'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ], task_list, [64, 32],
    history_agg="attention", agg_kwargs={}
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
        'domain_1_id': np.zeros([n_samples]),
        'domain_2_id': np.ones([n_samples]),
        'context_id': np.random.randint(0, 20, [n_samples]),
    }
    labels = {f"{domain}_{t.name}": np.random.randint(0, 2, [n_samples]) for t in task_list for domain in ['domain_1', 'domain_2']}

    return data, labels


if __name__ == '__main__':
    model = create_model()
    data, labels = create_dataset()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(data, labels, batch_size=32, epochs=10)
