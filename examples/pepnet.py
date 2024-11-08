import numpy as np
import tensorflow as tf

from recsys.multidomain.pepnet import pepnet, Field, Task

task_list = [
    Task(name='click'),
    Task(name='like'),
    Task(name='fav')
]

domain_list = ['domain_1', 'domain_2']


def create_model():
    fields = [
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item_id', length=20, belong='history'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ]
    # each domain's fields
    for domain in domain_list:
        fields.append(Field(f'{domain}_id', vocabulary_size=len(domain_list), emb='domain_id', belong='domain', group=domain))
        # dense feature
        fields.append(Field(f'{domain}_impression', vocabulary_size=1, emb='domain_impression', belong='domain', group=domain, dtype="float32"))

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
    }
    for i, domain in enumerate(domain_list):
        data[f'{domain}_id'] = np.ones([n_samples]) * i
        data[f'{domain}_impression'] = np.random.random([n_samples])
    labels = {f'{domain}_{t.name}': np.random.randint(0, 2, [n_samples]) for t in task_list for domain in domain_list}

    return data, labels


if __name__ == '__main__':
    model = create_model()
    data, labels = create_dataset()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(data, labels, batch_size=32, epochs=10)
