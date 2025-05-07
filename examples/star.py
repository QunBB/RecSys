import numpy as np
import tensorflow as tf

from recsys.feature import Field
from recsys.multidomain.star import star


num_domain = 3


def create_model():
    model = star([
            Field('uid', vocabulary_size=100),
            Field('item_id', vocabulary_size=20, belong='item'),
            Field('his_item_id', vocabulary_size=20, emb='item_id', length=20, belong='history'),
            Field('domain_id', vocabulary_size=num_domain, belong='domain'),
            Field('context_id', vocabulary_size=20, belong='context'),
        ], num_domain=num_domain
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
        'domain_id': np.random.randint(0, num_domain, [n_samples]),
        'context_id': np.random.randint(0, 20, [n_samples]),
    }
    labels = np.random.randint(0, 2, [n_samples])

    return data, labels


if __name__ == '__main__':
    model = create_model()
    data, labels = create_dataset()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(data, labels, batch_size=32, epochs=10)
