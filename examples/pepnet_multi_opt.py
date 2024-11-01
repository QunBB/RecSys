import tensorflow as tf

from examples.pepnet import create_model, create_dataset


def train(data, labels):
    model = create_model()

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
