import tensorflow as tf

from recsys.train.losses import RankingLoss
from examples.tin import create_model, create_dataset, Task


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