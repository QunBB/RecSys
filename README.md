# RecSys

This project is about recommendation system including rank&match models and metrics which are all implemented by `tensorflow 2.x`.

You can use these models with `model.fit()` ，and `model.predict()`  through `tf.keras.Model`.

The implement for `tensorflow 1.x` is in this [github](https://github.com/QunBB/DeepLearning/tree/main/recommendation).

# 🛠️ Installation

- ## Install via pip

To install, simply use `pip` to pull down from [PyPI](https://pypi.org/project/deep-rec-kit/).

```bash
pip install deep-rec-kit
```

- ## Install from source

If you want to use latest features, or develop new features, you can also build it from source.

```bash
git clone https://github.com/QunBB/RecSys
cd RecSys
pip install -e .
```


# 📖 Models List

`......` means that it will be continuously updated. 

## Multi-Task Multi-Domain

| model   | paper                                                        | blog                                             | implemented |
| ------- | ------------------------------------------------------------ | ------------------------------------------------ | ----------- |
| ......  |                                                              |                                                  |             |
| STEM | [KDD 2024] [Ads Recommendation in a Collapsed and Entangled World](https://arxiv.org/abs/2403.00793) | [zhihu](https://zhuanlan.zhihu.com/p/19885938029) | ✅ |
| PEPNet  | [KDD 2023] [PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information](https://arxiv.org/pdf/2302.01115) | [zhihu](https://zhuanlan.zhihu.com/p/4552106145) | ✅           |
| M2M     | [CIKM 2022] [Leaving No One Behind: A Multi-Scenario Multi-Task Meta Learning Approach for Advertiser Modeling](https://arxiv.org/abs/2201.06814) | [zhihu](https://zhuanlan.zhihu.com/p/939534954)  | ✅           |
| SAR-Net | [CIKM 2021] [SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios](https://arxiv.org/pdf/2110.06475) | [zhihu](https://zhuanlan.zhihu.com/p/718704281)  |             |
| Star    | [CIKM 2021] [One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction](https://arxiv.org/abs/2101.11427) | [zhihu](https://zhuanlan.zhihu.com/p/717054800)  | ✅           |
| PLE     | [RecSys 2020] [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236) | [zhihu](https://zhuanlan.zhihu.com/p/425209494)  | ✅            |
| MMoE    | [KDD 2018] [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://arxiv.org/pdf/2305.16360) | [zhihu](https://zhuanlan.zhihu.com/p/425209494)  | ✅           |

## Rank

| model         | paper                                                        | blog                                              | implemented |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------- | ----------- |
| ......        |                                                              |                                                   |             |
| AdaF^2M^2 | [DASFAA 2025] [AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System](https://arxiv.org/abs/2501.15816) | [zhihu](https://zhuanlan.zhihu.com/p/1903561181152641052) | ✅ |
| HMoE          | [KDD 2024] [Ads Recommendation in a Collapsed and Entangled World](https://arxiv.org/abs/2403.00793) | [zhihu](https://zhuanlan.zhihu.com/p/19885938029) | ✅           |
| GwPFM         | [KDD 2024] [Ads Recommendation in a Collapsed and Entangled World](https://arxiv.org/abs/2403.00793) | [zhihu](https://zhuanlan.zhihu.com/p/19885938029) | ✅           |
| TIN           | [WWW 2024] [Temporal Interest Network for User Response Prediction](https://arxiv.org/abs/2308.08487) | [zhihu](https://zhuanlan.zhihu.com/p/7832498217)  | ✅           |
| FiBiNet++     | [CIKM 2023 ] [FiBiNet++: Reducing Model Size by Low Rank Feature Interaction Layer for CTR Prediction](https://arxiv.org/abs/2209.05016) | [zhihu](https://zhuanlan.zhihu.com/p/603262632)   | ✅ |
| MaskNet       | [DLP-KDD 2021] [MaskNet: Introducing Feature-Wise Multiplication to CTR Ranking Models by Instance-Guided Mask](https://arxiv.org/abs/2102.07619) | [zhihu](https://zhuanlan.zhihu.com/p/660375034)   |             |
| ContextNet    | [arXiv 2021] [ContextNet: A Click-Through Rate Prediction Framework Using Contextual information to Refine Feature Embedding](https://arxiv.org/abs/2107.12025) | [zhihu](https://zhuanlan.zhihu.com/p/660375034)   |             |
| DCN V2        | [WWW 2021] [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535) | [zhihu](https://zhuanlan.zhihu.com/p/631668163)   | ✅           |
| FEFM          | [arXiv 2020] [Field-Embedded Factorization Machines for Click-through rate prediction](https://arxiv.org/abs/2009.09931) | [zhihu](https://zhuanlan.zhihu.com/p/613030015)   |             |
| FiBiNET       | [RecSys 2019] [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) | [zhihu](https://zhuanlan.zhihu.com/p/603262632)   | ✅ |
| DSIN          | [IJCAI 2019] [Deep Session Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1905.06482) | [zhihu](https://zhuanlan.zhihu.com/p/688338754)   |             |
| DIEN          | [AAAI 2019] [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672) | [zhihu](https://zhuanlan.zhihu.com/p/685855305)   |             |
| DIN           | [KDD 2018] [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) | [zhihu](https://zhuanlan.zhihu.com/p/679852484)   | ✅           |
| xDeepFM       | [KDD 2018] [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170) | [zhihu](https://zhuanlan.zhihu.com/p/634584585)   |             |
| FwFM          | [WWW 2018] [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/abs/1806.03514) | [zhihu](https://zhuanlan.zhihu.com/p/613030015)   |             |
| NFM           | [SIGIR 2017] [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/abs/1708.05027) | [zhihu](https://zhuanlan.zhihu.com/p/634584585)   |             |
| DeepFM        | [IJCAI 2017] [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) | [zhihu](https://zhuanlan.zhihu.com/p/631668163)   | ✅            |
| Wide & Deep   | [DLRS 2016] [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) | [zhihu](https://zhuanlan.zhihu.com/p/631668163)   |             |
| Deep Crossing | [KDD 2016] [Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf) | [zhihu](https://zhuanlan.zhihu.com/p/623567076)   |             |
| PNN           | [ICDM 2016] [Product-based Neural Networks for User Response Prediction](https://arxiv.org/abs/1611.00144) | [zhihu](https://zhuanlan.zhihu.com/p/623567076)   | ✅           |
| FNN           | [arXiv 2016] [Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/abs/1601.02376) | [zhihu](https://zhuanlan.zhihu.com/p/623567076)   |             |
| FFM           | [RecSys 2016] [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/doi/10.1145/2959100.2959134) | [zhihu](https://zhuanlan.zhihu.com/p/613030015)   |             |

## Match

| model                          | paper                                                        | blog                                            | implemented |
| ------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- | ----------- |
| ......                         |                                                              |                                                 |             |
| Dual Augmented Two-tower Model | [DLP-KDD 2021] [A Dual Augmented Two-tower Model for Online Large-scale Recommendation](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf) | [zhihu](https://zhuanlan.zhihu.com/p/608636233) |             |
| ComiRec                        | [KDD 2020] [Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/abs/2005.09347) | [zhihu](https://zhuanlan.zhihu.com/p/568781562) |             |
| MIND                           | [CIKM 2019] [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/abs/1904.08030) | [zhihu](https://zhuanlan.zhihu.com/p/463064543) |             |
| Youtube DNN                    | [RecSys 2016] [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/10.1145/2959100.2959190) | [zhihu](https://zhuanlan.zhihu.com/p/405907646) |             |

# 🏗️ Metrics

**Metrics for recommendation system.**

It will be coming soon.

# 📘 Example

```python
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


if __name__ == '__main__':
    model = create_model()
    data, labels = create_dataset()

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(data, labels, batch_size=32, epochs=10)
```

# 🚀 Mulitple Optimizers

Those layers with prefix "dnn" will use the adam optimizer, and adagrad for prefix "embedding". 
Also, you must have the default optimizer for legacy layers.

```python
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
    # absolutely same as the above ......


def create_dataset():
    # absolutely same as the above ......


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
```

