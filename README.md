# Feed: Towards Personalization-Effective Federated Learning

This repository contains the code and baselines for the manuscript:

> [Feed: Towards Personalization-Effective Federated Learning](https://github.com/DoublePg/Feed)
>

Federated learning (FL) has become an emerging paradigm via cooperative training models among distributed clients without leaking data privacy. The performance degradation of FL on heterogeneous data has driven the development of personalized FL (PFL) solutions, where different models are built for individual clients. However, our in-depth analysis on existing PFL approaches discloses that they only support limited personalization regarding modeling capability and training strategy. To this end, we propose a novel PFL solution, Feed, that employs an enhanced shared-private model architecture and equips with a hybrid federated training strategy. Specifically, to model heterogeneous data in different clients, we design an ensemble-based shared encoder that  generates an ensemble of embeddings, and a private decoder that adaptively aggregates the embeddings for personalized prediction. In addition, we propose a server-side hybrid federated aggregation strategy to enable effective training the heterogeneous shared-private model. To prevent personalization degradation in local model update, we further optimize the personalized local training on the client-side by smoothing the historical encoders. Extensive experiments on MNIST/FEMNIST, CIFAR10/CIFAR100, and YELP datasets validate that Feed consistently outperforms the state-of-the-art approaches.

## Preparation

### Dataset generation

For each dataset, we provide `IID` and `Non-IID` cases used in our experiments. We partition these datasets into 10 clients in IID and Non-IID cases, and store in `Feed/dataset/` folders.

| tasks | public datasets      | private datasets    |
| ---------- | --------------- | ------------- |
| HR          | MNIST          | EMNIST       |
| OC          | CIFAR10         | CIFAR100      |
| SA          | YELP-automotive | YELP-restaurant |


### Downloading dependencies

```
- python=3.7.13=h12debd9_0
- numpy==1.19.2
- scikit-learn=1.0.2=py37h51133e4_1
- tensorflow=2.0.0=gpu_py37h768510d_0
- tensorflow-gpu=2.0.0=h0d30ee6_0
- keras=2.3.1=0
``` 
更多详细设置参见一级目录下的`environment.yml`文件。
