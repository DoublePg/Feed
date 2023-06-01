# Feed: Towards Personalization-Effective Federated Learning

This repository contains the code and baselines for the manuscript:

> [Feed: Towards Personalization-Effective Federated Learning](https://github.com/DoublePg/Feed)
>

Federated learning (FL) has become an emerging paradigm via cooperative training models among distributed clients without leaking data privacy. The performance degradation of FL on heterogeneous data has driven the development of personalized FL (PFL) solutions, where different models are built for individual clients. However, our in-depth analysis on existing PFL approaches discloses that they only support limited personalization regarding modeling capability and training strategy. To this end, we propose a novel PFL solution, Feed, that employs an enhanced shared-private model architecture and equips with a hybrid federated training strategy. Specifically, to model heterogeneous data in different clients, we design an ensemble-based shared encoder that  generates an ensemble of embeddings, and a private decoder that adaptively aggregates the embeddings for personalized prediction. In addition, we propose a server-side hybrid federated aggregation strategy to enable effective training the heterogeneous shared-private model. To prevent personalization degradation in local model update, we further optimize the personalized local training on the client-side by smoothing the historical encoders. Extensive experiments on MNIST/FEMNIST, CIFAR10/CIFAR100, and YELP datasets validate that Feed consistently outperforms the state-of-the-art approaches.

## Preparation

### Dataset generation

For each dataset, we provide `IID` and `Non-IID` cases used in our experiments. For more details, please refer to `data_utils.py`. We partition these datasets into 10 clients in IID and Non-IID cases, and store in `Feed/dataset/` folders. 

| tasks | public datasets      | private datasets    |
| ---------- | --------------- | ------------- |
| HR          | MNIST          | EMNIST       |
| OC          | CIFAR10         | CIFAR100      |
| SA          | YELP-automotive | YELP-restaurant |


### Installing dependencies

```
- python=3.7.13=h12debd9_0
- numpy==1.19.2
- scikit-learn=1.0.2=py37h51133e4_1
- tensorflow=2.0.0=gpu_py37h768510d_0
- tensorflow-gpu=2.0.0=h0d30ee6_0
- keras=2.3.1=0
``` 
更多详细设置参见一级目录下的`environment.yml`文件。

## Instructions

### Files Instructions

* `main files` : The files  `FEMNIST_Balanced.py `,  `FEMNIST_Imbalanced.py `,  `CIFAR_Balanced.py `,  `CIFAR_Imbalanced.py `,  `YELP_Balanced.py `, and  `YELP_Imbalanced.py ` represent three tasks under IID and Non-IID cases.
* `losses.py` : contains all the custom loss functions.
* `Neural_Networks.py` : contains all the model architecture.
* `data_utils.py` : includes dataset loading and partitioning operations.
* `./methods` : contains 13 federated learning algorithms, including `Feed`, `Ditto`, `FedAvg`, `FedBabu`, `FedMD`, `FedPer`, `FedPhp`, `FedProc`, `FedProto`, `FedProx`, `FedRep`, `LGFedAvg` and `MOON`.
* `./conf` : stores model and hyperparameter configuration files for the three tasks under IID and Non-IID cases.

### Running Instructions

The running command includes two parameters: `--conf` and `--method`. 
The `--conf` parameter is used to set the path for the program's hyperparameter configuration file (refer to `./conf`), while the `--method` parameter is used to specify the federated learning algorithm (refer to `./methods`).

running on `HR task`:
```
- IID case
python FEMNIST_Balanced.py --method Feed --conf conf/FEMNIST_balanced_conf.json

- Non-IID case
python FEMNIST_Imbalanced.py --method Feed --conf conf/FEMNIST_imbalanced_conf.json
```

running on `OC task`:
```
- IID case
python CIFAR_Balanced.py --method Feed --conf conf/CIFAR_balanced_conf.json

- Non-IID case
python CIFAR_Imbalanced.py --method Feed --conf conf/CIFAR_imbalanced_conf.json
```

running on `SA task`:
```
- IID case
python YELP_Balanced.py --method Feed --conf conf/YELP_balanced_conf.json

- Non-IID case
python YELP_Imbalanced.py --method Feed --conf conf/YELP_imbalanced_conf.json
```
