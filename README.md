# DESA
The official implementation of paper "[Overcoming Data and Model heterogeneities in Decentralized Federated Learning via Synthetic Anchors](https://arxiv.org/abs/2405.11525)" accepted at ICML 2024

![DeSA_main_figure](/img/DeSA_pipeline.png)

## Abstract

Conventional Federated Learning (FL) involves collaborative training of a global model while maintaining user data privacy. One of its branches, decentralized FL, is a serverless network that allows clients to own and optimize different local models separately, which results in saving management and communication resources. Despite the promising advancements in decentralized FL, it may reduce model generalizability due to lacking a global model. In this scenario, managing data and model heterogeneity among clients becomes a crucial problem, which poses a unique challenge that must be overcome: *How can every client's local model learn generalizable representation in a decentralized manner?* To address this challenge, we propose a novel **De**centralized FL technique by introducing **S**ynthetic **A**nchors, dubbed as DeSA. Based on the theory of domain adaptation and Knowledge Distillation (KD), we theoretically and empirically show that synthesizing global anchors based on raw data distribution facilitates mutual knowledge transfer. We further design two effective regularization terms for local training: 1) *REG loss* that regularizes the distribution of the client's latent embedding with the anchors and 2) *KD loss* that enables clients to learn from others. Through extensive experiments on diverse client data distributions, we showcase the effectiveness of DeSA in enhancing both inter- and intra-domain accuracy of each client.

![loss_effect](/img/loss_effect.png)

## System requirement

We recommend using conda to install the environment.
Please use [environment.txt](https://github.com/ubc-tea/DESA/blob/main/environment.txt) to set up the conda environment.

## Verify pretrained DIGITS model


### Download DIGITS data

Please download the digits data [here](https://drive.google.com/drive/folders/1s_QRtmLG6ibUlycMjUeSsqy4pwaqdi7o?usp=sharing) and put it under digit_data folder.

### Run with pretrained models

```
python iterative_desab.py --dataset=digits --ipc=50 --model_hetero=True
```

## How to run - OFFICE and CIFAR10C experiments

### Download data

Please download the preprocessed data and put them into data folder. [OFFICE](https://drive.google.com/drive/folders/1fALcd1iYzJynuy4imc0zqFsGbGf5kVFw?usp=sharing) [CIFAR10C](https://drive.google.com/drive/folders/1BIBvskSH-gbt7s50fRrJO5Rld1XXqCbb?usp=sharing)

### Run experiments

Please use the following scripts to run the experiments from scratch

```
python iterative_desab.py --dataset=office --ipc=10 --model_hetero=True --pretrain=True --generate_image=True --KD=True
python iterative_desab.py --dataset=cifar10c --ipc=50 --model_hetero=True --pretrain=True --generate_image=True --KD=True --client_ratio=0.1
python iterative_desab.py --dataset=cifar10c --ipc=10 --model_hetero=True --pretrain=True --generate_image=True --KD=True --client_ratio=0.2
```

## DP experiments

Currently, DeSA only supports DIGITS experiment.

### Run experiments

Please use the following scripts to run the experiment from scratch

```
python iterative_desab.py --dataset=digits --ipc=10 --model_hetero=True --pretrain=True --generate_image=True --KD=True --DP=True
```

## MIA experiments

![mia](/img/mia.png)

For our MIA experiment, we follow the metric from [Carlini et al.](https://arxiv.org/abs/2112.03570).

## Citation
If you find this work helpful, please cite our paper as follows:
```
@article{huang2024overcoming,
  title={Overcoming Data and Model Heterogeneities in Decentralized Federated Learning via Synthetic Anchors},
  author={Huang, Chun-Yin and Srinivas, Kartik and Zhang, Xin and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2405.11525},
  year={2024}
}
```