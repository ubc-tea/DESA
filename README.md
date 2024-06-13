# DESA
The official implementation of paper "Overcoming Data and Model heterogeneities in Decentralized Federated Learning via Synthetic Anchors" accepted at ICML 2024

## How to run

### System requirement

We recommend using conda to install the environment.
Please use [environment.txt](https://github.com/ubc-tea/DESA/blob/main/environment.txt) to set up the conda environment.

### Download DIGITS data

Please download the digits data [here](https://drive.google.com/drive/folders/1s_QRtmLG6ibUlycMjUeSsqy4pwaqdi7o?usp=sharing) and put it under digit_data folder.

### Run with pretrained models

```
python iterative_desab.py --dataset=digits --ipc=50 --model_hetero=True
```