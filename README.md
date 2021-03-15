# Clustering-friendly Representation Learning via Instance Discrimination and Feature Decorrelation 

This repository contains the Pytorch implementation of [our paper(IDFD)](https://openreview.net/pdf?id=e12NDM7wkEY).
The implementation can reproduce the main result on CIFAR10. 

if you found our work useful in your work, please cite:
```
@article{tao2021idfd,
title = {Clustering-friendly Representation Learning via Instance Discrimination and Feature Decorrelation},
author = {Yaling Tao, Kentaro Takagi, Kouta Nakata},
year = {2021},
journal = {Proceedings of ICLR 2021},
url = {https://openreview.net/forum?id=e12NDM7wkEY}
}
```

# How to train

## Setup environment

Install packages as:

```shell
pip install -r requirements.txt
```

## Training

To train IDFD on GPU-0, run main script as:
```shell
./main.py --gpus 0
```

