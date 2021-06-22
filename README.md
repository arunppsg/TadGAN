# TadGAN: Time Series Anomaly Detection Using Generative Adversarial Networks

This is a Python3 / [Pytorch](https://pytorch.org/) implementation 
of [TadGAN](https://arxiv.org/abs/2009.07769) paper. The associated blog explaining the architecture details can be found [here](https://arunpalaniappan.github.io/2021/time-series-anomaly-detection-using-GAN.html).

## Data:

The TadGAN architecture can be used for detecting anomalies in time series data.

## Pretrained Model:

The trained model is saved in the `Model` directory. The training is incomplete and the model has to be retrained for other datasets.

## Architecture:

The model implements an encoder and decoder as generator and two critics as discriminators as described in the paper. The loss function is wasserstein loss with gradient penalty.

## Usage:

1. Format of the dataset - The dataset should have a column name as `signal` containing the signals and a column with name `anomaly` containing the true labels (used during validation). 

2. Delete the contents of the directory `Model`.

3. Change the file name `exchange-2_cpc_results.csv` in `main.py` to the name of your dataset.

## Note:

This is an independent implementation and I am not related to the authors of the paper.
