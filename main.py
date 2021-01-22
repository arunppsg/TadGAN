#!/usr/bin/env python
# coding: utf-8

import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Conv1D
from tensorflow.keras.optimizers import Adam

import model

logging.basicConfig(filename='train.log', level=logging.DEBUG)

def normalize(values):
    max_value = np.max(values)
    min_value = np.min(values)
    #Normalizing between -1 and +1
    values = 2 * ((values - min_value) / (max_value - min_value)) - 1
    return (values)


def make_rolling_data(signal, window_size=100, step_size=1):
    """
    signal should be of pandas series type
    """
    original_signal = list()
    rolled_signal = list()
    ext = window_size // 2 #extended window - ext
    for i in range(ext, signal.shape[0]-ext-1, step_size):
        rolled_signal.append(signal.iloc[i-ext:i+ext].values)
        original_signal.append(signal.iloc[i])
    
    df = pd.DataFrame({"original_signal":original_signal, "rolled_signal":rolled_signal })
    return (df)


def preprocess(df):
    """
    Modify code here to make the signal suitable for TadGAN.
    """
    signal = df

    signal = normalize(signal)
    signal = make_rolling_data(signal)

    return signal

if __name__ == "__main__":

	df = pd.read_csv('exchange-2_cpc_results.csv')

	signal = df.value
	signal = preprocess(signal)

	model = model.TadGAN(dataset=signal)

	model.train(n_epochs=1000)
	
	df = model.test()
