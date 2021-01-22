#!/usr/bin/env python
# coding: utf-8

import os
import logging
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Conv1D
from tensorflow.keras.optimizers import Adam

logging.basicConfig(filename='train.log', level=logging.DEBUG)

class TadGAN():

    def __init__(self, dataset, encoder_hidden_units=100, decoder_hidden_unit=64, input_shape=(100, 1)):
        # The below setting of loading the model allows to train the model over multiple sessions.
        self.dataset = dataset
        self.train_dataset = self.make_train_dataset()
        self.test_dataset = self.dataset

        if len(os.listdir('Model')) >= 4:
            self.encoder = tf.keras.models.load_model('Model/encoder.h5')
            self.decoder = tf.keras.models.load_model('Model/decoder.h5')
            self.critic_x = tf.keras.models.load_model('Model/critic_x.h5')
            self.critic_z = tf.keras.models.load_model('Model/critic_z.h5')
        else:
            self.encoder, self.decoder = self.make_generator(encoder_hidden_units, decoder_hidden_unit, input_shape)
            self.critic_x, self.critic_z = self.make_critic(input_shape)

        self.critic_x_optim = Adam(lr=0.00005)
        self.critic_z_optim = Adam(lr=0.00005)

        self.enc_optim = Adam(lr=0.00005)
        self.dec_optim = Adam(lr=0.00005)


    def make_train_dataset(self):
        l = list()
        for i in self.dataset.rolled_signal.values:
            l.append(tf.convert_to_tensor(i, tf.float32))
        dataset = tf.data.Dataset.from_tensor_slices((np.asarray(l))).batch(batch_size=16)
        
        return dataset

    #The length of data in a sequence should be of shape 100.
    def make_generator(self, encoder_hidden_units, decoder_hidden_unit, input_shape):
        encoder = Sequential()
        encoder.add(Bidirectional(LSTM(encoder_hidden_units), input_shape=input_shape))
        encoder.add(Dense(20))

        decoder = Sequential()
        #To do: Try return_sequences=False
        decoder.add(Bidirectional(LSTM(decoder_hidden_unit, return_sequences=True), input_shape=(20, 1)))
        decoder.add(Bidirectional(LSTM(decoder_hidden_unit, dropout=0.4)))
        decoder.add(Dense(100))

        return encoder, decoder

    def make_critic(self, input_shape):
        #The activation in original implementation of paper is linear. For some reason, linear activation function leads to diverging loss.
        #Hence, sigmoid activation function is used in both critics.
        critic_x =  Sequential()
        critic_x.add(Conv1D(filters=1, kernel_size=100, input_shape=input_shape))
        critic_x.add(Dense(units=1, activation='sigmoid'))

        critic_z =  Sequential()
        critic_z.add(Conv1D(filters=1, kernel_size=20, input_shape=(20, 1)))
        critic_z.add(Dense(units=1, activation='sigmoid'))

        return critic_x, critic_z


    def wasserstein_loss(self, y_true, y_pred):
        """
        y_true = +1 for real data points
        y_true = -1 for generated data points
        """
        return K.mean(y_true * y_pred, axis=-1)

    def random_weighted_average(self, x, x_):
        alpha = K.random_uniform((x.shape[0], 1, 1))
        return (alpha * x) + ((1 - alpha) * x_)

    def train(self, n_epochs=1000):
        for epoch in range(n_epochs):
            logging.debug('Epoch {}'.format(epoch))
            n_critics = 5
            self.encoder.trainable = False
            self.decoder.trainable = False
            for i in range(n_critics):
                for x in self.train_dataset:
                    self.critic_x.trainable = True
                    self.critic_z.trainable = False

                    #Sampling z from random
                    z = tf.random.uniform(shape=(x.shape[0], 20, 1), minval=-1, maxval=1, dtype=tf.float32, seed=10)
                    with tf.GradientTape(persistent=True) as tape_cx:
                        #Critic score for real samples : When trained should be 1
                        x = tf.expand_dims(x, axis=2)
                        valid_x = self.critic_x(x)
                        wl1 = self.wasserstein_loss(tf.ones_like(valid_x), valid_x)

                        #Generating critic score for noise samples : When trained should be 0
                        x_ = self.decoder(z)
                        x_ = tf.expand_dims(x_, axis=2)
                        fake_x = self.critic_x(x_)
                        wl2 = self.wasserstein_loss(-tf.ones_like(fake_x), fake_x)

                        #Calculating gradient penalty loss
                        x = tf.cast(x, tf.float32)
                        x_ = tf.cast(x, tf.float32)

                        ix = self.random_weighted_average(x, x_) #ix - interpolated_x
                        ix = tf.Variable(ix)
                        v_ix = self.critic_x(ix)

                        gradient = tape_cx.gradient(v_ix, ix)
                        gp_loss_x = K.sqrt(K.sum(K.batch_flatten(K.square(gradient)), axis=1, keepdims=True))-1

                        #wl1 + wl2 together is the wasserstein loss.
                        loss = wl1 + wl2 + gp_loss_x
                        loss = tf.reduce_mean(loss)
                        
                    gradients = tape_cx.gradient(loss, self.critic_x.trainable_variables)
                    self.critic_x_optim.apply_gradients(zip(gradients, self.critic_x.trainable_variables))
                    loss_critic_x = loss
                    
                    self.critic_x.trainable = False
                    self.critic_z.trainable = True
                    with tf.GradientTape(persistent=True) as tape_cz:
                        valid_z = self.critic_z(z)
                        wl1 = self.wasserstein_loss(tf.ones_like(valid_z), valid_z)

                        z_ = self.encoder(x)
                        z_ = tf.expand_dims(z_, axis=2)
                        fake_z = self.critic_z(z_)
                        wl2 = self.wasserstein_loss(-tf.ones_like(fake_z), fake_z)

                        iz = self.random_weighted_average(z, z_)
                        iz = tf.Variable(iz)
                        v_iz = self.critic_z(iz)

                        gradient = tape_cz.gradient(v_iz, iz)
                        gp_loss_z = K.sqrt(K.sum(K.batch_flatten(K.square(gradient)), axis=1, keepdims=True))-1

                        loss = wl1 + wl2 + gp_loss_z
                        loss = tf.reduce_mean(loss)

                    gradients = tape_cz.gradient(loss, self.critic_z.trainable_variables)
                    self.critic_z_optim.apply_gradients(zip(gradients, self.critic_z.trainable_variables))
                    loss_critic_z = loss

            self.critic_z.trainable = False
            self.critic_x.trainable = False
            self.encoder.trainable = True
            self.decoder.trainable = True

            for x in self.train_dataset:
                z = tf.random.uniform(shape=(x.shape[0], 20, 1), minval=-1, maxval=1, dtype=tf.float32, seed=10)
                x = tf.expand_dims(x, axis=2)
                with tf.GradientTape(persistent=True) as tape:
                    valid_x = self.critic_x(x)
                    wl1 = self.wasserstein_loss(tf.ones_like(valid_x), valid_x)

                    x_ = self.decoder(z)
                    x_ = tf.expand_dims(x_, axis=2)
                    fake_x = self.critic_x(x_)
                    wl2 = self.wasserstein_loss(-tf.ones_like(fake_x), fake_x)

                    valid_z = self.critic_z(z)
                    wl3 = self.wasserstein_loss(tf.ones_like(valid_z), valid_z)

                    z_ = self.encoder(x)
                    z_ = tf.expand_dims(z_, axis=2)
                    fake_z = self.critic_z(z_)
                    wl4 = self.wasserstein_loss(-tf.ones_like(fake_z), fake_z)

                    enc_z = self.encoder(x)
                    enc_z = tf.expand_dims(enc_z, axis=2)
                    gen_x = self.decoder(enc_z)
                    x = tf.squeeze(x)
                    sqd_diff = tf.keras.losses.MSE(x, gen_x)
                    loss = sqd_diff + wl1 + wl2 + wl3 + wl4
                    loss = tf.reduce_mean(loss)
                grad_e = tape.gradient(loss, self.encoder.trainable_variables)
                grad_d = tape.gradient(loss, self.decoder.trainable_variables)
                self.enc_optim.apply_gradients(zip(grad_e, self.encoder.trainable_variables))
                self.dec_optim.apply_gradients(zip(grad_d, self.decoder.trainable_variables))
                enc_dec_loss = loss


            logging.debug('critic x loss {} critic y loss {} enc-dec loss {}'.format(loss_critic_x, loss_critic_z, enc_dec_loss))

            if (epoch % 10 == 0):
                self.encoder.save('Model/encoder.h5')
                self.decoder.save('Model/decoder.h5')

                self.critic_x.save('Model/critic_x.h5')
                self.critic_z.save('Model/critic_z.h5')



    def dtw_reconstruction_error(self, x, x_):
    #Other error metrics - point wise difference, Area difference.
        n, m = x.shape[0], x_.shape[0]
        dtw_matrix = np.zeros((n+1, m+1))
        for i in range(n+1):
            for j in range(m+1):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(x[i-1] - x_[j-1])
                # take last min from a square box
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix[n][m]

    def unroll_signal(self, x):
        x = np.array(x).reshape(100)
        return np.median(x)

    def test(self):
        """
        Returns a dataframe with original value, reconstructed value, reconstruction error, critic score
        """
        df = self.test_dataset.copy()
        X_ = list()

        RE = list()  #Reconstruction error
        CS = list()  #Critic score

        for i in range(0, df.shape[0]):
            x = df.rolled_signal[i]
            x = tf.reshape(x, (1, 100, 1))
            z = encoder(x)
            z = tf.expand_dims(z, axis=2)
            x_ = decoder(z)

            re = dtw_reconstruction_error(tf.squeeze(x_).numpy(), tf.squeeze(x).numpy()) #reconstruction error
            cs = critic_x(x)
            cs = tf.squeeze(cs).numpy()
            RE.append(re)
            CS.append(cs)

            x_ = unroll_signal(x_)

            X_.append(x_)

        df['generated_signals'] = X_

        return df
