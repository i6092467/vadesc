"""
Loss functions for the reconstruction term of the ELBO.
"""
import tensorflow as tf


class Losses:
    def __init__(self, configs):
        self.input_dim = configs['training']['inp_shape']
        self.tuple = False
        if isinstance(self.input_dim, list):
            print("\nData is tuple!\n")
            self.tuple = True
            self.input_dim = self.input_dim[0] * self.input_dim[1]

    def loss_reconstruction_binary(self, inp, x_decoded_mean):
        x = inp
        # NB: transpose to make the first dimension correspond to MC samples
        if self.tuple:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2, 3])
        else:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
        loss = self.input_dim * tf.math.reduce_mean(tf.stack([tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean[i])
                                                              for i in range(x_decoded_mean.shape[0])], axis=-1),
                                                    axis=-1)
        return loss

    def loss_reconstruction_mse(self, inp, x_decoded_mean):
        x = inp
        # NB: transpose to make the first dimension correspond to MC samples
        if self.tuple:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2, 3])
        else:
            x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
        loss = self.input_dim * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                                              for i in range(x_decoded_mean.shape[0])], axis=-1),
                                                    axis=-1)
        return loss
