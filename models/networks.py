"""
Encoder and decoder architectures used by VaDeSC.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import layers

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


# Wide MLP encoder and decoder architectures
class Encoder(layers.Layer):
    def __init__(self, encoded_size):
        super(Encoder, self).__init__(name='encoder')
        self.dense1 = tfkl.Dense(500, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(2000, activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):
        x = tfkl.Flatten()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(Decoder, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(2000, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(500, activation='relu')
        if activation == "sigmoid":
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


# VGG-based architectures
class VGGConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGConvBlock, self).__init__(name="VGGConvBlock{}".format(block_id))
        self.conv1 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu')
        self.conv2 = tfkl.Conv2D(filters=num_filters, kernel_size=(3, 3), activation='relu')
        self.maxpool = tfkl.MaxPooling2D((2, 2))

    def call(self, inputs, **kwargs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.maxpool(out)

        return out


class VGGDeConvBlock(layers.Layer):
    def __init__(self, num_filters, block_id):
        super(VGGDeConvBlock, self).__init__(name="VGGDeConvBlock{}".format(block_id))
        self.upsample = tfkl.UpSampling2D((2, 2), interpolation='bilinear')
        self.convT1 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')
        self.convT2 = tfkl.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), padding='valid', activation='relu')

    def call(self, inputs, **kwargs):
        out = self.upsample(inputs)
        out = self.convT1(out)
        out = self.convT2(out)

        return out


class VGGEncoder(layers.Layer):
    def __init__(self, encoded_size):
        super(VGGEncoder, self).__init__(name='VGGEncoder')
        self.layers = [VGGConvBlock(32, 1), VGGConvBlock(64, 2)]
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs, **kwargs):
        out = inputs

        # Iterate through blocks
        for block in self.layers:
            out = block(out)
        out_flat = tfkl.Flatten()(out)
        mu = self.mu(out_flat)
        sigma = self.sigma(out_flat)

        return mu, sigma


class VGGDecoder(layers.Layer):
    def __init__(self, input_shape, activation):
        super(VGGDecoder, self).__init__(name='VGGDecoder')

        target_shape = (13, 13, 64)     # 64 x 64

        self.activation = activation
        self.dense = tfkl.Dense(target_shape[0] * target_shape[1] * target_shape[2])
        self.reshape = tfkl.Reshape(target_shape=target_shape)
        self.layers = [VGGDeConvBlock(64, 1), VGGDeConvBlock(32, 2)]
        self.convT = tfkl.Conv2DTranspose(filters=input_shape[2], kernel_size=3, padding='same')

    def call(self, inputs, **kwargs):
        out = self.dense(inputs[0])
        out = self.reshape(out)

        # Iterate through blocks
        for block in self.layers:
            out = block(out)

        # Last convolution
        out = self.convT(out)

        if self.activation == "sigmoid":
            out = tf.sigmoid(out)

        return tf.expand_dims(out, 0)


# Smaller encoder and decoder architectures for low-dimensional datasets
class Encoder_small(layers.Layer):
    def __init__(self, encoded_size):
        super(Encoder_small, self).__init__(name='encoder')
        self.dense1 = tfkl.Dense(50, activation='relu')
        self.dense2 = tfkl.Dense(100, activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
        x = tfkl.Flatten()(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class Decoder_small(layers.Layer):
    def __init__(self, input_shape, activation):
        super(Decoder_small, self).__init__(name='dec')
        self.inp_shape = input_shape
        self.dense1 = tfkl.Dense(100, activation='relu')
        self.dense2 = tfkl.Dense(50, activation='relu')
        if activation == "sigmoid":
            print("yeah")
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense4(x)
        return x
