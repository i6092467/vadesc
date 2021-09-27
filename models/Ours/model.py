import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os

from tensorflow.keras import layers

from models.Ours.utils import tensor_slice, weibull_log_pdf, weibull_scale

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


class Encoder(layers.Layer):
    def __init__(self, encoded_size):
        super(Encoder, self).__init__(name='encoder')
        self.dense1 = tfkl.Dense(500, activation='relu')
        self.dense2 = tfkl.Dense(500, activation='relu')
        self.dense3 = tfkl.Dense(2000, activation='relu')
        self.mu = tfkl.Dense(encoded_size, activation=None)
        self.sigma = tfkl.Dense(encoded_size, activation=None)

    def call(self, inputs):
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
            print("yeah")
            self.dense4 = tfkl.Dense(self.inp_shape, activation="sigmoid")
        else:
            self.dense4 = tfkl.Dense(self.inp_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


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


class GMM_Survival(tf.keras.Model):
    def __init__(self, **kwargs):
        super(GMM_Survival, self).__init__(name="GMM_Survival")
        self.encoded_size = kwargs['latent_dim']
        self.num_clusters = kwargs['num_clusters']
        self.inp_shape = kwargs['inp_shape']
        self.activation = kwargs['activation']
        self.survival = kwargs['survival']
        self.s = kwargs['monte_carlo']
        self.sample_surv = kwargs['sample_surv']
        self.learn_prior = kwargs['learn_prior']
        if self.inp_shape <= 100:
            self.encoder = Encoder_small(self.encoded_size)
            self.decoder = Decoder_small(self.inp_shape, self.activation)
        else:
            self.encoder = Encoder(self.encoded_size)
            self.decoder = Decoder(self.inp_shape, self.activation)
        self.c_mu = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size]), name='mu')
        self.log_c_sigma = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size]), name='sigma') #tf.Variable(tf.ones([self.num_clusters, self.encoded_size]), name="sigma")
        # Cluster-specific survival model parameters
        self.c_beta = tf.Variable(tf.initializers.GlorotNormal()(shape=[self.num_clusters, self.encoded_size + 1]),
                                  name='beta')
        # Weibull distribution shape parameter
        self.weibull_shape = kwargs['weibull_shape']
        if self.learn_prior:
            self.prior_logits = tf.Variable(tf.ones([self.num_clusters]), name="prior")
        else:
            self.prior = tf.constant(tf.ones([self.num_clusters]) * (1 / self.num_clusters))
        self.use_t = tf.Variable([1.0], trainable=False)

    def call(self, inputs, training=True):
        # NB: inputs have to include predictors/covariates/features (x), time-to-event (t), and
        # event indicators (d). d[i] == 1 if the i-th event is a death, and d[i] == 0 otherwise.
        x, y = inputs
        t = y[:, 0]
        d = y[:, 1]
        enc_input = x
        z_mu, log_z_sigma = self.encoder(enc_input)
        tf.debugging.check_numerics(z_mu, message="z_mu")
        z = tfd.MultivariateNormalDiag(loc=z_mu, scale_diag=tf.math.sqrt(tf.math.exp(log_z_sigma)))
        if training:
            z_sample = z.sample(self.s)
        else:
            z_sample = tf.expand_dims(z_mu, axis=0)
        tf.debugging.check_numerics(self.c_mu, message="c_mu")
        tf.debugging.check_numerics(self.log_c_sigma, message="c_sigma")
        c_sigma = tf.math.exp(self.log_c_sigma)
        p_z_c = tf.stack([tf.math.log(
            tfd.MultivariateNormalDiag(loc=tf.cast(self.c_mu[i, :], tf.float64),
                                       scale_diag=tf.math.sqrt(tf.cast(c_sigma[i, :], tf.float64))).prob(
                tf.cast(z_sample, tf.float64)) + 1e-60) for i in range(self.num_clusters)], axis=-1)
        tf.debugging.check_numerics(p_z_c, message="p_z_c")

        # Cluster prior
        if self.learn_prior:
            prior_logits = tf.math.abs(self.prior_logits)
            norm = tf.math.reduce_sum(prior_logits, keepdims=True)
            prior = prior_logits / (norm + tf.keras.backend.epsilon())
        else:
            prior = self.prior
        tf.debugging.check_numerics(prior, message="prior")

        # Survival model
        if self.survival:
            tf.debugging.check_numerics(self.c_beta, message="c_beta")
            if self.sample_surv:
                lambda_z_c = tf.stack([weibull_scale(x=z_sample, beta=self.c_beta[i, :])
                                        for i in range(self.num_clusters)], axis=-1)
            else:
                lambda_z_c = tf.stack([weibull_scale(x=tf.stack([z_mu for i in range(self.s)], axis=0),
                                                     beta=self.c_beta[i, :]) for i in range(self.num_clusters)], axis=-1)
            tf.debugging.check_numerics(lambda_z_c, message="lambda_z_c")
            # Evaluate p(t|z,c), assuming t|z,c ~ Weibull(lambda_z_c, self.weibull_shape)
            p_t_z_c = tf.stack([weibull_log_pdf(t=t, d=d, lmbd=lambda_z_c[:, :, i], k=self.weibull_shape)
                                for i in range(self.num_clusters)], axis=-1)
            tf.debugging.check_numerics(p_t_z_c, message="p_t_z_c")
            p_c_z = tf.math.log(tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64) + p_t_z_c
        else:
            p_c_z = tf.math.log(tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64)
        p_c_z = tf.nn.log_softmax(p_c_z, axis=-1)
        p_c_z = tf.math.exp(p_c_z)
        tf.debugging.check_numerics(p_c_z, message="p_c_z")

        if self.survival:
            loss_survival = -tf.reduce_sum(tf.multiply(p_t_z_c, tf.cast(p_c_z, tf.float64)), axis=-1)
            tf.debugging.check_numerics(loss_survival, message="loss_survival")

        loss_clustering = - tf.reduce_sum(tf.multiply(tf.cast(p_c_z, tf.float64), tf.cast(p_z_c, tf.float64)),
                                                  axis=-1)

        loss_prior = - tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64), 1e-60 +
                                                                tf.cast(prior, tf.float64)), axis=-1)

        loss_variational_1 = - 1 / 2 * tf.reduce_sum(log_z_sigma + 1, axis=-1)

        loss_variational_2 = tf.math.reduce_sum(tf.math.xlogy(tf.cast(p_c_z, tf.float64),
                                                                      1e-60 + tf.cast(p_c_z, tf.float64)), axis=-1)

        tf.debugging.check_numerics(loss_clustering, message="loss_clustering")
        tf.debugging.check_numerics(loss_prior, message="loss_prior")
        tf.debugging.check_numerics(loss_variational_1, message="loss_variational_1")
        tf.debugging.check_numerics(loss_variational_2, message="loss_variational_2")

        if self.survival:
            self.add_loss(tf.math.reduce_mean(loss_survival))
        self.add_loss(tf.math.reduce_mean(loss_clustering))
        self.add_loss(tf.math.reduce_mean(loss_prior))
        self.add_loss(tf.math.reduce_mean(loss_variational_1))
        self.add_loss(tf.math.reduce_mean(loss_variational_2))

        # Logging metrics in TensorBoard
        self.add_metric(loss_clustering, name='loss_clustering', aggregation="mean")
        self.add_metric(loss_prior, name='loss_prior', aggregation="mean")
        self.add_metric(loss_variational_1, name='loss_variational_1', aggregation="mean")
        self.add_metric(loss_variational_2, name='loss_variational_2', aggregation="mean")
        if self.survival:
            self.add_metric(loss_survival, name='loss_survival', aggregation="mean")

        dec = self.decoder(z_sample)

        # Evaluate risk scores based on hard clustering assignments
        # Survival time may ba unobserved, so a special procedure is needed when time is not observed...
        p_z_c = p_z_c[0]    # take the first sample
        p_c_z = p_c_z[0]
        if self.use_t[0] < 0.5:
            if self.survival:
                lambda_z_c = lambda_z_c[0]  # Take the first sample
                # Use Bayes rule to compute p(c|z) instead of p(c|z,t), since t is unknown
                p_c_z = tf.math.log(tf.cast(prior, tf.float64) + 1e-60) + tf.cast(p_z_c, tf.float64)
                p_c_z = tf.nn.log_softmax(p_c_z, axis=-1)
                p_c_z = tf.math.exp(p_c_z)

                inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
                risk_scores = tensor_slice(target_tensor=tf.cast(lambda_z_c, tf.float64), index_tensor=inds)
            else:
                inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
                risk_scores = tensor_slice(target_tensor=p_c_z, index_tensor=inds)
        else:
            inds = tf.dtypes.cast(tf.argmax(p_c_z, axis=-1), tf.int32)
            if self.survival:
                lambda_z_c = lambda_z_c[0]  # Take the first sample
                risk_scores = tensor_slice(target_tensor=lambda_z_c, index_tensor=inds)
            else:
                risk_scores = tensor_slice(target_tensor=p_c_z, index_tensor=inds)

        p_z_c = tf.cast(p_z_c, tf.float64)

        # NB: transposition to avoid issues in model.predict
        dec = tf.transpose(dec, [1, 0, 2])
        z_sample = tf.transpose(z_sample, [1, 0, 2])

        return dec, z_sample, p_z_c, p_c_z, risk_scores

    def generate_samples(self, j, n_samples):
        z = tfd.MultivariateNormalDiag(loc=self.c_mu[j, :], scale_diag=tf.math.sqrt(tf.math.exp(self.log_c_sigma[j, :])))
        z_sample = z.sample(n_samples)
        dec = self.decoder(z_sample)
        return dec
