import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from sklearn.utils.linear_assignment_ import linear_assignment

import numpy as np

from scipy.stats import weibull_min, fisk

import sys
sys.path.insert(0, '../../')
from utils.constants import ROOT_LOGGER_STR

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


def setup_logger(results_path, create_stdlog):
    """Setup a general logger which saves all logs in the experiment folder"""

    f_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler = logging.FileHandler(str(results_path))
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    root_logger = logging.getLogger(ROOT_LOGGER_STR)
    root_logger.handlers = []
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(f_handler)

    if create_stdlog:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        root_logger.addHandler(handler)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.astype(int).max(), y_true.astype(int).max()) + 1
    w = np.zeros((int(D), (D)), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def sample_weibull(scales, shape, n_samples=200):
    return np.transpose(weibull_min.rvs(shape, loc=0, scale=scales, size=(n_samples, scales.shape[0])))


def save_mnist_reconstructions(recs, x, y):
    labels = y[:, 2]
    unique_labels = np.unique(labels)

    imgs_sampled = []
    recs_sampled = []
    for l in unique_labels:
        recs_l = recs[labels == l, :, :]
        x_l = x[labels == l, :]
        y_l = y[labels == l]
        j = np.random.randint(0, len(y_l))
        imgs_sampled.append(np.reshape(x_l[j, :], (28, 28)))
        recs_sampled.append(np.reshape(recs_l[j, 0, :], (28, 28)))
    imgs_cat = np.concatenate(imgs_sampled, axis=1)
    recs_cat = np.concatenate(recs_sampled, axis=1)
    img_final = np.concatenate([imgs_cat, recs_cat], axis=0)
    plt.imsave("recs.png", img_final)


def save_mnist_generated_samples(model, grid_size=4):
    for j in range(model.num_clusters):
        samples = model.generate_samples(j=j, n_samples=grid_size**2)
        cnt = 0
        img = None
        for k in range(grid_size):
            row_k = []
            for l in range(grid_size):
                row_k.append(np.reshape(samples[cnt, :], (28, 28)))
                cnt = cnt + 1
            if img is None:
                img = np.concatenate(row_k, axis=1)
            else:
                img = np.concatenate([img, np.concatenate(row_k, axis=1)], axis=0)
        plt.imsave("generated_" + str(j) + ".png", img)


def save_generated_samples(model, inp_size, grid_size=4, cmap='viridis', postfix=None):
    for j in range(model.num_clusters):
        samples = model.generate_samples(j=j, n_samples=grid_size**2)
        cnt = 0
        img = None
        for k in range(grid_size):
            row_k = []
            for l in range(grid_size):
                row_k.append(np.reshape(samples[0, cnt, :], (inp_size[0], inp_size[1])))
                cnt = cnt + 1
            if img is None:
                img = np.concatenate(row_k, axis=1)
            else:
                img = np.concatenate([img, np.concatenate(row_k, axis=1)], axis=0)
        if postfix is not None:
            plt.imsave("generated_" + str(j) + "_" + postfix + ".png", img, cmap=cmap)
        else:
            plt.imsave("generated_" + str(j) + ".png", img, cmap=cmap)


# Weibull(lmbd, k) log-pdf
def weibull_log_pdf(t, d, lmbd, k):
    t_ = tf.ones_like(lmbd) * tf.cast(t, tf.float64)
    d_ = tf.ones_like(lmbd) * tf.cast(d, tf.float64)
    k = tf.cast(k, tf.float64)
    a = t_ / (1e-60 + tf.cast(lmbd, tf.float64))
    tf.debugging.check_numerics(a, message="weibull_log_pdf")

    return tf.cast(d_, tf.float64) * (tf.math.log(1e-60 + k) - tf.math.log(1e-60 + tf.cast(lmbd, tf.float64)) +
                                     (k - 1) * tf.math.log(1e-60 + tf.cast(t_, tf.float64)) - (k - 1) *
                                     tf.math.log(1e-60 + tf.cast(lmbd, tf.float64))) - (a) ** k


def weibull_scale(x, beta):
    beta_ = tf.cast(beta, tf.float64)
    beta_ = tf.cast(tf.ones([tf.shape(x)[0], tf.shape(x)[1], beta.shape[0]]), tf.float64) * beta_
    return tf.clip_by_value(tf.math.log(1e-60 + 1.0 + tf.math.exp(tf.reduce_sum(-tf.cast(x, tf.float64) * beta_[:, :, :-1], axis=2) -
                                                 tf.cast(beta[-1], tf.float64))), -1e+64, 1e+64)


def sample_weibull_mixture(scales, shape, p_c, n_samples=200):
    scales_ = np.zeros((scales.shape[0], n_samples))
    cs = np.zeros((scales.shape[0], n_samples)).astype(int)
    for i in range(scales.shape[0]):
        cs[i] = np.random.choice(a=np.arange(0, p_c.shape[1]), p=p_c[i], size=(n_samples,))
        scales_[i] = scales[i, cs[i]]
    return scales_ * np.random.weibull(shape, size=(scales.shape[0], n_samples))


def tensor_slice(target_tensor, index_tensor):
    indices = tf.stack([tf.range(tf.shape(index_tensor)[0]), index_tensor], 1)
    return tf.gather_nd(target_tensor, indices)