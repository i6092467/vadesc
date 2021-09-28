# Some utility functions for model comparison

import numpy as np

from lifelines.utils import concordance_index

import sys

sys.path.insert(0, '../')

from utils.plotting import plot_group_kaplan_meier

from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score
import tensorflow as tf

from lifelines import KaplanMeierFitter

from scipy import stats
from scipy.stats import linregress


def accuracy_metric(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(normalized_mutual_info_score, [y, y_pred], tf.float64)


def cindex_metric(inp, risk_scores):
    # Evaluates the concordance index based on provided predicted risk scores, computed using hard clustering
    # assignments.
    t = inp[:, 0]
    d = inp[:, 1]
    risk_scores = tf.squeeze(risk_scores)
    return tf.cond(tf.reduce_any(tf.math.is_nan(risk_scores)),
                   lambda: tf.numpy_function(cindex, [t, d, tf.zeros_like(risk_scores)], tf.float64),
                   lambda: tf.numpy_function(cindex, [t, d, risk_scores], tf.float64))
    # if tf.reduce_any(tf.math.is_nan(risk_scores)):
    #     Warning("NaNs in risk scores!")
    #     return tf.numpy_function(cindex, [t, d, tf.zeros_like(risk_scores)], tf.float64)
    # else:
    #     return tf.numpy_function(cindex, [t, d, risk_scores], tf.float64)


def cindex(t: np.ndarray, d: np.ndarray, scores_pred: np.ndarray):
    """
    Evaluates concordance index based on the given predicted risk scores.

    :param t: observed time-to-event.
    :param d: labels of the type of even observed. d[i] == 1, if the i-th event is failure (death); d[i] == 0 otherwise.
    :param scores_pred: predicted risk/hazard scores.
    :return: return the concordance index.
    """
    try:
        ci = concordance_index(event_times=t, event_observed=d, predicted_scores=scores_pred)
    except ZeroDivisionError:
        print('Cannot devide by zero.')
        ci = float(0.5)
    return ci


def rae(t_pred, t_true, cens_t):
    # Relative absolute error as implemented by Chapfuwa et al.
    abs_error_i = np.abs(t_pred - t_true)
    pred_great_empirical = t_pred > t_true
    min_rea_i = np.minimum(np.divide(abs_error_i, t_true + 1e-8), 1.0)
    idx_cond = np.logical_and(cens_t, pred_great_empirical)
    min_rea_i[idx_cond] = 0.0

    return np.sum(min_rea_i) / len(t_true)


def calibration(predicted_samples, t, d):
    kmf = KaplanMeierFitter()
    kmf.fit(t, event_observed=d)

    range_quant = np.arange(start=0, stop=1.010, step=0.010)
    t_empirical_range = np.unique(np.sort(np.append(t, [0])))
    km_pred_alive_prob = [kmf.predict(i) for i in t_empirical_range]
    empirical_dead = 1 - np.array(km_pred_alive_prob)

    km_dead_dist, km_var_dist, km_dist_ci = compute_km_dist(predicted_samples, t_empirical_range=t_empirical_range,
                                                            event=d)

    slope, intercept, r_value, p_value, std_err = linregress(x=km_dead_dist, y=empirical_dead)

    return slope


# Bounds
def ci_bounds(surv_t, cumulative_sq_, alpha=0.95):
    # print("surv_t: ", surv_t, "cumulative_sq_: ", cumulative_sq_)
    # This method calculates confidence intervals using the exponential Greenwood formula.
    # See https://www.math.wustl.edu/%7Esawyer/handouts/greenwood.pdf
    # alpha = 0.95
    if surv_t > 0.999:
        surv_t = 1
        cumulative_sq_ = 0
    alpha = 0.95
    constant = 1e-8
    alpha2 = stats.norm.ppf((1. + alpha) / 2.)
    v = np.log(surv_t)
    left_ci = np.log(-v)
    right_ci = alpha2 * np.sqrt(cumulative_sq_) * 1 / v

    c_plus = left_ci + right_ci
    c_neg = left_ci - right_ci

    ci_lower = np.exp(-np.exp(c_plus))
    ci_upper = np.exp(-np.exp(c_neg))

    return [ci_lower, ci_upper]


# Population wise cdf
def compute_km_dist(predicted_samples, t_empirical_range, event):
    km_dead = []
    km_surv = 1

    km_var = []
    km_ci = []
    km_sum = 0

    kernel = []
    e_event = event

    for j in np.arange(len(t_empirical_range)):
        r = t_empirical_range[j]
        low = 0 if j == 0 else t_empirical_range[j - 1]
        area = 0
        censored = 0
        dead = 0
        at_risk = len(predicted_samples)
        count_death = 0
        for i in np.arange(len(predicted_samples)):
            e = e_event[i]
            if len(kernel) != len(predicted_samples):
                kernel_i = stats.gaussian_kde(predicted_samples[i])
                kernel.append(kernel_i)
            else:
                kernel_i = kernel[i]
            at_risk = at_risk - kernel_i.integrate_box_1d(low=0, high=low)

            if e == 1:
                count_death += kernel_i.integrate_box_1d(low=low, high=r)
        if at_risk == 0:
            break
        km_int_surv = 1 - count_death / at_risk
        km_int_sum = count_death / (at_risk * (at_risk - count_death))

        km_surv = km_surv * km_int_surv
        km_sum = km_sum + km_int_sum

        km_ci.append(ci_bounds(cumulative_sq_=km_sum, surv_t=km_surv))

        km_dead.append(1 - km_surv)
        km_var.append(km_surv * km_surv * km_sum)

    return np.array(km_dead), np.array(km_var), np.array(km_ci)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Requires scikit-learn installed
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
