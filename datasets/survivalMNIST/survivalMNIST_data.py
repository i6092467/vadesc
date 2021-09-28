# Based on Pölsterl's survival MNIST dataset:
#       https://k-d-w.org/blog/2019/07/survival-analysis-for-deep-learning/

import numpy as np
from numpy.random import choice, uniform, normal, exponential
import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist


def load_MNIST(split: str, flatten=True):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    assert split == "train" or split == "test"

    # Flatten
    if flatten:
        train_X = train_X.reshape((train_X.shape[0], train_X.shape[1] * train_X.shape[2]))
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[1] * test_X.shape[2]))

    if split == "train":
        return train_X, train_y
    else:
        return test_X, test_y


def generate_surv_MNIST(n_groups: int, seed: int, p_cens: float, risk_range=[0.5, 15.0], risk_stdev=0.00, valid_perc=.05):
    assert 2 <= n_groups <= 10
    assert risk_range[0] < risk_range[1]

    # Replicability
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_X, labels_train = load_MNIST(split="train")
    test_X, labels_test = load_MNIST(split="test")

    # Cluster assignments of digits
    c0 = choice(np.arange(n_groups), replace=False, size=(n_groups,))
    c1 = np.array([])
    if 10 - n_groups > 0:
        c1 = choice(np.arange(n_groups), replace=True, size=(10 - n_groups,))
    c = np.concatenate((c0, c1))
    np.random.shuffle(c)

    # Risk scores
    r_scores = uniform(risk_range[0], risk_range[1], size=(n_groups,))
    r_scores = normal(r_scores[c], risk_stdev)

    print("-" * 50)
    print("Cluster Assignments & Risk Scores:")
    print("Digit:       " + str(np.arange(10)))
    print("Risk group:  " + str(c))
    print("Risk score:  " + str(r_scores))
    print("-" * 50)
    print()
    print()

    r_scores_train = r_scores[labels_train]
    r_scores_test = r_scores[labels_test]

    stg_train = SurvivalTimeGenerator(num_samples=train_X.shape[0], mean_survival_time=150., prob_censored=p_cens)
    t_train, d_train = stg_train.gen_censored_time(r_scores_train)
    stg_test = SurvivalTimeGenerator(num_samples=test_X.shape[0], mean_survival_time=150., prob_censored=p_cens)
    t_test, d_test = stg_test.gen_censored_time(r_scores_test)

    c_train = c[labels_train]
    c_test = c[labels_test]

    t_train = t_train / max([np.max(t_train), np.max(t_test)]) + 0.001
    t_test = t_test / max([np.max(t_train), np.max(t_test)]) + 0.001

    if valid_perc > 0:
        n_valid = int(valid_perc * (train_X.shape[0] + test_X.shape[0]))
        shuffled_idx = np.arange(0, train_X.shape[0])
        np.random.shuffle(shuffled_idx)
        train_idx = shuffled_idx[0:(shuffled_idx.shape[0] - n_valid)]
        valid_idx = shuffled_idx[(shuffled_idx.shape[0] - n_valid):]

        c_train_ = c_train[train_idx]
        c_valid = c_train[valid_idx]
        c_train = c_train_

        return train_X[train_idx, :], train_X[valid_idx, :], test_X, \
               t_train[train_idx], t_train[valid_idx], t_test, \
               d_train[train_idx], d_train[valid_idx], d_test, \
               c_train, c_valid, c_test
    else:
        return train_X, test_X, t_train, t_test, d_train, d_test, c_train, c_test


class SurvivalTimeGenerator:
    def __init__(self, num_samples: int, mean_survival_time: float, prob_censored: float):
        self.num_samples = num_samples
        self.mean_survival_time = mean_survival_time
        self.prob_censored = prob_censored

    def gen_censored_time(self, risk_score: np.ndarray, seed: int = 89):
        rnd = np.random.RandomState(seed)
        # generate survival time
        baseline_hazard = 1. / self.mean_survival_time
        scale = baseline_hazard * np.exp(risk_score)
        u = rnd.uniform(low=0, high=1, size=risk_score.shape[0])
        t = -np.log(u) / scale

        # generate time of censoring
        qt = np.quantile(t, 1.0 - self.prob_censored)
        c = rnd.uniform(low=t.min(), high=qt)

        # apply censoring
        observed_event = t <= c
        observed_time = np.where(observed_event, t, c)
        return observed_time, observed_event
