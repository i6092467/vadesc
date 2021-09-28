# Cox PH model
import numpy as np

from lifelines import CoxPHFitter

import sys
sys.path.insert(0, '../../')
from utils.data_utils import construct_surv_df
from utils.eval_utils import cindex


def fit_coxph(X: np.ndarray, t: np.ndarray, d: np.ndarray, penalty_weight=0.0, X_test=None, t_test=None, d_test=None):
    """
    Fits and evaluates a Cox proportional hazards (PH) model on the provided data. A wrapper function for the lifelines
    CoxPHFitter.

    :param X: predictor variables [n_samples, n_features].
    :param t: time-to-event.
    :param d: labels of the type of event observed. d[i] == 1, if the i-th event is failure (death); d[i] == 0 otherwise.
    :param penalty_weight: weight of the penalty term in the Cox regression, 0.0 by default. Hint: use a non-zero
    penalty weight for strongly correlated features.
    :param X_test: test set predictor variables.
    :param t_test: test set time-to-event.
    :param d_test: test set labels of the type of event observed.
    :return: returns the fitted Cox PH model, predicted hazard function values, and the concordance index on the train
    set. If applicable, returns hazard scores and the concordance index on the test data as well.
    """
    df = construct_surv_df(X, t, d)
    cph = CoxPHFitter(penalizer=penalty_weight)
    cph.fit(df, duration_col='time_to_event', event_col='failure')
    hazard_train = np.exp(-np.squeeze(np.matmul(X, np.expand_dims(cph.params_, 1))))
    ci_train = cindex(t=t, d=d, scores_pred=hazard_train)

    if X_test is not None:
        assert t_test is not None and d_test is not None
        hazard_test = np.exp(-np.squeeze(np.matmul(X_test, np.expand_dims(cph.params_, 1))))
        ci_test = cindex(t=t_test, d=d_test, scores_pred=hazard_test)
        return cph, hazard_train, ci_train, hazard_test, ci_test

    return cph, hazard_train, ci_train
