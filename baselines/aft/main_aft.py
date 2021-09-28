import argparse

import os

import numpy as np

import pandas as pd

import time

import uuid

from lifelines import WeibullAFTFitter

import sys
sys.path.insert(0, '../../')
from datasets.support.support_data import generate_support
from datasets.hgg.hgg_data import generate_hgg
from datasets.nsclc_lung.nsclc_lung_data import generate_radiomic_features

from utils.data_utils import construct_surv_df

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.eval_utils import cindex, calibration
from utils.eval_utils import rae as RAE

from utils import utils


def get_data(args, val=False):
    if args.data == 'support':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_support(seed=args.seed)
    elif args.data == "flchain":
        data = pd.read_csv('../DCM/data/flchain.csv')
        feats = ['age', 'sex', 'sample.yr', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']
        prot = 'sex'
        feats = set(feats)
        feats = list(feats - set([prot]))
        t = data['futime'].values + 1
        d = data['death'].values
        x = data[feats].values
        c = data[prot].values
        X = StandardScaler().fit_transform(x)
        t = t / np.max(t) + 0.001
        x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                              random_state=args.seed)
    elif args.data == 'hgg':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_hgg(seed=args.seed)
    elif args.data == 'nsclc':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_radiomic_features(n_slices=11, dsize=[256, 256], seed=args.seed)
    else:
        NotImplementedError('This dataset is not supported!')

    # Wrap t, d, and c together
    y_train = np.stack([t_train, d_train, c_train], axis=1)
    if val:
        y_valid = np.stack([t_valid, d_valid, c_valid], axis=1)
    y_test = np.stack([t_test, d_test, c_test], axis=1)

    if val:
        return x_train, x_valid, x_test, y_train, y_valid, y_test
    else:
        return x_train, x_test, x_test, y_train, y_test, y_test


def run_experiment(args):
    os.chdir('../../bin/')

    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])

    x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(args)
    # Check variances of columns
    feat_var = np.var(x_train, axis=0)
    # Filter out features with low variance
    x_train = x_train[:, feat_var > 0.0001]
    x_valid = x_valid[:, feat_var > 0.0001]
    x_test = x_test[:, feat_var > 0.0001]
    print("Remaining dimensions: " + str(x_train.shape))

    aft = WeibullAFTFitter(penalizer=args.penalty_weight)
    df = construct_surv_df(x_train, y_train[:, 0], y_train[:, 1])
    aft = aft.fit(df, duration_col='time_to_event', event_col='failure', show_progress=True)

    # Training set performance
    ci = cindex(t=y_train[:, 0], d=y_train[:, 1], scores_pred=aft.predict_median(df))
    rae_nc = RAE(t_pred=aft.predict_median(df)[y_train[:, 1] == 1], t_true=y_train[y_train[:, 1] == 1, 0],
                 cens_t=1 - y_train[y_train[:, 1] == 1, 1])
    rae_c = RAE(t_pred=aft.predict_median(df)[y_train[:, 1] == 0], t_true=y_train[y_train[:, 1] == 0, 0],
                cens_t=1 - y_train[y_train[:, 1] == 0, 1])

    if args.data == 'support':
        f = open("results_SUPPORT_AFT.txt", "a+")
    elif args.data == 'flchain':
        f = open("results_FLChain_AFT.txt", "a+")
    elif args.data == 'nki':
        f = open("results_NKI_AFT.txt", "a+")
    elif args.data == 'hgg':
        f = open("results_HGG_AFT.txt", "a+")
    elif args.data == 'nsclc':
        f = open("results_nsclc_AFT.txt", "a+")

    f.write("weight_penalty= %f, name= %s, seed= %d.\n" % (args.penalty_weight, ex_name, args.seed))
    f.write("Train  |   CI: %f, RAE (nc.): %f, RAE (c.): %f.\n" % (ci, rae_nc, rae_c))

    # Test set performance
    df = construct_surv_df(x_test, y_test[:, 0], y_test[:, 1])
    ci = cindex(t=y_test[:, 0], d=y_test[:, 1], scores_pred=aft.predict_median(df))
    rae_nc = RAE(t_pred=aft.predict_median(df)[y_test[:, 1] == 1], t_true=y_test[y_test[:, 1] == 1, 0],
                 cens_t=1 - y_test[y_test[:, 1] == 1, 1])
    rae_c = RAE(t_pred=aft.predict_median(df)[y_test[:, 1] == 0], t_true=y_test[y_test[:, 1] == 0, 0],
                cens_t=1 - y_test[y_test[:, 1] == 0, 1])

    lambda_, rho_ = aft._prep_inputs_for_prediction_and_return_scores(df, ancillary_X=None)
    t_sample = utils.sample_weibull(scales=lambda_, shape=rho_)
    cal = calibration(predicted_samples=t_sample, t=y_test[:, 0], d=y_test[:, 1])

    f.write("Test   |   CI: %f, RAE (nc.): %f, RAE (c.): %f, CAL: %f.\n" % (ci, rae_nc, rae_c, cal))
    f.close()
    print(str(ci))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        default='support',
                        type=str,
                        choices=['support', 'flchain', 'hgg', 'nsclc'],
                        help='specify the data (support, flchain, hgg, nsclc)')
    parser.add_argument('--num_clusters',
                        default=5,
                        type=int,
                        help='specify the number of clusters')
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='specify the random generator seed')
    parser.add_argument('--penalty_weight',
                        default=0.0,
                        type=float,
                        help='specify the penalty weight in the Cox PH model')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
