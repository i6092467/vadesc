"""
Runs Cox PH regression.
"""
import argparse
import os

import numpy as np

import time

import uuid

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter

import sys
sys.path.insert(0, '../../')
from datasets.survivalMNIST.survivalMNIST_data import generate_surv_MNIST
from datasets.simulations import simulate_nonlin_profile_surv
from datasets.support.support_data import generate_support
from datasets.hemodialysis.hemo_data import generate_hemo
from datasets.nsclc_lung.nsclc_lung_data import generate_radiomic_features

from utils.data_utils import construct_surv_df

from utils.eval_utils import cindex

from utils.eval_utils import rae as RAE, calibration


def get_data(args, val=False):
    if args.data == 'MNIST':
        valid_perc = .15
        if not val:
            valid_perc = .0
        if val:
            x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
                generate_surv_MNIST(n_groups=args.num_clusters, seed=args.seed, p_cens=.3, valid_perc=valid_perc)
        else:
            x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = generate_surv_MNIST(n_groups=args.num_clusters,
                                                                                                     seed=args.seed,
                                                                                                     p_cens=.3,
                                                                                                     valid_perc=valid_perc)
        # Normalisation
        x_test = x_test / 255.
        if val:
            x_valid = x_valid / 255.
        x_train = x_train / 255.
    elif args.data == "sim":
        X, t, d, c, Z, mus, sigmas, betas, betas_0, mlp_dec = simulate_nonlin_profile_surv(p=1000, n=60000,
                                                                                           latent_dim=16,
                                                                                           k=args.num_clusters,
                                                                                           p_cens=.3, seed=args.seed,
                                                                                           clust_mean=True,
                                                                                           clust_cov=True,
                                                                                           clust_coeffs=True,
                                                                                           clust_intercepts=True,
                                                                                           balanced=True,
                                                                                           weibull_k=1,
                                                                                           brange=[-10.0, 10.0],
                                                                                           isotropic=True,
                                                                                           xrange=[-.5, .5])
        # Normalisation
        t = t / np.max(t) + 0.001
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                              random_state=args.seed)
    elif args.data == 'support':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_support(seed=args.seed)
    elif args.data == 'hemo':
        c = args.num_clusters
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = generate_hemo(seed=args.seed, label=c)
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

    cph = CoxPHFitter(penalizer=args.penalty_weight)
    df = construct_surv_df(x_train, y_train[:, 0], y_train[:, 1])
    cph = cph.fit(df, duration_col='time_to_event', event_col='failure', show_progress=True)

    # Training set performance
    risk_scores = np.exp(-np.squeeze(np.matmul(x_train, np.expand_dims(cph.params_, 1))))
    ci = cindex(t=y_train[:, 0], d=y_train[:, 1], scores_pred=risk_scores)

    if args.data == 'MNIST':
        f = open("results_MNIST_Cox.txt", "a+")
    elif args.data == 'sim':
        f = open("results_sim_Cox.txt", "a+")
    elif args.data == 'liverani':
        f = open("results_liverani_Cox.txt", "a+")
    elif args.data == 'nki':
        f = open("results_NKI_Cox.txt", "a+")
    elif args.data == 'support':
        f = open("results_SUPPORT_Cox.txt", "a+")
    elif args.data == 'hemo':
        f = open("results_hemo_Cox.txt", "a+")
    elif args.data == 'nsclc':
        f = open("results_nsclc_Cox.txt", "a+")

    f.write("weight_penalty= %f, name= %s, seed= %d.\n" % (args.penalty_weight, ex_name, args.seed))
    f.write("Train  |   CI: %f.\n" % (ci))

    # Test set performance
    risk_scores = np.exp(-np.squeeze(np.matmul(x_test, np.expand_dims(cph.params_, 1))))
    ci = cindex(t=y_test[:, 0], d=y_test[:, 1], scores_pred=risk_scores)

    rae_nc = RAE(t_pred=cph.predict_median(x_test)[y_test[:, 1] == 1], t_true=y_test[y_test[:, 1] == 1, 0],
                 cens_t=1 - y_test[y_test[:, 1] == 1, 1])
    rae_c = RAE(t_pred=cph.predict_median(x_test)[y_test[:, 1] == 0], t_true=y_test[y_test[:, 1] == 0, 0],
                cens_t=1 - y_test[y_test[:, 1] == 0, 1])

    times_sorted = np.sort(np.unique(y_train[y_train[:, 1] == 1, 0]))
    cdfs = np.transpose(1 - cph.predict_survival_function(X=x_test, times=times_sorted))
    cdfs = np.concatenate((np.zeros((cdfs.shape[0], 1)), cdfs), axis=1)
    pdfs = np.diff(cdfs)
    t_sample = np.zeros((cdfs.shape[0], 200))
    for i in range(cdfs.shape[0]):
        pdf = pdfs[i]
        probs = pdf / np.sum(pdf)
        t_sample[i, :] = np.random.choice(a=times_sorted, p=probs, size=(200,))
    cal = calibration(predicted_samples=t_sample, t=y_test[:, 0], d=y_test[:, 1])

    f.write("Test   |   CI: %f, RAE (nc.): %f, RAE (c.): %f, CAL: %f.\n" % (ci, rae_nc, rae_c, cal))
    f.close()
    print(str(ci))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        default='MNIST',
                        type=str,
                        choices=['MNIST', 'sim', 'support', 'hemo', 'nsclc'],
                        help='specify the data (MNIST, sim, support, hemo, nsclc)')
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
