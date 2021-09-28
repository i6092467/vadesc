# Runs k-means clustering
import argparse
import numpy as np
import time
import uuid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, '../../')

from datasets.survivalMNIST.survivalMNIST_data import generate_surv_MNIST
from datasets.simulations import simulate_nonlin_profile_surv
from datasets.hemodialysis.hemo_data import generate_hemo

from utils import utils


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
    elif args.data == 'hemo':
        c = args.num_clusters
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = generate_hemo(seed=args.seed, label=c)
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
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])

    x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(args)

    km = KMeans(n_clusters=args.num_clusters)
    km = km.fit(X=x_train)

    # Training set performance
    yy = km.predict(X=x_train)
    acc = utils.cluster_acc(y_train[:, 2], yy)
    nmi = normalized_mutual_info_score(y_train[:, 2], yy)
    ari = adjusted_rand_score(y_train[:, 2], yy)

    if args.data == 'MNIST':
        f = open("results_MNIST_KM.txt", "a+")
    elif args.data == 'sim':
        f = open("results_sim_KM.txt", "a+")
    elif args.data == 'liverani':
        f = open("results_liverani_KM.txt", "a+")
    elif args.data == 'hemo':
        f = open("results_hemo_KM.txt", "a+")

    f.write("Accuracy train: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))

    # Test set performance
    yy = km.predict(X=x_test)

    acc = utils.cluster_acc(y_test[:, 2], yy)
    nmi = normalized_mutual_info_score(y_test[:, 2], yy)
    ari = adjusted_rand_score(y_test[:, 2], yy)

    f.write("Accuracy test: %f, NMI: %f, ARI: %f.\n" % (acc, nmi, ari))
    f.close()
    print(str(acc))
    print(str(nmi))
    print(str(ari))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        default='MNIST',
                        type=str,
                        choices=['MNIST', 'sim', 'hemo'],
                        help='specify the data (MNIST, sim, hemo)')
    parser.add_argument('--num_clusters',
                        default=5,
                        type=int,
                        help='specify the number of clusters')
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='specify the random generator seed')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
