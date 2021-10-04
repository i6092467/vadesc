"""
Numerical simulations and utility functions for constructing the synthetic dataset.
"""
import numpy as np
from numpy.random import multivariate_normal, uniform, choice

from sklearn.datasets import make_spd_matrix

from scipy.stats import weibull_min

from utils.sim_utils import random_nonlin_map

from baselines.sca.sca_utils.pre_processing import formatted_data


def simulate_profile_surv(p: int, n: int, k: int, p_cens: float, seed: int, p_c=None, balanced=False, clust_mean=True,
                          clust_cov=True, isotropic=False, clust_coeffs=True, clust_intercepts=True, density=0.2,
                          weibull_k=1, xrange=[-5, 5], brange=[-1, 1]):
    """
    Simulates data with heterogeneous survival profiles.

    :param p: number of predictor variables.
    :param n: number of data points.
    :param k: nmber of clusters.
    :param p_cens: probability of censoring.
    :param seed: random generator seed.
    :param p_c: prior probabilities of clusters.
    :param balanced: if p_c is not specified, should cluster sizes be balanced?
    :param clust_mean: should predictors have clusterwise means?
    :param clust_cov: should predictors have clusterwise covariance matrices?
    :param isotropic: should predictor covariance matrices be isotropic?
    :param clust_coeffs: should survival time predictor coefficients be cluster-specific?
    :param clust_intercepts: should survival time intercept be cluster-specific?
    :param density: proportion of predictor variables contributing to survival time.
    :param weibull_k: the shape parameter of the Weibull distribution for survival time (> 0)
    :param xrange: range for the mean of predictors.
    :param brange: range for the survival coefficients.
    :return:
    """
    # Replicability
    np.random.seed(seed)

    # Sanity checks
    assert p > 0 and n > 0 and k > 0
    assert 1 < k < n
    assert len(xrange) == 2 and xrange[0] < xrange[1]
    assert len(brange) == 2 and brange[0] < brange[1]
    assert 0 < density <= 1.0 and int((1 - density) * p) >= 1
    assert weibull_k > 0

    # Cluster prior prob-s
    if p_c is not None:
        assert len(p_c) == k and sum(p_c) == 1
    else:
        if balanced:
            p_c = np.ones((k, )) / k
        else:
            p_c = uniform(0, 1, (k, ))
            p_c = p_c / np.sum(p_c)

    # Cluster assignments
    c = choice(a=np.arange(k), size=(n, ), replace=True, p=p_c)

    # Cluster-specific means
    means = np.zeros((k, p))
    mu = uniform(xrange[0], xrange[1], (1, p))
    for l in range(k):
        if clust_mean:
            mu_l = uniform(xrange[0], xrange[1], (1, p))
            means[l, :] = mu_l
        else:
            means[l, :] = mu

    # Cluster-specific covariances
    cov_mats = []
    sigma = make_spd_matrix(p, random_state=seed)
    if isotropic:
        sigma = sigma * np.eye(p)
    for l in range(k):
        if clust_cov:
            sigma_l = make_spd_matrix(p, random_state=(seed + l))
            if isotropic:
                sigma_l = sigma_l * np.eye(p)
            cov_mats.append(sigma_l)
        else:
            cov_mats.append(sigma)

    # Predictors
    X = np.zeros((n, p))
    for l in range(k):
        n_l = np.sum(c == l)
        X_l = multivariate_normal(mean=means[l, :], cov=cov_mats[l], size=n_l)
        X[c == l, :] = X_l

    # Cluster-specific coefficients for the survival model
    coeffs = np.zeros((k, p))
    intercepts = np.zeros((k, ))
    beta = uniform(brange[0], brange[1], (1, p))
    beta0 = uniform(brange[0], brange[1], (1, 1))
    n_zeros = int((1 - density) * p)
    zero_coeffs = choice(np.arange(p), (n_zeros, ), replace=False)
    beta[:, zero_coeffs] = 0.0
    for l in range(k):
        if clust_coeffs:
            beta_l = uniform(brange[0], brange[1], (1, p))
            zero_coeffs_l = choice(np.arange(p), (n_zeros, ), replace=False)
            beta_l[:, zero_coeffs_l] = 0.0
            coeffs[l, :] = beta_l
        else:
            coeffs[l, :] = beta
        if clust_intercepts:
            beta0_l = uniform(brange[0], brange[1], (1, 1))
            intercepts[l] = beta0_l
        else:
            intercepts[l] = beta0

    # Survival times
    t = np.zeros((n, ))
    for l in range(k):
        n_l = np.sum(c == l)
        X_l = X[c == l, :]
        coeffs_l = np.expand_dims(coeffs[l, :], 1)
        intercept_l = intercepts[l]
        logexps_l = np.log(1 + np.exp(intercept_l + np.squeeze(np.matmul(X_l, coeffs_l))))

        t_l = weibull_min.rvs(weibull_k, loc=0, scale=logexps_l, size=n_l)

        t[c == l] = t_l

    # Censoring
    # NB: d == 1 if failure; 0 if censored
    d = (uniform(0, 1, (n, )) >= p_cens) * 1.0
    t_cens = uniform(0, t, (n, ))
    t[d == 0] = t_cens[d == 0]

    return X, t, d, c, means, cov_mats, coeffs, intercepts


def simulate_nonlin_profile_surv(p: int, n: int, k: int, latent_dim: int, p_cens: float, seed: int, p_c=None,
                                 balanced=False, clust_mean=True, clust_cov=True, isotropic=False, clust_coeffs=True,
                                 clust_intercepts=True, weibull_k=1, xrange=[-5, 5], brange=[-1, 1]):
    """
    Simulates data with heterogeneous survival profiles and nonlinear (!) relationships
    (covariates are generated from latent features using an MLP decoder).
    """
    # Replicability
    np.random.seed(seed)

    # Sanity checks
    assert p > 0 and latent_dim > 0 and n > 0 and k > 0
    assert 1 < k < n
    assert latent_dim < p
    assert len(xrange) == 2 and xrange[0] < xrange[1]
    assert len(brange) == 2 and brange[0] < brange[1]
    assert weibull_k > 0

    # Cluster prior prob-s
    if p_c is not None:
        assert len(p_c) == k and sum(p_c) == 1
    else:
        if balanced:
            p_c = np.ones((k, )) / k
        else:
            p_c = uniform(0, 1, (k, ))
            p_c = p_c / np.sum(p_c)

    # Cluster assignments
    c = choice(a=np.arange(k), size=(n, ), replace=True, p=p_c)

    # Cluster-specific means
    means = np.zeros((k, latent_dim))
    mu = uniform(xrange[0], xrange[1], (1, latent_dim))
    for l in range(k):
        if clust_mean:
            mu_l = uniform(xrange[0], xrange[1], (1, latent_dim))
            means[l, :] = mu_l
        else:
            means[l, :] = mu

    # Cluster-specific covariances
    cov_mats = []
    sigma = make_spd_matrix(latent_dim, random_state=seed)
    if isotropic:
        sigma = sigma * np.eye(latent_dim)
    for l in range(k):
        if clust_cov:
            sigma_l = make_spd_matrix(latent_dim, random_state=(seed + l))
            if isotropic:
                sigma_l = sigma_l * np.eye(latent_dim)
            cov_mats.append(sigma_l)
        else:
            cov_mats.append(sigma)

    # Latent features
    Z = np.zeros((n, latent_dim))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = multivariate_normal(mean=means[l, :], cov=cov_mats[l], size=n_l)
        Z[c == l, :] = Z_l

    # Predictors
    mlp_dec = random_nonlin_map(n_in=latent_dim, n_out=p, n_hidden=int((latent_dim + p) / 2))
    X = mlp_dec(Z)

    # Cluster-specific coefficients for the survival model
    coeffs = np.zeros((k, latent_dim))
    intercepts = np.zeros((k, ))
    beta = uniform(brange[0], brange[1], (1, latent_dim))
    beta0 = uniform(brange[0], brange[1], (1, 1))
    for l in range(k):
        if clust_coeffs:
            beta_l = uniform(brange[0], brange[1], (1, latent_dim))
            coeffs[l, :] = beta_l
        else:
            coeffs[l, :] = beta
        if clust_intercepts:
            beta0_l = uniform(brange[0], brange[1], (1, 1))
            intercepts[l] = beta0_l
        else:
            intercepts[l] = beta0

    # Survival times
    t = np.zeros((n, ))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = Z[c == l, :]
        coeffs_l = np.expand_dims(coeffs[l, :], 1)
        intercept_l = intercepts[l]
        logexps_l = np.log(1 + np.exp(intercept_l + np.squeeze(np.matmul(Z_l, coeffs_l))))

        t_l = weibull_min.rvs(weibull_k, loc=0, scale=logexps_l, size=n_l)

        t[c == l] = t_l

    # Censoring
    # NB: d == 1 if failure; 0 if censored
    d = (uniform(0, 1, (n, )) >= p_cens) * 1.0
    t_cens = uniform(0, t, (n, ))
    t[d == 0] = t_cens[d == 0]

    return X, t, d, c, Z, mlp_dec, means, cov_mats, coeffs, intercepts


def format_profile_surv_data_tf(p: int, n: int, k: int, p_cens: float, seed: int, p_c=None, balanced=False,
                                clust_mean=True, clust_cov=True, isotropic=False, clust_coeffs=True,
                                clust_intercepts=True, density=0.2, weibull_k=1, xrange=[-5, 5], brange=[-1, 1]):
    # Generates data with heterogeneous survival profiles, performs train-validation-test split, and returns data in the
    # same format as in the code of SCA by Chapfuwa et al.
    np.random.seed(seed)

    # Simulate the data
    X, t, d, c, means, cov_mats, coeffs, intercepts = simulate_profile_surv(p, n, k, p_cens, seed, p_c, balanced,
                                                                            clust_mean, clust_cov, isotropic,
                                                                            clust_coeffs, clust_intercepts, density,
                                                                            weibull_k, xrange, brange)

    # Renaming
    x = X
    e = d

    print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]
    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))

    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx, imputation_values=None),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx, imputation_values=None),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx, imputation_values=None)
    }

    return preprocessed


def format_nonlin_profile_surv_data_tf(p: int, n: int, k: int, latent_dim: int, p_cens: float, seed: int, p_c=None,
                                 balanced=False, clust_mean=True, clust_cov=True, isotropic=False, clust_coeffs=True,
                                 clust_intercepts=True, weibull_k=1, xrange=[-5, 5], brange=[-1, 1]):
    # Generates data with heterogeneous survival profiles and nonlinear relationships,
    # performs train-validation-test split, and returns data in the same format as in the code of SCA by Chapfuwa et al.
    np.random.seed(seed)

    # Simulate the data
    X, t, d, c, Z, mus, sigmas, betas, betas_0, mlp_dec = simulate_nonlin_profile_surv(p=p, n=n, latent_dim=latent_dim,
                                                                        k=k, p_cens=p_cens, seed=seed,
                                                                        clust_mean=clust_mean, clust_cov=clust_cov,
                                                                        clust_coeffs=clust_coeffs,
                                                                        clust_intercepts=clust_intercepts,
                                                                        balanced=balanced, weibull_k=weibull_k,
                                                                        brange=brange, isotropic=isotropic,
                                                                        xrange=xrange)

    # Renaming
    x = X
    e = d

    print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]
    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))

    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx, imputation_values=None),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx, imputation_values=None),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx, imputation_values=None)
    }

    return preprocessed
