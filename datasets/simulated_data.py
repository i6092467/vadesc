"""
Returns the synthetic data.
"""
from datasets.simulations import format_profile_surv_data_tf


def generate_data():
    preproc = format_profile_surv_data_tf(p=100, n=1000, k=5, p_cens=0.2, seed=42, clust_mean=False, clust_cov=False,
                                          clust_intercepts=False, density=0.2, weibull_k=1, xrange=[-5, 5],
                                          brange=[-2.5, 2.5])
    return preproc
