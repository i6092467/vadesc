"""
Utility functions for plotting.
"""
import os

import numpy as np

from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt
from matplotlib import rc

from openTSNE import TSNE as fastTSNE

import sys

sys.path.insert(0, '../')

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
GRAY_COLOR_CYCLE = ['black', 'dimgray', 'darkgray', 'gainsboro', 'whitesmoke']
LINE_TYPES = ['solid', 'dashed', 'dashdot', 'dotted', 'dashed']
MARKER_STYLES = ['', '', '', '', '']
DASH_STYLES = [[], [4, 4], [4, 1], [1, 1, 1], [2, 1, 2]]


def plotting_setup(font_size=12):
    # plot settings
    plt.style.use("seaborn-colorblind")
    plt.rcParams['font.size'] = font_size
    rc('text', usetex=False)


def plot_overall_kaplan_meier(t, d, dir=None):
    kmf = KaplanMeierFitter()
    kmf.fit(t, d, label="Overall KM estimate")
    kmf.plot(ci_show=True)
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "km_plot.png"), dpi=300, pad_inches=0.2)
    plt.show()


def plot_group_kaplan_meier(t, d, c, dir=None, experiment_name=''):
    fig = plt.figure()
    labels = np.unique(c)
    for l in labels:
        kmf = KaplanMeierFitter()
        kmf.fit(t[c == l], d[c == l], label="Cluster " + str(int(l + 1)))
        kmf.plot(ci_show=True, color=CB_COLOR_CYCLE[int(l)])
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "km_group_plot_" + experiment_name +".png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_bigroup_kaplan_meier(t, d, c, c_, dir=None, postfix=None, legend=False, legend_outside=False):
    fig = plt.figure()

    # Plot true clusters
    labels = np.unique(c)
    for l in labels:
        kmf = KaplanMeierFitter()
        if legend:
            kmf.fit(t[c == l], d[c == l], label="Cluster " + str(int(l + 1)))
        else:
            kmf.fit(t[c == l], d[c == l])
        kmf.plot(ci_show=True, alpha=0.75, color=CB_COLOR_CYCLE[int(l)], linewidth=5)

    # Plot assigned clusters
    labels = np.unique(c_)
    for l in labels:
        kmf = KaplanMeierFitter()
        if legend:
            kmf.fit(t[c_ == l], d[c_ == l], label="Ass. cluster " + str(int(l + 1)))
        else:
            kmf.fit(t[c_ == l], d[c_ == l])
        kmf.plot(ci_show=True, color='black', alpha=0.25, linestyle=LINE_TYPES[int(l)], dashes=DASH_STYLES[int(l)],
                 linewidth=5)

    plt.xlabel("Time")
    plt.ylabel("Survival Probability")

    if legend:
        if legend_outside:
            leg = plt.legend(loc='upper right', frameon=False, bbox_to_anchor=(-0.15, 1))
        else:
            leg = plt.legend(loc='lower right', frameon=False)
    else:
        leg = plt.legend('', frameon=False)

    if dir is not None:
        fname = 'km_bigroup_plot'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_dataset(X, t, d, c, font_size=12, seed=42, dir=None, postfix=None):
    plotting_setup(font_size=font_size)

    plot_group_kaplan_meier(t=t, d=d, c=c, dir=dir)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000, ))
        c_ = c[inds]
        X_ = X[inds]
    else:
        c_ = c
        X_ = X

    X_embedded = fastTSNE(n_components=2, n_jobs=8, random_state=seed).fit(X_)

    fig = plt.figure()
    for l in np.unique(c_):
        plt.scatter(X_embedded[c_ == l, 0], X_embedded[c_ == l, 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                    label=("Cluster " + str(int(l + 1))))
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(markerscale=3.0)
    if dir is not None:
        fname = 'tsne'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300)
    else:
        plt.show()


def plot_tsne_by_cluster(X, c, font_size=12, seed=42, dir=None, postfix=None):
    np.random.seed(seed)

    plotting_setup(font_size=font_size)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000,))
        c_ = c[inds]
        X_ = X[inds]
    else:
        c_ = c
        X_ = X

    X_embedded = fastTSNE(n_components=2, n_jobs=8, random_state=seed).fit(X_)

    fig = plt.figure()
    for l in np.unique(c_):
        plt.scatter(X_embedded[c_ == l, 0], X_embedded[c_ == l, 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                    label=("Cluster " + str(int(l + 1))))
    plt.xlabel(r'$t$-SNE Dimension 1')
    plt.ylabel(r'$t$-SNE Dimension 2')
    plt.legend(markerscale=3.0)
    if dir is not None:
        fname = 'tsne_vs_c'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300)
    else:
        plt.show()


def plot_tsne_by_survival(X, t, d, font_size=16, seed=42, dir=None, postfix=None, plot_censored=True):
    np.random.seed(seed)

    plotting_setup(font_size=font_size)

    if X.shape[0] > 10000:
        inds = np.random.choice(a=np.arange(0, X.shape[0]), size=(10000,))
        t_ = t[inds]
        d_ = d[inds]
        X_ = X[inds]
    else:
        t_ = t
        d_ = d
        X_ = X

    X_embedded = fastTSNE(n_components=2, n_jobs=8, random_state=seed).fit(X_)

    fig = plt.figure()
    plt.scatter(X_embedded[d_ == 1, 0], X_embedded[d_ == 1, 1], s=1.5, c=np.log(t_[d_ == 1]), cmap='cividis', alpha=0.5)
    if plot_censored:
        plt.scatter(X_embedded[d_ == 0, 0], X_embedded[d_ == 0, 1], s=1.5, c=np.log(t_[d_ == 0]), cmap='cividis',
                    alpha=0.5, marker='s')
    clb = plt.colorbar()
    clb.ax.set_title(r'$\log(T)$')
    plt.xlabel(r'$t$-SNE Dimension 1')
    plt.ylabel(r'$t$-SNE Dimension 2')
    plt.axis('off')
    if dir is not None:
        fname = 'tsne_vs_t'
        if postfix is not None:
            fname += '_' + postfix
        fname += '.png'
        plt.savefig(fname=os.path.join(dir, fname), dpi=300)
    else:
        plt.show()


def plot_elbow(ks, avg, sd, xlab, ylab, dir=None):
    plotting_setup(16)
    plt.errorbar(ks, avg, yerr=sd, color=CB_COLOR_CYCLE[0], ecolor=CB_COLOR_CYCLE[0], barsabove=True,  marker='D')
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "elbow_plot.png"), dpi=300, bbox_inches="tight")
    plt.show()
