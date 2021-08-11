# Some plotting utility functions
import os

import numpy as np

from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt
from matplotlib import rc

from sklearn.manifold import TSNE

from lifelines import CoxPHFitter
from data_utils import construct_surv_df

CB_COLOR_CYCLE = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']


def plot_overall_kaplan_meier(t, d, dir=None):
    kmf = KaplanMeierFitter()
    kmf.fit(t, d, label="Overall KM estimate")
    kmf.plot(ci_show=True)
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "km_plot.png"), dpi=300, pad_inches=0.2)
    plt.show()


def plot_group_kaplan_meier(t, d, c, dir=None):
    labels = np.unique(c)
    for l in labels:
        kmf = KaplanMeierFitter()
        kmf.fit(t[c == l], d[c == l], label="Cluster " + str(int(l + 1)))
        kmf.plot(ci_show=True, color=CB_COLOR_CYCLE[int(l)])
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "km_group_plot.png"), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_group_coxph(x, t, d, c, dir=None):
    labels = np.unique(c)
    for l in labels:
        aft = CoxPHFitter(penalizer=0.01)
        variances = np.var(x[c == l, :], axis=0)
        x_c = x[c == l, :]
        t_c = t[c == l]
        # t_c = t_c / np.max(t_c) + 0.001
        df = construct_surv_df(x_c, t_c, d[c == l])
        aft = aft.fit(df, duration_col='time_to_event', event_col='failure', show_progress=True)
        # betas = np.zeros(shape=(x.shape[1], ))
        betas = aft.params_
        thetas = np.linspace(0, 1, len(betas)) * 2 * np.pi
        thetas = np.append(thetas, 0)
        rs = np.abs(betas) / np.max(np.abs(betas))
        rs = np.append(rs, rs[0])
        plt.polar(thetas, rs, linewidth=0.75, color=CB_COLOR_CYCLE[int(l)], label="Cluster " + str(int(l + 1)))
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.legend()
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "coeff_plot.png"), dpi=300, pad_inches=0.2)
    plt.show()


def plotting_setup(font_size=12):
    # plot settings
    plt.style.use("seaborn-colorblind")
    plt.rcParams['font.size'] = font_size
    rc('text', usetex=False)
    plt.rcParams["font.family"] = "Times New Roman"
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def plot_dataset(X, t, d, c, font_size=12, seed=42, dir=None):
    plotting_setup(font_size=font_size)

    plot_group_kaplan_meier(t=t, d=d, c=c, dir=dir)

    X_embedded = TSNE(n_components=2, random_state=seed).fit_transform(X)

    for l in np.unique(c):
        plt.scatter(X_embedded[c == l, 0], X_embedded[c == l, 1], s=1.5, c=CB_COLOR_CYCLE[int(l)],
                    label=("Cluster " + str(int(l + 1))))
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(markerscale=3.0)
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "tsne_plot.png"), dpi=300)
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


def plot_elbow_parcoord(ks, vals, xlab, ylab, dir=None):
    plotting_setup(16)
    for i in range(vals.shape[0]):
        plt.plot(ks, vals[i, :], color=CB_COLOR_CYCLE[0])
    plt.plot(ks, np.mean(vals, axis=0), color="black")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    if dir is not None:
        plt.savefig(fname=os.path.join(dir, "elbow_plot.png"), dpi=300, bbox_inches="tight")
    plt.show()
