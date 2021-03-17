# A rough Python implementation of the semi-supervised survival data clustering described in
# https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0020108

import numpy as np

from lifelines import CoxPHFitter

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from data_utils import construct_surv_df


class SSC_Bair():
    def __init__(self, n_clusters: int, input_dim: int, clustering_dim: int, random_state: int, penalty_weight=0.0):
        self.cph = []
        self.X_train = None
        self.t_train = None
        self.d_train = None
        self.hazard_ratios = []
        self.clustering_features = None

        assert n_clusters >= 2
        assert 0 < clustering_dim <= input_dim

        self.n_clusters = n_clusters

        self.input_dim = input_dim
        self.clustering_dim = clustering_dim

        self.penalty_weight = penalty_weight

        self.km = KMeans(n_clusters=self.n_clusters, random_state=random_state)

        self.random_state = random_state

    def fit(self, X: np.ndarray, t: np.ndarray, d: np.ndarray):
        self.X_train = X
        self.t_train = t
        self.d_train = d

        for j in range(self.X_train.shape[1]):
            print("Fitting Cox PH model " + str(j) + "/" + str(self.X_train.shape[1]), end="\r")
            # Fit a univariate Cox PH model
            cph_j = CoxPHFitter(penalizer=self.penalty_weight)
            df = construct_surv_df(np.expand_dims(X[:, j], 1), t, d)
            cph_j.fit(df, duration_col='time_to_event', event_col='failure', show_progress=False)
            self.cph.append(cph_j)
            # Retrieve the hazard ratio
            self.hazard_ratios.append(cph_j.hazard_ratios_.array[0])
        print()
        self.hazard_ratios = np.array(self.hazard_ratios)

        # Choose top significant features
        self.clustering_features = np.argsort(-self.hazard_ratios)[:self.clustering_dim]

        # Perform k-means
        self.km = self.km.fit(X[:, self.clustering_features])

        return self

    def re_fit(self, new_clustering_dim: int):
        # Re-fits with a new dimensionality
        assert self.X_train is not None and self.t_train is not None and self.d_train is not None

        self.clustering_dim = new_clustering_dim

        self.km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        # Choose top significant features
        self.clustering_features = np.argsort(-self.hazard_ratios)[:self.clustering_dim]

        # Perform k-means
        self.km = self.km.fit(self.X_train[:, self.clustering_features])

        return self

    def predict(self, X: np.ndarray):
        assert self.clustering_features is not None

        c_pred = self.km.predict(X=X[:, self.clustering_features])

        return c_pred


def find_best_dim(model: SSC_Bair, c: np.ndarray, step=30):
    dims = np.arange(1, model.input_dim, step)
    d_best = -1
    nmi_best = 0.0
    for d in dims:
        model_d = model.re_fit(new_clustering_dim=d)
        c_pred_d = model_d.predict(X=model.X_train)
        nmi_d = normalized_mutual_info_score(labels_true=c, labels_pred=c_pred_d)
        if nmi_d > nmi_best:
            nmi_best = nmi_d
            d_best = d
    model_best = model.re_fit(new_clustering_dim=d_best)
    return model_best, d_best, nmi_best
