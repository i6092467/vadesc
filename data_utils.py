# Some utility functions for data handling
import numpy as np

import pandas as pd


def construct_surv_df(X, t, d):
    p = X.shape[1]
    df = pd.DataFrame(X, columns=["X_" + str(i) for i in range(p)])
    df["time_to_event"] = t
    df["failure"] = d
    return df
