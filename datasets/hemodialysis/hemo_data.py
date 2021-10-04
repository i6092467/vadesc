"""
Dataset of children undergoing hemodialysis.
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_hemo(seed=42, label=3):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'data.csv'))
    df = pd.read_csv(path)
    df["cause.of.death"].loc[df['death'] == 0] = 'alive'
    df["fcensor.reason"] = df["fcensor.reason"].fillna(value='unkown')
    df["PatientRace4"] = df["PatientRace4"].fillna(value='unkown')
    df = df.interpolate(method="nearest")
    df["mean_rdw"][0] = df["mean_rdw"].mean()
    t = df['TIME'].to_numpy()
    t.astype(np.float64)
    del df['TIME']
    d = df['death'].to_numpy()
    d = np.array(d, dtype=bool)
    del df['death']
    del df['cause.of.death']
    del df['fcensor.reason']
    del df['fspktv3']
    del df['raceB']

    #clusters
    c_2 = df['fage2'].to_numpy()
    del df['fage2']
    c_3 = df['fage3'].to_numpy()
    del df['fage3']
    c_2[c_2 == '0-12 years'] = 0
    c_2[c_2 == '>12 years'] = 1
    c_3[c_3 == '<6 years'] = 0
    c_3[c_3 == '6-12 years'] = 1
    c_3[c_3 == '>12 years'] = 2
    if label == 2:
        c = c_2
    else:
        c = c_3
    c = np.array(c, dtype=np.int64)
    df = pd.get_dummies(df)

    # Covariates to exclude (repetition)
    no_list = ['PatientRace4_unkown', 'raceB_African', 'fspktv4_(1.56,1.73]', #'fspktv4_[0.784,1.39]',
               'USRDS_class_Etiology uncertain ', 'other', 'tidwg_day', 'tUFR_mLkgh',
               'raceB_other', 'cDeath', 'cTIME', 'PatientIdentifier', 'PatientGender_Male',
               'etiology2_other', 'PatientRace4_Other', 'etiology2_sec_glomerulonephritis_vasculitis']

    for col in df.columns:
        if col in no_list:
            del df[col]

    data = df.to_numpy()
    X = StandardScaler().fit_transform(data)
    X = X.astype(np.float64)
    t = t / np.max(t) + 0.001

    x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                          random_state=seed)
    x_valid = x_test
    t_valid = t_test
    d_valid = d_test
    c_valid = c_test

    return x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test
