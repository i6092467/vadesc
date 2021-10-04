"""
Data loaders for NSCLC datasets.
"""
import os
import re

import numpy as np
import progressbar
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from datasets.nsclc_lung.CT_preproc_utils import (preprocess_lung1_images, preprocess_radiogenomics_images,
                                                  preprocess_radiogenomics_images_amc, preprocess_lung3_images,
                                                  preprocess_basel_images, downscale_images, IGNORED_PATIENTS,
                                                  IGNORED_RADIOGENOMICS_PATIENTS, IGNORED_LUNG3_PATIENTS,
                                                  IGNORED_BASEL_PATIENTS)

from utils.radiomics_utils import extract_radiomics_features

# TODO: insert directories with CT scans and clinical data for NSCLC datasets
LUNG1_CT_DIR = '...'
RADIOGENOMICS_DIR = '...'
LUNG3_DIR = '...'
BASEL_DIR = '...'


def generate_lung1_images(n_slices: int, dsize, seed=42, verbose=1, normalise_t=True):
    """
    Loads Lung1 CT and survival data.
    """
    np.random.seed(seed)

    # Load CT data
    X = preprocess_lung1_images(lung1_dir=LUNG1_CT_DIR, n_slices=n_slices, dsize=[256, 256], n_bins=40)

    # Downscale
    if dsize[0] < 256:
        X_d = np.zeros([X.shape[0], X.shape[1], dsize[0], dsize[1]])
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=X.shape[0])
            bar.start()
            print("Downsizing data...")
        for i in range(len(X)):
            X_d[i] = downscale_images(X[i], dsize)
            if verbose > 0:
                bar.update(i)
        X = np.expand_dims(X_d, axis=-1)
        print(X.shape)

    clinical_data = pd.read_csv('../datasets/nsclc_lung/clinical_data.csv')

    t = clinical_data['Survival.time'].values.astype('float32')
    d = clinical_data['deadstatus.event'].values.astype('float32')
    stages = clinical_data['clinical.T.Stage'].values
    stages[np.isnan(stages)] = 3
    stages[stages == 5] = 4
    c = stages - 1
    c = c.astype('int32')

    # Normalisation
    if normalise_t:
        t = t / np.max(t) + 0.001
    t = np.delete(t, 127)
    d = np.delete(d, 127)
    c = np.delete(c, 127)
    t = np.delete(t, IGNORED_PATIENTS)
    d = np.delete(d, IGNORED_PATIENTS)
    c = np.delete(c, IGNORED_PATIENTS)

    # Train-test split
    X_train, X_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=0.25,
                        random_state=seed, stratify=np.digitize(t, np.quantile(t, np.array([0.3, 0.5, 0.75, 0.9]))))

    X_train = np.reshape(X_train, newshape=(X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
    X_test = X_test[:, 0]
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    return X_train, X_test, X_test, t_train, t_test, t_test, d_train, d_test, d_test, c_train, c_test, c_test


def generate_radiogenomics_images(n_slices: int, dsize, seed=42, verbose=1, normalise_t=True):
    """
    Loads a subset of NSCLC Radiogenomics CT and survival data.
    """
    np.random.seed(seed)

    # Load CT data
    X = preprocess_radiogenomics_images(radiogenomics_dir=RADIOGENOMICS_DIR, n_slices=n_slices, dsize=[256, 256],
                                        n_bins=40)

    # Downscale
    if dsize[0] < 256:
        X_d = np.zeros([X.shape[0], X.shape[1], dsize[0], dsize[1]])
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=X.shape[0])
            bar.start()
            print("Downsizing data...")
        for i in range(len(X)):
            X_d[i] = downscale_images(X[i], dsize)
            if verbose > 0:
                bar.update(i)
        X = np.expand_dims(X_d, axis=-1)
        print(X.shape)

    clinical_data = pd.read_csv(os.path.join(RADIOGENOMICS_DIR, 'clinical_data.csv'))

    subj_ids = np.array([re.search('R01-0', str(clinical_data['Case ID'].values[i])) and
                         not(re.search('R01-097', str(clinical_data['Case ID'].values[i])) or
                             re.search('R01-098', str(clinical_data['Case ID'].values[i])) or
                             re.search('R01-099', str(clinical_data['Case ID'].values[i])))
                         for i in range(len(clinical_data['Case ID'].values))])
    subj_ids[subj_ids == None] = False
    subj_ids = subj_ids.astype(bool)
    t = (pd.to_datetime(clinical_data['Date of Last Known Alive']) -
         pd.to_datetime(clinical_data['CT Date'])).dt.days.values.astype('float32')
    t = t[subj_ids]

    d = clinical_data['Survival Status'].values
    d[d == 'Alive'] = 0
    d[d == 'Dead'] = 1
    d = d[subj_ids].astype('float32')

    stages = clinical_data['Pathological T stage'].values
    c = np.zeros_like(stages)
    c[np.logical_or(stages == 'T1a', stages == 'T1b')] = 0
    c[np.logical_or(stages == 'T2a', stages == 'T2b')] = 1
    c[stages == 'T3'] = 2
    c[stages == 'T4'] = 3
    c[stages == 'Tis'] = 0
    c = c.astype('int32')
    c = c[subj_ids]

    # Normalisation
    if normalise_t:
        t = t / np.max(t) + 0.001

    t = np.delete(t, IGNORED_RADIOGENOMICS_PATIENTS)
    d = np.delete(d, IGNORED_RADIOGENOMICS_PATIENTS)
    c = np.delete(c, IGNORED_RADIOGENOMICS_PATIENTS)

    # Train-test split
    X_train, X_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=0.25,
                        random_state=seed, stratify=np.digitize(t, np.quantile(t, np.array([0.3, 0.5, 0.75, 0.9]))))

    X_train = np.reshape(X_train, newshape=(X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
    X_test = X_test[:, 0]
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    return X_train, X_test, X_test, t_train, t_test, t_test, d_train, d_test, d_test, c_train, c_test, c_test


def generate_radiogenomics_images_amc(n_slices: int, dsize, seed=42, verbose=1, normalise_t=True):
    """
    Loads a subset of NSCLC Radiogenomics CT and survival data.
    """
    np.random.seed(seed)

    # Load CT data
    X = preprocess_radiogenomics_images_amc(radiogenomics_dir=RADIOGENOMICS_DIR, n_slices=n_slices, dsize=[256, 256],
                                            n_bins=40)

    # Downscale
    if dsize[0] < 256:
        X_d = np.zeros([X.shape[0], X.shape[1], dsize[0], dsize[1]])
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=X.shape[0])
            bar.start()
            print("Downsizing data...")
        for i in range(len(X)):
            X_d[i] = downscale_images(X[i], dsize)
            if verbose > 0:
                bar.update(i)
        X = np.expand_dims(X_d, axis=-1)
        print(X.shape)

    master_file = pd.read_csv(os.path.join(RADIOGENOMICS_DIR, 'master_file_amc.csv'))

    t = (pd.to_datetime(master_file['Date of last known alive']) -
         pd.to_datetime(master_file['CT date'])).dt.days.values.astype('float32')

    d = master_file['Survival status'].values
    d[d == 'Alive'] = 0
    d[d == 'Dead'] = 1
    d = d.astype('float32')

    # NB: no stage information in AMC subjects
    c = np.zeros_like(d)
    c = c.astype('int32')

    # Normalisation
    if normalise_t:
        t = t / np.max(t) + 0.001

    # Train-test split
    X_train, X_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=0.25,
                            random_state=seed, stratify=np.digitize(t, np.quantile(t, np.array([0.3, 0.5, 0.75, 0.9]))))

    X_train = np.reshape(X_train, newshape=(X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
    X_test = X_test[:, 0]
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    return X_train, X_test, X_test, t_train, t_test, t_test, d_train, d_test, d_test, c_train, c_test, c_test


def generate_lung3_images(n_slices: int, dsize, seed=42, verbose=1, normalise_t=True):
    """
    Loads Lung3 CT and survival data.
    """
    np.random.seed(seed)

    # Load CT data
    X = preprocess_lung3_images(lung3_dir=LUNG3_DIR, n_slices=n_slices, dsize=[256, 256], n_bins=40)

    # Downscale
    if dsize[0] < 256:
        X_d = np.zeros([X.shape[0], X.shape[1], dsize[0], dsize[1]])
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=X.shape[0])
            bar.start()
            print("Downsizing data...")
        for i in range(len(X)):
            X_d[i] = downscale_images(X[i], dsize)
            if verbose > 0:
                bar.update(i)
        X = np.expand_dims(X_d, axis=-1)
        print(X.shape)

    master_table = pd.read_csv(os.path.join(LUNG3_DIR, 'Lung3_master.csv'))

    t = np.zeros((len(master_table['Case ID'].values), ))
    d = np.zeros((len(master_table['Case ID'].values),))
    c = master_table['Tumor stage'].values - 1

    t = np.delete(t, IGNORED_LUNG3_PATIENTS)
    d = np.delete(d, IGNORED_LUNG3_PATIENTS)
    c = np.delete(c, IGNORED_LUNG3_PATIENTS)

    # Normalisation
    if normalise_t:
        t = t / np.max(t) + 0.001

    # Train-test split
    X_train, X_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=0.25,
                                                                                          random_state=seed)

    X_train = np.reshape(X_train, newshape=(X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
    X_test = X_test[:, 0]
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    return X_train, X_test, X_test, t_train, t_test, t_test, d_train, d_test, d_test, c_train, c_test, c_test


def generate_basel_images(n_slices: int, dsize, seed=42, verbose=1, normalise_t=True):
    """
    Loads Basel University Hospital CT and survival data.
    """
    np.random.seed(seed)

    # Load CT data
    X = preprocess_basel_images(basel_dir=BASEL_DIR, n_slices=n_slices, dsize=[256, 256], n_bins=40)

    # Downscale
    if dsize[0] < 256:
        X_d = np.zeros([X.shape[0], X.shape[1], dsize[0], dsize[1]])
        if verbose > 0:
            bar = progressbar.ProgressBar(maxval=X.shape[0])
            bar.start()
            print("Downsizing data...")
        for i in range(len(X)):
            X_d[i] = downscale_images(X[i], dsize)
            if verbose > 0:
                bar.update(i)
        X = np.expand_dims(X_d, axis=-1)
        print(X.shape)

    clinical_data = pd.read_csv('../datasets/nsclc_lung/clinical_data_basel.csv')

    t = clinical_data['Survival time'].values.astype('float32')
    d = clinical_data['Event'].values.astype('float32')
    c = np.zeros_like(d)

    # Normalisation
    if normalise_t:
        t = t / np.max(t) + 0.001

    t = np.delete(t, IGNORED_BASEL_PATIENTS)
    d = np.delete(d, IGNORED_BASEL_PATIENTS)
    c = np.delete(c, IGNORED_BASEL_PATIENTS)

    # Train-test split
    X_train, X_test, t_train, t_test, d_train, d_test, c_train, c_test = \
        train_test_split(X, t, d, c, test_size=0.25, random_state=seed,
                         stratify=np.digitize(t, np.quantile(t, np.array([0.3, 0.5, 0.75, 0.9]))))

    X_train = np.reshape(X_train, newshape=(X_train.shape[0] * X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))
    X_test = X_test[:, 0]
    X_test = np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

    return X_train, X_test, X_test, t_train, t_test, t_test, d_train, d_test, d_test, c_train, c_test, c_test


def generate_radiomic_features(n_slices: int, seed: int, dsize):
    """
    Loads radiomic features from all NSCLC datasets with segmentations available.
    """
    _ = preprocess_lung1_images(lung1_dir=LUNG1_CT_DIR, n_slices=n_slices, dsize=dsize, n_bins=40)
    _ = preprocess_radiogenomics_images(radiogenomics_dir=RADIOGENOMICS_DIR, n_slices=n_slices, dsize=dsize, n_bins=40)
    _ = preprocess_basel_images(basel_dir=BASEL_DIR, n_slices=n_slices, dsize=dsize, n_bins=40)

    seg_file = '../datasets/nsclc_lung/lung1_segmentations_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + \
               str(dsize[1]) + '.npy'
    masks = np.load(file=seg_file, allow_pickle=True)
    masks = np.delete(masks, np.concatenate([IGNORED_PATIENTS, [127]]), axis=0)
    radiomic_features_lung1 = extract_radiomics_features(
        data_file='../datasets/nsclc_lung/lung1_best_slices_preprocessed_' + str(n_slices) + '_' +
                  str(dsize[0]) + 'x' + str(dsize[1]) + '.npy', masks=masks)

    seg_file = '../datasets/nsclc_lung/radiogenomics_segmentations_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + \
               str(dsize[1]) + '.npy'
    masks = np.load(file=seg_file, allow_pickle=True)
    masks = np.delete(masks, IGNORED_RADIOGENOMICS_PATIENTS, axis = 0)
    radiomic_features_radiogenomics = extract_radiomics_features(
        data_file='../datasets/nsclc_lung/radiogenomics_best_slices_preprocessed_' + str(n_slices) + '_' +
                  str(dsize[0]) + 'x' + str(dsize[1]) + '.npy', masks=masks)

    seg_file = '../datasets/nsclc_lung/basel_segmentations_' + str(n_slices) + '_' + str(dsize[0]) + 'x' + \
               str(dsize[1]) + '.npy'
    masks = np.load(file=seg_file, allow_pickle=True)
    masks = np.delete(masks, IGNORED_BASEL_PATIENTS, axis = 0)
    basel_features_radiogenomics = extract_radiomics_features(
        data_file='../datasets/nsclc_lung/basel_best_slices_preprocessed_' + str(n_slices) + '_' +
                  str(dsize[0]) + 'x' + str(dsize[1]) + '.npy', masks = masks)

    clinical_data_lung1 = pd.read_csv('../datasets/nsclc_lung/clinical_data.csv')
    t_lung1 = clinical_data_lung1['Survival.time'].values.astype('float32')
    d_lung1 = clinical_data_lung1['deadstatus.event'].values.astype('float32')
    stages_lung1 = clinical_data_lung1['clinical.T.Stage'].values
    stages_lung1[np.isnan(stages_lung1)] = 3
    stages_lung1[stages_lung1 == 5] = 4
    c_lung1 = stages_lung1 - 1
    c_lung1 = c_lung1.astype('int32')
    t_lung1 = np.delete(t_lung1, np.concatenate([IGNORED_PATIENTS, [127]]))
    d_lung1 = np.delete(d_lung1, np.concatenate([IGNORED_PATIENTS, [127]]))
    c_lung1 = np.delete(c_lung1, np.concatenate([IGNORED_PATIENTS, [127]]))

    clinical_data_radiogenomics = pd.read_csv(os.path.join(RADIOGENOMICS_DIR, 'clinical_data.csv'))
    subj_ids = np.array([re.search('R01-0', str(clinical_data_radiogenomics['Case ID'].values[i])) and not (
                re.search('R01-097', str(clinical_data_radiogenomics['Case ID'].values[i])) or re.search('R01-098', str(
            clinical_data_radiogenomics['Case ID'].values[i])) or
                re.search('R01-099', str(clinical_data_radiogenomics['Case ID'].values[i]))) for i
                         in range(len(clinical_data_radiogenomics['Case ID'].values))])
    subj_ids[subj_ids == None] = False
    subj_ids = subj_ids.astype(bool)
    t_radiogenomics = (pd.to_datetime(clinical_data_radiogenomics['Date of Last Known Alive']) - pd.to_datetime(
        clinical_data_radiogenomics['CT Date'])).dt.days.values.astype('float32')
    t_radiogenomics = t_radiogenomics[subj_ids]
    d_radiogenomics = clinical_data_radiogenomics['Survival Status'].values
    d_radiogenomics[d_radiogenomics == 'Alive'] = 0
    d_radiogenomics[d_radiogenomics == 'Dead'] = 1
    d_radiogenomics = d_radiogenomics[subj_ids].astype('float32')
    # d = d * 0 # Just use for AE
    stages_radiogenomics = clinical_data_radiogenomics['Pathological T stage'].values
    c_radiogenomics = np.zeros_like(stages_radiogenomics)
    c_radiogenomics[np.logical_or(stages_radiogenomics == 'T1a', stages_radiogenomics == 'T1b')] = 0
    c_radiogenomics[np.logical_or(stages_radiogenomics == 'T2a', stages_radiogenomics == 'T2b')] = 1
    c_radiogenomics[stages_radiogenomics == 'T3'] = 2
    c_radiogenomics[stages_radiogenomics == 'T4'] = 3
    c_radiogenomics[stages_radiogenomics == 'Tis'] = 0
    c_radiogenomics = c_radiogenomics.astype('int32')
    c_radiogenomics = c_radiogenomics[subj_ids]
    t_radiogenomics = np.delete(t_radiogenomics, IGNORED_RADIOGENOMICS_PATIENTS)
    d_radiogenomics = np.delete(d_radiogenomics, IGNORED_RADIOGENOMICS_PATIENTS)
    c_radiogenomics = np.delete(c_radiogenomics, IGNORED_RADIOGENOMICS_PATIENTS)

    clinical_data_basel = pd.read_csv('../datasets/nsclc_lung/clinical_data_basel.csv')
    t_basel = clinical_data_basel['Survival time'].values.astype('float32')
    d_basel = clinical_data_basel['Event'].values.astype('float32')
    c_basel = np.zeros_like(d_basel)
    t_basel = np.delete(t_basel, IGNORED_BASEL_PATIENTS)
    d_basel = np.delete(d_basel, IGNORED_BASEL_PATIENTS)
    c_basel = np.delete(c_basel, IGNORED_BASEL_PATIENTS)

    X = np.concatenate((radiomic_features_lung1, radiomic_features_radiogenomics, basel_features_radiogenomics), axis=0)
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float64)

    t = np.concatenate((t_lung1, t_radiogenomics, t_basel))
    d = np.concatenate((d_lung1, d_radiogenomics, d_basel))
    c = np.concatenate((c_lung1, c_radiogenomics, c_basel))

    t = t / np.max(t) + 0.001
    # Train-test split
    X_train, X_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=0.25,
                            random_state=seed, stratify=np.digitize(t, np.quantile(t, np.array([0.3, 0.5, 0.75, 0.9]))))

    return X_train, X_test, X_test, t_train, t_test, t_test, d_train, d_test, d_test, c_train, c_test, c_test
