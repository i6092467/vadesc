# Based on the code from Chapfuwa et al.: https://github.com/paidamoyo/survival_cluster_analysis
import os

import numpy as np
import pandas

from baselines.sca.sca_utils.pre_processing import one_hot_encoder, formatted_data, missing_proportion, \
    one_hot_indices, get_train_median_mode, log_transform, impute_missing

from sklearn.preprocessing import StandardScaler


def generate_data(seed=42):
    np.random.seed(seed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'hgg.csv'))
    print("path:{}".format(path))
    data_frame = pandas.read_csv(path, index_col=None)

    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    print("missing:{}".format(missing_proportion(data_frame)))

    one_hot_encoder_list = ['sex', 'sy_group', 'pre_cognition', 'pre_motor_re_arm', 'pre_motor_li_arm',
                            'pre_motor_re_leg', 'pre_motor_li_leg', 'pre_sensibility', 'pre_language',
                            'pre_visualfield', 'pre_seizure', 'pre_headache', 'pre_nausea', 'post_cognition',
                            'post_motor_re_arm', 'post_motor_li_arm', 'post_motor_re_leg', 'post_motor_li_leg',
                            'post_sensibility', 'post_language', 'post_visualfield', 'post_seizure', 'post_headache',
                            'adjuvant_therapy', 'op_type', 'ultrasound', 'io_mri', 'ala', 'io_mapping', 'histology',
                            'antibody', 'idh1_seq', 'idh2_seq', 'mgmt', 'idh_status', 'tumor_side', 'frontal',
                            'central', 'parietal', 'occipital', 'temporal', 'insular', 'limbic', 'central_gray_matter',
                            't1_t2_pre_solid']
    data_frame = one_hot_encoder(data=data_frame, encode=one_hot_encoder_list)
    print("na columns:{}".format(data_frame.columns[data_frame.isnull().any()].tolist()))
    t_data = data_frame[['loss_or_death_d']]
    e_data = 1 - data_frame[['censored']]
    c_data = 1 - data_frame[['censored']]
    c_data['censored'] = c_data['censored'].astype('category')
    c_data['censored'] = c_data['censored'].cat.codes

    to_drop = ['loss_or_death_d', 'censored']
    x_data = data_frame.drop(labels=to_drop, axis=1)

    encoded_indices = one_hot_indices(x_data, one_hot_encoder_list)
    include_idx = set(np.array(sum(encoded_indices, [])))
    mask = np.array([(i in include_idx) for i in np.arange(x_data.shape[1])])
    print("head of x data:{}, data shape:{}".format(x_data.head(), x_data.shape))
    print("data description:{}".format(x_data.describe()))
    covariates = np.array(x_data.columns.values)
    print("columns:{}".format(covariates))
    x = np.array(x_data).reshape(x_data.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))
    c = np.array(c_data).reshape(len(c_data))

    print("x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    c = c[idx]

    # Normalization
    t = t / np.max(t) + 0.001
    scaler = StandardScaler()
    scaler.fit(x[:, ~mask])
    x[:, ~mask] = scaler.transform(x[:, ~mask])

    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))
    print("shuffled x:{}, t:{}, e:{}, len:{}".format(x[0], t[0], e[0], len(t)))
    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]
    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))

    imputation_values = get_train_median_mode(x=x[train_idx], categorial=encoded_indices)

    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx, imputation_values=imputation_values),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx, imputation_values=imputation_values),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx, imputation_values=imputation_values)
    }

    preprocessed['train']['c'] = c[train_idx]
    preprocessed['valid']['c'] = c[valid_idx]
    preprocessed['test']['c'] = c[test_idx]

    return preprocessed


def generate_hgg(seed=42):
    preproc = generate_data(seed)

    x_train = preproc['train']['x']
    x_valid = preproc['valid']['x']
    x_test = preproc['test']['x']

    t_train = preproc['train']['t']
    t_valid = preproc['valid']['t']
    t_test = preproc['test']['t']

    d_train = preproc['train']['e']
    d_valid = preproc['valid']['e']
    d_test = preproc['test']['e']

    c_train = preproc['train']['c']
    c_valid = preproc['valid']['c']
    c_test = preproc['test']['c']

    return x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test


def generate_hgg_full():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'hgg.csv'))
    print("path:{}".format(path))
    data_frame = pandas.read_csv(path, index_col=None)

    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    print("missing:{}".format(missing_proportion(data_frame)))

    one_hot_encoder_list = ['sex', 'sy_group', 'pre_cognition', 'pre_motor_re_arm', 'pre_motor_li_arm',
                            'pre_motor_re_leg', 'pre_motor_li_leg', 'pre_sensibility', 'pre_language',
                            'pre_visualfield', 'pre_seizure', 'pre_headache', 'pre_nausea', 'post_cognition',
                            'post_motor_re_arm', 'post_motor_li_arm', 'post_motor_re_leg', 'post_motor_li_leg',
                            'post_sensibility', 'post_language', 'post_visualfield', 'post_seizure', 'post_headache',
                            'adjuvant_therapy', 'op_type', 'ultrasound', 'io_mri', 'ala', 'io_mapping', 'histology',
                            'antibody', 'idh1_seq', 'idh2_seq', 'mgmt', 'idh_status', 'tumor_side', 'frontal',
                            'central', 'parietal', 'occipital', 'temporal', 'insular', 'limbic', 'central_gray_matter',
                            't1_t2_pre_solid']
    data_frame = one_hot_encoder(data=data_frame, encode=one_hot_encoder_list)
    print("na columns:{}".format(data_frame.columns[data_frame.isnull().any()].tolist()))
    t_data = data_frame[['loss_or_death_d']]
    e_data = 1 - data_frame[['censored']]
    c_data = 1 - data_frame[['censored']]
    c_data['censored'] = c_data['censored'].astype('category')
    c_data['censored'] = c_data['censored'].cat.codes

    to_drop = ['loss_or_death_d', 'censored']
    x_data = data_frame.drop(labels=to_drop, axis=1)

    encoded_indices = one_hot_indices(x_data, one_hot_encoder_list)
    include_idx = set(np.array(sum(encoded_indices, [])))
    mask = np.array([(i in include_idx) for i in np.arange(x_data.shape[1])])
    print("head of x data:{}, data shape:{}".format(x_data.head(), x_data.shape))
    print("data description:{}".format(x_data.describe()))
    covariates = np.array(x_data.columns.values)
    print("columns:{}".format(covariates))
    x = np.array(x_data).reshape(x_data.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))
    c = np.array(c_data).reshape(len(c_data))

    # Normalization
    t = t / np.max(t) + 0.001
    scaler = StandardScaler()
    scaler.fit(x[:, ~mask])
    x[:, ~mask] = scaler.transform(x[:, ~mask])

    imputation_values = get_train_median_mode(x=x, categorial=encoded_indices)
    impute_covariates = impute_missing(data=x, imputation_values=imputation_values)

    return impute_covariates, t, e, c
