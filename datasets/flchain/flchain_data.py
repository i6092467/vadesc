# Based on the code from Chapfuwa et al.: https://github.com/paidamoyo/survival_cluster_analysis
import os

import numpy as np
import pandas

from baselines.sca.sca_utils.pre_processing import one_hot_encoder, formatted_data, missing_proportion, \
    one_hot_indices, get_train_median_mode

from sklearn.preprocessing import StandardScaler

# age: age in years
# sex: F=female, M=male
# sample.yr: the calendar year in which a blood sample was obtained
# kappa: serum free light chain, kappa portion
# lambda: serum free light chain, lambda portion
# flc.grp: the FLC group for the subject, as used in the original analysis
# creatinine: serum creatinine
# mgus: 1 if the subject had been diagnosed with monoclonal gammapothy (MGUS)
# futime: days from enrollment until death. Note that there are 3 subjects whose sample was obtained on their death date.
# death 0=alive at last contact date, 1=dead
# chapter: for those who died, a grouping of their primary cause of death by chapter headings of
# the International Code of Diseases ICD-9


def generate_data(seed):
    np.random.seed(seed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'flchain.csv'))
    print("path:{}".format(path))
    data_frame = pandas.read_csv(path, index_col=0)
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    # x_data = data_frame[['age', 'sex', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']]
    # Preprocess
    to_drop = ['futime', 'death', 'chapter']
    print("missing:{}".format(missing_proportion(data_frame.drop(labels=to_drop, axis=1))))
    one_hot_encoder_list = ['sex', 'flc.grp', 'sample.yr']
    data_frame = one_hot_encoder(data_frame, encode=one_hot_encoder_list)
    t_data = data_frame[['futime']]
    e_data = data_frame[['death']]
    c_data = data_frame[['death']]
    c_data['death'] = c_data['death'].astype('category')
    c_data['death'] = c_data['death'].cat.codes
    dataset = data_frame.drop(labels=to_drop, axis=1)
    print("head of dataset data:{}, data shape:{}".format(dataset.head(), dataset.shape))
    encoded_indices = one_hot_indices(dataset, one_hot_encoder_list)
    include_idx = set(np.array(sum(encoded_indices, [])))
    mask = np.array([(i in include_idx) for i in np.arange(dataset.shape[1])])
    print("data description:{}".format(dataset.describe()))
    covariates = np.array(dataset.columns.values)
    print("columns:{}".format(covariates))
    x = np.array(dataset).reshape(dataset.shape)
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

    imputation_values = get_train_median_mode(x=np.array(x[train_idx]), categorial=encoded_indices)
    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx, imputation_values=imputation_values),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx, imputation_values=imputation_values),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx, imputation_values=imputation_values)
    }

    preprocessed['train']['c'] = c[train_idx]
    preprocessed['valid']['c'] = c[valid_idx]
    preprocessed['test']['c'] = c[test_idx]

    return preprocessed


def generate_flchain(seed=42):
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
