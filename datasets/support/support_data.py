# Code from Chapfuwa et al.: https://github.com/paidamoyo/survival_cluster_analysis
import os

import numpy as np
import pandas

from preproc_utils import one_hot_encoder, formatted_data, missing_proportion, \
    one_hot_indices, get_train_median_mode, log_transform

from sklearn.preprocessing import StandardScaler

# TODO Check scoma attribute
# TODO You may want to use the following normal values that have been found to
# be satisfactory in imputing missing baseline physiologic data:
# http://biostat.mc.vanderbilt.edu/wiki/Main/DataSets
# http://annals.org/aim/article/708396/support-prognostic-model-objective-estimates-survival-seriously-ill-hospitalized-adults
# To develop models without using findings from previous models, be sure not to use aps, sps, surv2m, surv6m as predictors.
# You also will probably not want to use prg2m, prg6m, dnr, dnrday.
# Columns with na
#   "edu"     "scoma"   "charges" "totcst"  "totmcst" "avtisst" "sps"     "aps"     "surv2m"  "surv6m"
# "prg2m"   "prg6m"   "dnrday"  "meanbp"  "wblc"    "hrt"     "resp"    "temp"    "pafi"    "alb"
#  "bili"    "crea"    "sod"     "ph"      "glucose" "bun"     "urine"   "adlp"    "adls"

# # Log transform: totmcst, totcst, charges, pafi, sod


def generate_data(seed=42):
    np.random.seed(seed)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'support2.csv'))
    print("path:{}".format(path))
    data_frame = pandas.read_csv(path, index_col=0)
    to_drop = ['hospdead', 'death', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'd.time', 'aps', 'sps', 'surv2m', 'surv6m',
               'totmcst']
    print("head of data:{}, data shape:{}".format(data_frame.head(), data_frame.shape))
    print("missing:{}".format(missing_proportion(data_frame.drop(labels=to_drop, axis=1))))
    # Preprocess
    one_hot_encoder_list = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'sfdm2']
    data_frame = one_hot_encoder(data=data_frame, encode=one_hot_encoder_list)

    data_frame = log_transform(data_frame, transform_ls=['totmcst', 'totcst', 'charges', 'pafi', 'sod'])
    print("na columns:{}".format(data_frame.columns[data_frame.isnull().any()].tolist()))
    t_data = data_frame[['d.time']]
    e_data = data_frame[['death']]
    # dzgroup roughly corresponds to the diagnosis; more fine-grained than dzclass
    c_data = data_frame[['death']]
    c_data['death'] = c_data['death'].astype('category')
    c_data['death'] = c_data['death'].cat.codes

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


def generate_support(seed=42):
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
