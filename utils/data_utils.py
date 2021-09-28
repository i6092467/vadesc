# Some utility functions for data handling
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

from datasets.survivalMNIST.survivalMNIST_data import generate_surv_MNIST
from datasets.simulations import simulate_nonlin_profile_surv
from datasets.support.support_data import generate_support
from datasets.hgg.hgg_data import generate_hgg, generate_hgg_full
from datasets.hemodialysis.hemo_data import generate_hemo
from datasets.nsclc_lung.nsclc_lung_data import generate_lung1_images, generate_radiogenomics_images, \
    generate_radiogenomics_images_amc, generate_lung3_images, generate_basel_images, generate_radiomic_features
from datasets.nsclc_lung.CT_preproc_utils import augment_images


class DataGen(tf.keras.utils.Sequence):

    def __init__(self, X, y, num_classes, ae=False, ae_class=False, batch_size=32, shuffle=True, augment=False):
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.ae = ae
        self.ae_class = ae_class
        self.num_classes = num_classes
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            inds = np.arange(len(self.X))
            np.random.shuffle(inds)
            self.X = self.X[inds]
            self.y = self.y[inds]

    def __getitem__(self, index):
        X = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        y = self.y[index * self.batch_size:(index + 1) * self.batch_size]
        # augmentation
        if self.augment:
            X = augment_images(X)
        if self.ae:
            return X, {'dec': X}
        elif self.ae_class:
            c = to_categorical(y[:, 2], self.num_classes)
            return X, {'dec': X, 'classifier': c}
        else:
            return (X, y), {"output_1": X, "output_4": y, "output_5": y}

    def __len__(self):
        return len(self.X) // self.batch_size


def get_gen(X, y, configs, batch_size, validation=False, ae=False, ae_class=False):
    num_clusters = configs['training']['num_clusters']
    input_dim = configs['training']['inp_shape']
    if isinstance(input_dim, list) and validation==False:
        if ae_class:
            data_gen = DataGen(X, y, 4, augment=True, ae=ae, ae_class=ae_class, batch_size=batch_size)
        else:
            data_gen = DataGen(X, y, num_clusters, augment=True, ae=ae, ae_class=ae_class, batch_size=batch_size)
    else:
        if ae_class:
            data_gen = DataGen(X, y, 4, ae=ae, ae_class=ae_class, batch_size=batch_size)
        else:
            data_gen = DataGen(X, y, num_clusters, ae=ae, ae_class=ae_class, batch_size=batch_size)
    return data_gen


def get_data(args, configs, val=False):
    if args.data == 'mnist':
        valid_perc = .15
        if not val:
            valid_perc = .0
        if val:
            x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
                generate_surv_MNIST(n_groups=5, seed=args.seed, p_cens=.3, valid_perc=valid_perc)
        else:
            x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = generate_surv_MNIST(n_groups=5,
                                                                                                     seed=args.seed,
                                                                                                     p_cens=.3,
                                                                                                     valid_perc=valid_perc)
            x_valid = x_test
            t_valid = t_test
            c_valid = c_test
        # Normalisation
        x_test = x_test / 255.
        if val:
            x_valid = x_valid / 255.
        x_train = x_train / 255.
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_valid)
        dat_label_test = np.zeros_like(t_test)

    elif args.data == "sim":
        X, t, d, c, Z, mus, sigmas, betas, betas_0, mlp_dec = simulate_nonlin_profile_surv(p=1000, n=60000,
                                                                                           latent_dim=16, k=3,
                                                                                           p_cens=.3, seed=args.seed,
                                                                                           clust_mean=True,
                                                                                           clust_cov=True,
                                                                                           clust_coeffs=True,
                                                                                           clust_intercepts=True,
                                                                                           balanced=True,
                                                                                           weibull_k=1,
                                                                                           brange=[-10.0, 10.0],
                                                                                           isotropic=True,
                                                                                           xrange=[-.5, .5])
        # Normalisation
        t = t / np.max(t) + 0.001
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                              random_state=args.seed)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_test)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == "support":
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_support(seed=args.seed)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_valid)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == "flchain":
        data = pd.read_csv('../baselines/DCM/data/flchain.csv')
        feats = ['age', 'sex', 'sample.yr', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']
        prot = 'sex'
        feats = set(feats)
        feats = list(feats)  # - set([prot]))
        t = data['futime'].values + 1
        d = data['death'].values
        x = data[feats].values
        c = data[prot].values
        X = StandardScaler().fit_transform(x)
        t = t / np.max(t) + 0.001
        x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                              random_state=args.seed)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_train)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == "hgg":
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_hgg(seed=args.seed)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_valid)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == 'hemo':
        c = configs['training']['num_clusters']
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_hemo(seed=args.seed, label=c)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_valid)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == 'nsclc_features':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_radiomic_features(n_slices=11, dsize=[256, 256], seed=args.seed)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_valid)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == 'lung1':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_lung1_images(dsize=(configs['training']['inp_shape'][0], configs['training']['inp_shape'][1]),
                                  n_slices=configs['training']['n_slices'], seed=args.seed)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_valid)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == 'basel':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_basel_images(dsize=(configs['training']['inp_shape'][0], configs['training']['inp_shape'][1]),
                                  n_slices=configs['training']['n_slices'], seed=args.seed, normalise_t=False)
        dat_label_train = np.zeros_like(t_train)
        dat_label_valid = np.zeros_like(t_valid)
        dat_label_test = np.zeros_like(t_test)
    elif args.data == 'nsclc':
        x_train_l, x_valid_l, x_test_l, t_train_l, t_valid_l, t_test_l, d_train_l, d_valid_l, d_test_l, c_train_l, c_valid_l, c_test_l = \
            generate_lung1_images(dsize=(configs['training']['inp_shape'][0], configs['training']['inp_shape'][1]),
                                  n_slices=configs['training']['n_slices'], seed=args.seed, normalise_t=False)
        x_train_r, x_valid_r, x_test_r, t_train_r, t_valid_r, t_test_r, d_train_r, d_valid_r, d_test_r, c_train_r, c_valid_r, c_test_r = \
            generate_radiogenomics_images(dsize=(configs['training']['inp_shape'][0], configs['training']['inp_shape'][1]),
                                          n_slices=configs['training']['n_slices'], seed=args.seed, normalise_t=False)
        x_train_ra, x_valid_ra, x_test_ra, t_train_ra, t_valid_ra, t_test_ra, d_train_ra, d_valid_ra, d_test_ra, c_train_ra, c_valid_ra, c_test_ra = \
            generate_radiogenomics_images_amc(dsize=(configs['training']['inp_shape'][0], configs['training']['inp_shape'][1]),
                                              n_slices=configs['training']['n_slices'], seed=args.seed, normalise_t=False)
        x_train_l3, x_valid_l3, x_test_l3, t_train_l3, t_valid_l3, t_test_l3, d_train_l3, d_valid_l3, d_test_l3, c_train_l3, c_valid_l3, c_test_l3 = \
            generate_lung3_images(dsize=(configs['training']['inp_shape'][0], configs['training']['inp_shape'][1]),
                n_slices=configs['training']['n_slices'], seed=args.seed, normalise_t=False)

        x_train_b, x_valid_b, x_test_b, t_train_b, t_valid_b, t_test_b, d_train_b, d_valid_b, d_test_b, c_train_b, c_valid_b, c_test_b = \
            generate_basel_images(dsize=(configs['training']['inp_shape'][0], configs['training']['inp_shape'][1]),
                                  n_slices=configs['training']['n_slices'], seed=args.seed, normalise_t=False)

        x_train = np.concatenate((x_train_l, x_train_r,  x_train_ra, x_train_l3, x_test_l3, x_train_b), axis=0)
        x_valid = np.concatenate((x_test_l, x_test_r, x_test_ra, x_test_b), axis=0)
        x_test = np.concatenate((x_test_l, x_test_r, x_test_ra, x_test_b), axis=0)
        dat_label_train = np.concatenate((np.zeros_like(t_train_l), np.ones_like(t_train_r), 2 * np.ones_like(t_train_ra),
                                          3 * np.ones_like(t_train_l3), 3 * np.ones_like(t_test_l3),
                                          4 * np.ones_like(t_train_b)))
        dat_label_valid = np.concatenate((np.zeros_like(t_test_l), np.ones_like(t_test_r), 2 * np.ones_like(t_test_ra), 4 * np.ones_like(t_test_b)))
        dat_label_test = np.concatenate((np.zeros_like(t_test_l), np.ones_like(t_test_r), 2 * np.ones_like(t_test_ra), 4 * np.ones_like(t_test_b)))
        t_train = np.concatenate((t_train_l, t_train_r, t_train_ra, t_train_l3, t_test_l3, t_train_b), axis=0)
        t_valid = np.concatenate((t_test_l, t_test_r, t_test_ra, t_test_b), axis=0)
        t_test = np.concatenate((t_test_l, t_test_r, t_test_ra, t_test_b), axis=0)
        d_train = np.concatenate((d_train_l, d_train_r, d_train_ra, d_train_l3, d_test_l3, d_train_b), axis=0)
        d_valid = np.concatenate((d_test_l, d_test_r, d_test_ra, d_test_b), axis=0)
        d_test = np.concatenate((d_test_l, d_test_r, d_test_ra, d_test_b), axis=0)
        c_train = np.concatenate((c_train_l, c_train_r, c_train_ra, c_train_l3, c_test_l3, c_train_b), axis=0)
        c_valid = np.concatenate((c_test_l, c_test_r, c_test_ra, c_test_b), axis=0)
        c_test = np.concatenate((c_test_l, c_test_r, c_test_ra, c_test_b), axis=0)

        t_max = np.max(np.concatenate((t_train, t_test)))
        t_train = t_train / t_max + 0.001
        t_valid = t_valid / t_max + 0.001
        t_test = t_test / t_max + 0.001
    else:
        NotImplementedError('This dataset is not supported!')

    # Wrap t, d, and c together
    y_train = np.stack([t_train, d_train, c_train, dat_label_train], axis=1)
    if val:
        y_valid = np.stack([t_valid, d_valid, c_valid, dat_label_valid], axis=1)
    y_test = np.stack([t_test, d_test, c_test, dat_label_test], axis=1)

    np.savetxt(fname='y_train_nsclc_' + str(args.seed) + '.csv', X=y_train)
    np.savetxt(fname='y_test_nsclc_' + str(args.seed) + '.csv', X=y_test)

    if val:
        return x_train, x_valid, x_test, y_train, y_valid, y_test
    else:
        return x_train, x_test, x_test, y_train, y_test, y_test


def construct_surv_df(X, t, d):
    p = X.shape[1]
    df = pd.DataFrame(X, columns=["X_" + str(i) for i in range(p)])
    df["time_to_event"] = t
    df["failure"] = d
    return df


