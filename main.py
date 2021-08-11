import time
import argparse
from pathlib import Path
import yaml
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import uuid
import math
import pandas as pd

from eval_utils import cindex, calibration
from eval_utils import rae as RAE

import os

from sklearn.preprocessing import StandardScaler

import models.Ours.utils as utils
from models.Ours.model import GMM_Survival

from datasets.survivalMNIST.survivalMNIST_data import generate_surv_MNIST
from datasets.simulations import simulate_nonlin_profile_surv
from datasets.support.support_data import generate_support
from datasets.hgg.hgg_data import generate_hgg, generate_hgg_full
from datasets.hemodialysis.hemo_data import generate_hemo

from plotting import plot_overall_kaplan_meier, plot_group_kaplan_meier, plot_group_coxph, plotting_setup

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

# project-wide constants:
ROOT_LOGGER_STR = "GMM_Survival"
LOGGER_RESULT_FILE = "logs.txt"
CHECKPOINT_PATH = 'models/Ours'

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def get_data(args, configs, val=False):
    # Data loaders
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
        # Normalisation
        x_test = x_test / 255.
        if val:
            x_valid = x_valid / 255.
        x_train = x_train / 255.
   
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
                                                                                           xrange=[-.5, .5], plot=False)
        # Normalisation
        t = t / np.max(t) + 0.001
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                              random_state=args.seed)
    elif args.data == 'support':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_support(seed=args.seed)

    elif args.data == 'flchain':
        data = pd.read_csv('../datasets/flchain/flchain.csv')
        feats = ['age', 'sex', 'sample.yr', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']
        prot = 'sex'
        feats = set(feats)
        feats = list(feats)
        t = data['futime'].values + 1
        d = data['death'].values
        x = data[feats].values
        c = data[prot].values
        X = StandardScaler().fit_transform(x)
        t = t / np.max(t) + 0.001
        x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                              random_state=args.seed)

    elif args.data == 'hgg':
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_hgg(seed=args.seed)

    elif args.data == 'hemo':
        c = configs['training']['num_clusters']
        x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, c_train, c_valid, c_test = \
            generate_hemo(seed=args.seed, label=c)

    else:
        NotImplementedError('This dataset is not supported!')

    # Wrap t, d, and c together
    y_train = np.stack([t_train, d_train, c_train], axis=1)
    if val:
        y_valid = np.stack([t_valid, d_valid, c_valid], axis=1)
    y_test = np.stack([t_test, d_test, c_test], axis=1)

    if val:
        return x_train, x_valid, x_test, y_train, y_valid, y_test
    else:
        return x_train, x_test, x_test, y_train, y_test, y_test


# Reconstruction losses
def loss_reconstruction_mnist(inp, x_decoded_mean):
    x = inp
    # NB: transpose to make the first dimension correspond to MC samples
    x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
    loss = 784 * tf.math.reduce_mean(tf.stack([tf.keras.losses.BinaryCrossentropy()(x, x_decoded_mean[i])
                                               for i in range(x_decoded_mean.shape[0])], axis=-1),axis=-1)
    return loss


def loss_reconstruction_sim(inp, x_decoded_mean):
    x = inp
    # NB: transpose to make the first dimension correspond to MC samples
    x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
    loss = 1000 * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                               for i in range(x_decoded_mean.shape[0])], axis=-1),axis=-1)
    return loss


def loss_reconstruction_support(inp, x_decoded_mean):
    x = inp
    # NB: transpose to make the first dimension correspond to MC samples
    x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
    loss = 59 * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                               for i in range(x_decoded_mean.shape[0])], axis=-1),axis=-1)
    return loss


def loss_reconstruction_flchain(inp, x_decoded_mean):
    x = inp
    # NB: transpose to make the first dimension correspond to MC samples
    x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
    loss = 8 * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                               for i in range(x_decoded_mean.shape[0])], axis=-1),axis=-1)
    return loss


def loss_reconstruction_hgg(inp, x_decoded_mean):
    x = inp
    # NB: transpose to make the first dimension correspond to MC samples
    x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
    loss = 147 * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                               for i in range(x_decoded_mean.shape[0])], axis=-1),axis=-1)
    return loss


def loss_reconstruction_hemo(inp, x_decoded_mean):
    x = inp
    # NB: transpose to make the first dimension correspond to MC samples
    x_decoded_mean = tf.transpose(x_decoded_mean, perm=[1, 0, 2])
    loss = 57 * tf.math.reduce_mean(tf.stack([tf.keras.losses.MeanSquaredError()(x, x_decoded_mean[i])
                                               for i in range(x_decoded_mean.shape[0])], axis=-1),axis=-1)
    return loss


# Metrics to track
def accuracy_metric(inp, p_c_z):
    y = inp[:, 2]
    y_pred = tf.math.argmax(p_c_z, axis=-1)
    return tf.numpy_function(normalized_mutual_info_score, [y, y_pred], tf.float64)


def cindex_metric(inp, risk_scores):
    # Evaluates the concordance index based on provided predicted risk scores, computed using hard clustering
    # assignments.
    t = inp[:, 0]
    d = inp[:, 1]
    if tf.reduce_any(tf.math.is_nan(risk_scores)):
        Warning("NaNs in risk scores!")
        return tf.numpy_function(cindex, [t, d, tf.zeros_like(risk_scores)], tf.float64)
    else:
        return tf.numpy_function(cindex, [t, d, risk_scores], tf.float64)


def pretrain(model, args, ex_name, configs):
    input_shape = configs['training']['inp_shape']
    num_clusters = configs['training']['num_clusters']
    learn_prior = configs['training']['learn_prior']

    # Get the AE from the model
    input = tfkl.Input(shape=input_shape)
    z, _ = model.encoder(input)
    dec = model.decoder(z)

    autoencoder = tfk.Model(inputs=input, outputs=dec)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # , decay=args.decay)
    autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy")
    autoencoder.summary()
    x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(args, configs)
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train[:, 2], y_test[:, 2]))

    # If the model should be run from scratch:
    if args.compute_pretrain_weights:
        print('\n******************** Pretraining **************************')
        project_dir = Path(__file__).absolute().parent
        os.makedirs(os.path.join(project_dir, "models/Ours/pretrain", "autoencoder_tmp", ex_name))
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(project_dir, "models/Ours/pretrain", "autoencoder_tmp", ex_name, "cp.ckpt"),
                                                         save_weights_only=True, verbose=1)
        inp_enc = X
        autoencoder.fit(inp_enc, X, epochs=args.epochs_pretrain, batch_size=32, callbacks=cp_callback)

        encoder = model.encoder
        input = tfkl.Input(shape=input_shape)
        z, _ = encoder(input)
        z_model = tf.keras.models.Model(inputs=input, outputs=z)
        z = z_model.predict(X)

        estimator = GaussianMixture(n_components=num_clusters, covariance_type='diag', n_init=3)
        estimator.fit(z)
        tmp = ex_name + "_gmm_save.sav"
        pickle.dump(estimator, open(project_dir / "models/Ours/pretrain" / "gmm_tmp" / tmp, 'wb'))

        print('\n******************** Pretraining Done**************************')
    else:
        #assigning autoencoder weights
        if args.data == 'MNIST':
            try:
                autoencoder.load_weights("models/Ours/pretrain/MNIST/autoencoder/cp.ckpt")
                estimator = pickle.load(open("models/Ours/pretrain/MNIST/gmm_save.sav", 'rb'))
                print('\n******************** Loaded MNIST Pretrain Weights **************************')
            except:
                print("Pretrained weights are not available. Change them in the code or set --pretrain True. ")
                exit(1)
        else:
            print('\nPretrained weights for {} not available, please rerun with \'--pretrain True option\''.format(
                args.data))
            exit(1)

    encoder = model.encoder
    input = tfkl.Input(shape=input_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)

    # Assign weights to GMM mixtures of VaDE
    prior_samples = estimator.weights_
    mu_samples = estimator.means_
    sigma_samples = estimator.covariances_
    prior_samples = prior_samples.reshape((num_clusters))
    model.c_mu.assign(mu_samples)
    model.log_c_sigma.assign(np.log(sigma_samples))
    if learn_prior:
        model.prior_logits.assign(prior_samples)

    yy = estimator.predict(z_model.predict(X))
    acc = utils.cluster_acc(yy, Y)
    pretrain_acc = acc
    print('\nPretrain accuracy: ' + str(acc))

    return model, pretrain_acc


# Runs the experiment
def run_experiment(args, configs, loss):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    project_dir = Path(__file__).absolute().parent
    # Set paths
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = args.results_dir / configs['data']['data_name'] / ex_name
    experiment_path.mkdir(parents=True)

    print(experiment_path)

    acc_tot = []
    nmi_tot = []
    ari_tot = []
    ci_tot = []

    # Override the survival argument
    configs['training']['survival'] = args.survival

    for i in range(args.runs):
        # Generate a new dataset each run
        x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(args, configs)

        model = GMM_Survival(**configs['training'])

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, decay=args.decay)

        os.makedirs(os.path.join(project_dir, 'models/Ours/logs', ex_name))
        cp_callback = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(project_dir, 'models/Ours/logs', ex_name))]

        if args.save_model:
            checkpoint_path = CHECKPOINT_PATH + '/' + configs['data']['data_name'] + '/' + ex_name
            cp_callback = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(project_dir, 'models/Ours/logs', ex_name)),
                           tf.keras.callbacks.ModelCheckpoint(
                               filepath=checkpoint_path,
                               verbose=1,
                               save_weights_only=True,
                               period=10)]

        def learning_rate_scheduler(epoch):
            initial_lrate = args.lr
            drop = args.decay_rate
            epochs_drop = args.epochs_lr
            lrate = initial_lrate * math.pow(drop,
                                             math.floor((1 + epoch) / epochs_drop))
            return lrate

        if args.lrs:
            cp_callback = [tf.keras.callbacks.TensorBoard(log_dir='logs/' + ex_name),
                           tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)]
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

        model.compile(optimizer, loss={"output_1": loss}, metrics={"output_4": accuracy_metric,
                                                                   "output_5": cindex_metric})

        # Sometimes the model gets stuck in a local minimum, pretraining can prevent this...
        # pretrain model
        if args.pretrain:
            model, pretrain_acc = pretrain(model, args, ex_name, configs)

        # create data generators
        model.fit((x_train, y_train), {"output_1": x_train, "output_4": y_train,
                                       "output_5": np.stack([y_train[:, 0], y_train[:, 1]], axis=1)},
                  validation_data=((x_valid, y_valid), {"output_1": x_valid, "output_4": y_valid,
                                                        "output_5": np.stack([y_valid[:, 0], y_valid[:, 1]], axis=1)}),
                  batch_size=args.batch_size,
                  callbacks=cp_callback,
                  epochs=args.num_epochs)

        if args.save_model:
            checkpoint_path = CHECKPOINT_PATH / configs['data']['data_name'] / ex_name
            model.save_weights(checkpoint_path)

        print("\n" * 2)
        print("Evaluation")
        print("\n" * 2)

        # NB: don't use MC samples to predict survival at evaluation
        model.sample_surv = False

        # Training set performance
        tf.keras.backend.set_value(model.use_t, np.array([1.0]))
        rec, z_sample, p_z_c, p_c_z, risk_scores = model.predict((x_train, y_train), batch_size=100)

        yy = np.argmax(p_c_z, axis=-1)
        
        acc = utils.cluster_acc(y_train[:,2], yy)
        nmi = normalized_mutual_info_score(y_train[:,2], yy)
        ari = adjusted_rand_score(y_train[:,2], yy)
        ci = cindex(t=y_train[:, 0], d=y_train[:, 1], scores_pred=risk_scores)

        t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
        rae_nc = RAE(t_pred=t_pred_med[y_train[:, 1] == 1], t_true=y_train[y_train[:, 1] == 1, 0],
                     cens_t=1 - y_train[y_train[:, 1] == 1, 1])
        rae_c = RAE(t_pred=t_pred_med[y_train[:, 1] == 0], t_true=y_train[y_train[:, 1] == 0, 0],
                    cens_t=1 - y_train[y_train[:, 1] == 0, 1])

        if args.data == 'mnist':
            f = open("results_MNIST.txt", "a+")
        elif args.data == 'sim':
            f = open("results_sim.txt", "a+")
        elif args.data == 'support':
            f = open("results_SUPPORT.txt", "a+")
        elif args.data == 'flchain':
            f = open("results_flchain.txt", "a+")
        elif args.data == 'hgg':
            f = open("results_HGG.txt", "a+")
        elif args.data == 'hemo':
            f = open("results_hemo.txt", "a+")
        f.write("Epochs= %d, batch_size= %d, latent_dim= %d, mc samples= %d, learning_rate= %f, decay= %f, name= %s, survival= %s, "
                "seed= %d.\n"
                % (args.num_epochs, args.batch_size, configs['training']['latent_dim'], configs['training']['monte_carlo'], args.lr, args.decay, ex_name, args.survival,
                   args.seed))

        f.write("Train (w t)    |   Accuracy: %f, NMI: %f, ARI: %f. CI: %f, RAE (nc.): %f, RAE (c.): %f.\n" % (acc, nmi, ari, ci, rae_nc, rae_c))

        # Some extra analysis
        if args.data == 'support':
            plot_group_coxph(x=x_train, t=y_train[:, 0], d=y_train[:, 1], c=yy, dir="")
        elif args.data == 'flchain':
            plot_group_coxph(x=x_train, t=y_train[:, 0], d=y_train[:, 1], c=yy, dir="")
        
        tf.keras.backend.set_value(model.use_t, np.array([0.0])) 
        rec, z_sample, p_z_c, p_c_z, risk_scores = model.predict((x_train, y_train), batch_size=100)
        yy = np.argmax(p_c_z, axis=-1)
        acc = utils.cluster_acc(y_train[:, 2], yy)
        nmi = normalized_mutual_info_score(y_train[:, 2], yy)
        ari = adjusted_rand_score(y_train[:, 2], yy)
        ci = cindex(t=y_train[:, 0], d=y_train[:, 1], scores_pred=risk_scores)
        t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
        rae_nc = RAE(t_pred=t_pred_med[y_train[:, 1] == 1], t_true=y_train[y_train[:, 1] == 1, 0],
                     cens_t=1 - y_train[y_train[:, 1] == 1, 1])
        rae_c = RAE(t_pred=t_pred_med[y_train[:, 1] == 0], t_true=y_train[y_train[:, 1] == 0, 0],
                    cens_t=1 - y_train[y_train[:, 1] == 0, 1])
        f.write("Train (w/o t)  |   Accuracy: %f, NMI: %f, ARI: %f. CI: %f, RAE (nc.): %f, RAE (c.): %f.\n" % (acc, nmi, ari, ci, rae_nc, rae_c))

        # Test set performance
        tf.keras.backend.set_value(model.use_t, np.array([1.0]))

        rec, z_sample, p_z_c, p_c_z, risk_scores = model.predict((x_test, y_test), batch_size=100)

        yy = np.argmax(p_c_z, axis=-1)
        acc = utils.cluster_acc(y_test[:,2], yy)
        nmi = normalized_mutual_info_score(y_test[:,2], yy)
        ari = adjusted_rand_score(y_test[:,2], yy)
        ci = cindex(t=y_test[:, 0], d=y_test[:, 1], scores_pred=risk_scores)
        t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
        rae_nc = RAE(t_pred=t_pred_med[y_test[:, 1] == 1], t_true=y_test[y_test[:, 1] == 1, 0],
                     cens_t=1 - y_test[y_test[:, 1] == 1, 1])
        rae_c = RAE(t_pred=t_pred_med[y_test[:, 1] == 0], t_true=y_test[y_test[:, 1] == 0, 0],
                    cens_t=1 - y_test[y_test[:, 1] == 0, 1])
        acc_tot.append(acc)
        nmi_tot.append(nmi)
        ari_tot.append(ari)
        ci_tot.append(ci)

        f.write("Test (w t)     |   Accuracy: %f, NMI: %f, ARI: %f. CI: %f, RAE (nc.): %f, RAE (c.): %f.\n" % (acc, nmi, ari, ci, rae_nc, rae_c))

        # Some extra analysis to explore clusters...
        if args.data == 'hgg':
            x_full, t_full, d_full, c_full = generate_hgg_full()
            y_full = np.stack([t_full, d_full, c_full], axis=1)
            rec_full, z_sample_full, p_z_c_full, p_c_z_full, risk_scores_full = model.predict((x_full, y_full), batch_size=100)
            yy_full = np.argmax(p_c_z_full, axis=-1)
            np.savetxt(fname="c_hat.csv", X=yy_full)
            plotting_setup(16)
            plot_group_kaplan_meier(t=y_full[:, 0], d=y_full[:, 1], c=yy_full, dir="")
        elif args.data == 'mnist':
            utils.save_mnist_reconstructions(recs=rec, x=x_test, y=y_test)
            utils.save_mnist_generated_samples(model=model)

        tf.keras.backend.set_value(model.use_t, np.array([0.0]))
        rec, z_sample, p_z_c, p_c_z, risk_scores = model.predict((x_test, y_test), batch_size=100)

        yy = np.argmax(p_c_z, axis=-1)
        acc = utils.cluster_acc(y_test[:, 2], yy)
        nmi = normalized_mutual_info_score(y_test[:, 2], yy)
        ari = adjusted_rand_score(y_test[:, 2], yy)
        ci = cindex(t=y_test[:, 0], d=y_test[:, 1], scores_pred=risk_scores)
        print(risk_scores)
        t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
        rae_nc = RAE(t_pred=t_pred_med[y_test[:, 1] == 1], t_true=y_test[y_test[:, 1] == 1, 0],
                     cens_t=1 - y_test[y_test[:, 1] == 1, 1])
        rae_c = RAE(t_pred=t_pred_med[y_test[:, 1] == 0], t_true=y_test[y_test[:, 1] == 0, 0],
                    cens_t=1 - y_test[y_test[:, 1] == 0, 1])

        if args.eval_cal:
            # NOTE: calibration evaluation is quite slow
            t_sample = utils.sample_weibull(scales=risk_scores, shape=model.weibull_shape)
            cal = calibration(predicted_samples=t_sample, t=y_test[:, 0], d=y_test[:, 1])
        else:
            cal = np.nan

        f.write("Test (w/o t)   |   Accuracy: %f, NMI: %f, ARI: %f. CI: %f, RAE (nc.): %f, RAE (c.): %f, CAL: %f.\n" % (acc, nmi, ari, ci, rae_nc, rae_c, cal))
        tf.keras.backend.set_value(model.use_t, np.array([1.0]))

        f.close()
        print(str(acc))
        print(str(nmi))
        print(str(ari))
        print(str(ci))
        print("(" + str(rae_nc) +"; " + str(rae_c) + ")")

    if args.runs > 1:

        acc_tot = np.array(acc_tot)
        nmi_tot = np.array(nmi_tot)
        ari_tot = np.array(ari_tot)
        ci_tot = np.array(ci_tot)

        if args.data == 'mnist':
            f = open("results_MNIST.txt", "a+")
        elif args.data == 'sim':
            f = open("results_sim.txt", "a+")
        elif args.data == 'support':
            f = open("results_SUPPORT.txt", "a+")
        elif args.data == 'flchain':
            f = open("results_flchain.txt", "a+")
        elif args.data == 'hgg':
            f = open("results_HGG.txt", "a+")
        elif args.data == 'hemo':
            f = open("results_hemo.txt", "a+")

        f.write("Epochs= %d, batch_size= %d, latent_space= %d, mc samples= %d, learning_rate= %f, decay= %f, runs= %d, name= %s, "
                "seed= %d.\n "
                % (args.num_epochs, args.batch_size, configs['training']['latent_dim'], configs['training']['monte_carlo'], args.lr, args.decay, args.runs, ex_name,
                   args.seed))
        f.write("Accuracy: %f std %f, NMI: %f std %f, ARI: %f std %f, CI: %f std %f. \n" % (
            np.mean(acc_tot), np.std(acc_tot), np.mean(nmi_tot), np.std(nmi_tot), np.mean(ari_tot), np.std(ari_tot),
            np.mean(ci_tot), np.std(ci_tot)))


def main():
    project_dir = Path(__file__).absolute().parent
    print(project_dir)

    parser = argparse.ArgumentParser()

    # parameters of the model
    parser.add_argument('--data',
                        default='mnist',
                        type=str,
                        choices=['mnist', 'sim',  'support', 'flchain', 'hgg', 'hemo'],
                        help='specify the data (mnist, sim, support, flchain, hgg, hemo)')
    parser.add_argument('--num_epochs',
                        default=1000,
                        type=int,
                        help='specify the number of epochs')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='specify the batch size')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='specify learning rate')
    parser.add_argument('--decay',
                        default=0.00001,
                        type=float,
                        help='specify decay')
    parser.add_argument('--weibull_shape',
                        default=1.0,
                        type=float,
                        help='specify Weibull shape parameter')
    parser.add_argument('--decay_rate',
                        default=0.9,
                        type=float,
                        help='specify decay')
    parser.add_argument('--epochs_lr',
                        default=10,
                        type=int,
                        help='specify decay')
    parser.add_argument('--lrs',
                        default=False,
                        type=bool,
                        help='specify decay')
    parser.add_argument('--no-survival',
                        dest='survival',
                        action='store_false',
                        help='specify if the survival model should not be included')
    parser.add_argument('--eval-cal',
                        default=False,
                        type=bool,
                        help='specify if the calibration needs to be evaluated')
    parser.set_defaults(survival=True)

    # other parameters
    parser.add_argument('--runs',
                        default=1,
                        type=int,
                        help='number of runs, the results will be averaged')
    parser.add_argument('--results_dir',
                        default=os.path.join(project_dir, 'models/Ours/experiments'),
                        type=lambda p: Path(p).absolute(),
                        help='specify the folder where the results get saved')
    parser.add_argument('--pretrain', default=False, type=bool,
                        help='True to pretrain the autoencoder.')
    parser.add_argument('--compute_pretrain_weights', default=False, type=bool,
                        help='True to pretrain the autoencoder, False to use pretrained weights')
    parser.add_argument('--epochs_pretrain', default=10, type=int,
                        help='Specify the number of pre-training epochs')
    parser.add_argument('--save_model', default=False, type=bool,
                        help='True to save the model')
    parser.add_argument("--ex_name", default="", type=str, help="Specify experiment name")
    parser.add_argument("--config_override", default="", type=str, help="Specify config.yml override file")
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='random number generator seed')
    args = parser.parse_args()

    if args.data == "mnist":
        config_path = project_dir / 'models' / 'Ours' / 'configs' / 'MNIST.yml'
        loss = loss_reconstruction_mnist
    elif args.data == "sim":
        config_path = project_dir / 'models' / 'Ours' / 'configs' / 'sim.yml'
        loss = loss_reconstruction_sim
    elif args.data == "support":
        config_path = project_dir / 'models' / 'Ours' / 'configs' / 'support.yml'
        loss = loss_reconstruction_support
    elif args.data == "flchain":
        config_path = project_dir / 'models' / 'Ours' / 'configs' / 'flchain.yml'
        loss = loss_reconstruction_flchain
    elif args.data == "hgg":
        config_path = project_dir / 'models' / 'Ours' / 'configs' / 'hgg.yml'
        loss = loss_reconstruction_hgg
    elif args.data == "hemo":
        config_path = project_dir / 'models' / 'Ours' / 'configs' / 'hemo.yml'
        loss = loss_reconstruction_hemo

    # Check for config override
    if args.config_override is not "":
        config_path = Path(args.config_override)

    with config_path.open(mode='r') as yamlfile:
        configs = yaml.safe_load(yamlfile)

    run_experiment(args, configs, loss)


if __name__ == "__main__":
    main()
