import time
from pathlib import Path
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import uuid
import math
from utils.eval_utils import cindex, calibration, accuracy_metric, cindex_metric
from utils.eval_utils import rae as RAE
import os

import utils.utils as utils
from models.model import GMM_Survival


from utils.plotting import plot_group_kaplan_meier, plot_bigroup_kaplan_meier, plot_tsne_by_cluster, \
    plot_tsne_by_survival
from utils.data_utils import get_data, get_gen

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras


def pretrain(model, args, ex_name, configs):
    input_shape = configs['training']['inp_shape']
    num_clusters = configs['training']['num_clusters']
    learn_prior = configs['training']['learn_prior']
    if isinstance(input_shape, list):
        input_shape = [input_shape[0], input_shape[1], 1]
    # Get the AE from the model
    input = tfkl.Input(shape=input_shape)
    z, _ = model.encoder(input)
    if isinstance(input_shape, list):
        z_dec = tf.expand_dims(z, 0)
    else:
        z_dec = z
    dec = model.decoder(z_dec)
    if isinstance(input_shape, list):
        dec = tf.reshape(dec, [-1, input_shape[0], input_shape[1],1])
    dec = tfkl.Lambda(lambda x: x, name="dec")(dec)
    autoencoder = tfk.Model(inputs=input, outputs=dec)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)#, decay=args.decay)
    autoencoder.compile(optimizer=optimizer, loss={"dec":"mse"})
    autoencoder.summary()

    s = tfkl.Dense(4, activation='softmax', name="classifier")(z)
    autoencoder_classifier = tfk.Model(inputs=input, outputs=[dec, s])
    losses = {"dec": "mse", "classifier": "categorical_crossentropy"}
    lossWeights = {'dec': 10.0, "classifier": 1.0}
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    autoencoder_classifier.compile(optimizer=opt, loss=losses, loss_weights=lossWeights,
                                   metrics={"classifier": "accuracy"})
    autoencoder_classifier.summary()

    x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(args, configs)
    gen_train = get_gen(x_train, y_train, configs, args.batch_size, ae_class=True)
    gen_test = get_gen(x_test, y_test, configs, args.batch_size, validation=True, ae_class=True)

    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train[:, 2], y_test[:, 2]))

    project_dir = Path(__file__).absolute().parent
    pretrain_dir = os.path.join(project_dir, 'models/pretrain/' + args.data + "/input_" + str(input_shape[0]) + 'x' + str(input_shape[1])\
                   + '_ldim_' + str(configs['training']['latent_dim']) + '_pretrain_'+ str(args.epochs_pretrain))

    print('\n******************** Pretraining **************************')
    inp_enc = X
    autoencoder_classifier.fit(gen_train, validation_data=gen_test,
                                  epochs=args.epochs_pretrain)#, callbacks=cp_callback)

    encoder = model.encoder
    input = tfkl.Input(shape=input_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)
    z = z_model.predict(X)

    estimator = GaussianMixture(n_components=num_clusters, covariance_type='diag', n_init=3)
    estimator.fit(z)
    print('\n******************** Pretraining Done**************************')

    encoder = model.encoder
    input = tfkl.Input(shape=input_shape)
    z, _ = encoder(input)
    z_model = tf.keras.models.Model(inputs=input, outputs=z)

    # Assign weights to GMM mixtures of VaDE
    prior_samples = estimator.weights_
    mu_samples = estimator.means_
    prior_samples = prior_samples.reshape((num_clusters))
    model.c_mu.assign(mu_samples)
    if learn_prior:
        model.prior_logits.assign(prior_samples)

    yy = estimator.predict(z_model.predict(X))
    acc = utils.cluster_acc(yy, Y)
    pretrain_acc = acc
    print('\nPretrain accuracy: ' + str(acc))

    return model, pretrain_acc


def run_experiment(args, configs, loss):

    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    if args.eager:
        tf.config.run_functions_eagerly(True)

    # Set paths
    project_dir = Path(__file__).absolute().parent
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = args.results_dir / configs['data']['data_name'] / ex_name
    experiment_path.mkdir(parents=True)
    os.makedirs(os.path.join(project_dir, 'models/logs', ex_name))
    print(experiment_path)

    # Override the survival argument
    configs['training']['survival'] = args.survival

    # Generate a new dataset each run
    x_train, x_valid, x_test, y_train, y_valid, y_test = get_data(args, configs)
    gen_train = get_gen(x_train, y_train, configs, args.batch_size)
    gen_test = get_gen(x_test, y_test, configs, args.batch_size, validation=True)

    # Override configs if the baseline DSA should be run
    configs['training']['dsa'] = args.dsa

    # Define model & optimizer
    model = GMM_Survival(**configs['training'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, decay=args.decay)
    cp_callback = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(project_dir, 'models/logs', ex_name))]
    model.compile(optimizer, loss={"output_1": loss}, metrics={"output_4": accuracy_metric,
                                                               "output_5": cindex_metric})
    # The survival time is used for training
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))

    # Pretrain model: the model gets stuck in a local minimum, pretraining can prevent this.
    if args.pretrain:
        model, pretrain_acc = pretrain(model, args, ex_name, configs)

    # Fit model
    model.fit(gen_train, validation_data=gen_test, callbacks=cp_callback, epochs=args.num_epochs)

    # Save model
    if args.save_model:
        checkpoint_path = experiment_path
        print("\nSaving weights at ", experiment_path)
        model.save_weights(checkpoint_path)

    print("\n" * 2)
    print("Evaluation")
    print("\n" * 2)

    # NB: don't use MC samples to predict survival at evaluation
    model.sample_surv = False

    # Training set performance
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((x_train, y_train), batch_size=args.batch_size)
    risk_scores = np.squeeze(risk_scores)
    if args.save_model:
        with open(experiment_path / 'c_train.npy', 'wb') as save_file:
            np.save(save_file, p_c_z)
    yy = np.argmax(p_c_z, axis=-1)
    if args.dsa:
        km_dsa = KMeans(n_clusters=args.dsa_k, random_state=args.seed)
        km_dsa = km_dsa.fit(z_sample[:, 0, :])
        yy = km_dsa.predict(z_sample[:, 0, :])
    acc = utils.cluster_acc(y_train[:, 2], yy)
    nmi = normalized_mutual_info_score(y_train[:, 2], yy)
    ari = adjusted_rand_score(y_train[:, 2], yy)
    ci = cindex(t=y_train[:, 0], d=y_train[:, 1], scores_pred=risk_scores)
    t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
    rae_nc = RAE(t_pred=t_pred_med[y_train[:, 1] == 1], t_true=y_train[y_train[:, 1] == 1, 0],
                 cens_t=1 - y_train[y_train[:, 1] == 1, 1])
    rae_c = RAE(t_pred=t_pred_med[y_train[:, 1] == 0], t_true=y_train[y_train[:, 1] == 0, 0],
                cens_t=1 - y_train[y_train[:, 1] == 0, 1])
    if args.results_fname is '':
        file_results = "results_" + args.data + ".txt"
    else:
        file_results = args.results_fname + ".txt"
    f = open(file_results, "a+")
    f.write(
        "Epochs= %d, batch_size= %d, latent_dim= %d, K= %d, mc samples= %d, weibull_shape= %d, learning_rate= %f, pretrain_e= %d, decay= %f, name= %s, survival= %s, "
        "sample_surv= %s, seed= %d.\n"
        % (args.num_epochs, args.batch_size, configs['training']['latent_dim'], configs['training']['num_clusters'],
           configs['training']['monte_carlo'],
           configs['training']['weibull_shape'], args.lr, args.epochs_pretrain, args.decay, ex_name, args.survival,
           configs['training']['sample_surv'], args.seed))

    if args.pretrain:
        f.write("epochs_pretrain: %d. Pretrain accuracy: %f , " % (args.epochs_pretrain, pretrain_acc))

    f.write("Train (w t)    |   Accuracy: %.3f, NMI: %.3f, ARI: %.3f. CI: %.3f, RAE (nc.): %.3f, RAE (c.): %.3f.\n" % (
        acc, nmi, ari, ci, rae_nc, rae_c))

    plot_bigroup_kaplan_meier(t=y_train[:, 0], d=y_train[:, 1], c=y_train[:, 2], c_=yy, dir='./',
                              postfix=args.data + '_' + str(args.seed))
    plot_tsne_by_cluster(X=z_sample[:, 0], c=y_train[:, 2], font_size=12, seed=42, dir='./',
                         postfix=args.data + '_' + str(args.seed) + '_z_wt')
    plot_tsne_by_survival(X=z_sample[:, 0], t=y_train[:, 0], d=y_train[:, 1], seed=42, dir='./',
                          postfix=args.data + '_' + str(args.seed) + '_z_wt', plot_censored=True)

    if args.data != 'nsclc' and args.data != 'lung1' and args.data != 'basel':
        plot_tsne_by_cluster(X=x_train, c=yy, font_size=12, seed=42, dir='./',
                             postfix=args.data + '_' + str(args.seed) + '_x_wt')

        plot_tsne_by_cluster(X=x_train, c=y_train[:, 2], font_size=12, seed=42, dir='./',
                             postfix=args.data + '_' + str(args.seed) + '_x_true_labels')

    # Some extra logging
    if args.data == 'nsclc':
        np.savetxt(fname="c_hat_nsclc_" + str(args.seed) + ".csv", X=yy)
        plot_group_kaplan_meier(t=y_train[y_train[:, 0] > 0.001, 0], d=y_train[y_train[:, 0] > 0.001, 1],
                                c=yy[y_train[:, 0] > 0.001], dir='', experiment_name='nsclc_' + str(args.seed))
    elif args.data == 'lung1':
        np.savetxt(fname="c_hat_lung1_" + str(args.seed) + ".csv", X=yy)
        plot_group_kaplan_meier(t=y_train[:, 0], d=y_train[:, 1], c=yy, dir='',
                                experiment_name='lung1_' + str(args.seed))

    elif args.data == 'basel':
        np.savetxt(fname="c_hat_basel_" + str(args.seed) + ".csv", X=yy)
        plot_group_kaplan_meier(t=y_train[:, 0], d=y_train[:, 1], c=yy, dir='',
                                experiment_name='basel_' + str(args.seed))

    # Test set performance
    tf.keras.backend.set_value(model.use_t, np.array([0.0]))
    rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((x_train, y_train), batch_size=args.batch_size)
    risk_scores = np.squeeze(risk_scores)
    yy = np.argmax(p_c_z, axis=-1)
    if args.dsa:
        yy = km_dsa.predict(z_sample[:, 0, :])
    acc = utils.cluster_acc(y_train[:, 2], yy)
    nmi = normalized_mutual_info_score(y_train[:, 2], yy)
    ari = adjusted_rand_score(y_train[:, 2], yy)
    ci = cindex(t=y_train[:, 0], d=y_train[:, 1], scores_pred=risk_scores)
    t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
    rae_nc = RAE(t_pred=t_pred_med[y_train[:, 1] == 1], t_true=y_train[y_train[:, 1] == 1, 0],
                 cens_t=1 - y_train[y_train[:, 1] == 1, 1])
    rae_c = RAE(t_pred=t_pred_med[y_train[:, 1] == 0], t_true=y_train[y_train[:, 1] == 0, 0],
                cens_t=1 - y_train[y_train[:, 1] == 0, 1])
    f.write("Train (w/o t)  |   Accuracy: %.3f, NMI: %.3f, ARI: %.3f. CI: %.3f, RAE (nc.): %.3f, RAE (c.): %.3f.\n" % (
        acc, nmi, ari, ci, rae_nc, rae_c))

    plot_tsne_by_cluster(X=z_sample[:, 0], c=y_train[:, 2], font_size=12, seed=42, dir='./',
                         postfix=args.data + '_' + str(args.seed) + '_z_wot')
    plot_tsne_by_survival(X=z_sample[:, 0], t=y_train[:, 0], d=y_train[:, 1], seed=42, dir='./',
                          postfix=args.data + '_' + str(args.seed) + '_z_wot', plot_censored=True)

    if args.data != 'nsclc' and args.data != 'lung1' and args.data != 'basel':
        plot_tsne_by_cluster(X=x_train, c=yy, font_size=12, seed=42, dir='./',
                             postfix=args.data + '_' + str(args.seed) + '_x_wot')

    # Test set performance
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))
    rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((x_test, y_test), batch_size=args.batch_size)
    risk_scores = np.squeeze(risk_scores)
    if args.save_model:
        with open(experiment_path / 'c_test.npy', 'wb') as save_file:
            np.save(save_file, p_c_z)
    yy = np.argmax(p_c_z, axis=-1)
    if args.dsa:
        yy = km_dsa.predict(z_sample[:, 0, :])
    acc = utils.cluster_acc(y_test[:, 2], yy)
    nmi = normalized_mutual_info_score(y_test[:, 2], yy)
    ari = adjusted_rand_score(y_test[:, 2], yy)
    ci = cindex(t=y_test[:, 0], d=y_test[:, 1], scores_pred=risk_scores)
    t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
    rae_nc = RAE(t_pred=t_pred_med[y_test[:, 1] == 1], t_true=y_test[y_test[:, 1] == 1, 0],
                 cens_t=1 - y_test[y_test[:, 1] == 1, 1])
    rae_c = RAE(t_pred=t_pred_med[y_test[:, 1] == 0], t_true=y_test[y_test[:, 1] == 0, 0],
                cens_t=1 - y_test[y_test[:, 1] == 0, 1])

    if args.data == 'nsclc':
        np.savetxt(fname="c_hat_test_nsclc_" + str(args.seed) + ".csv", X=yy)
    if args.data == 'basel':
        np.savetxt(fname="c_hat_test_basel_" + str(args.seed) + ".csv", X=yy)

    f.write("Test (w t)     |   Accuracy: %.3f, NMI: %.3f, ARI: %.3f. CI: %.3f, RAE (nc.): %.3f, RAE (c.): %.3f.\n" % (
        acc, nmi, ari, ci, rae_nc, rae_c))

    # Plot generated samples..
    if args.data == 'lung1' or args.data == 'nsclc' or args.data == 'basel':
        utils.save_generated_samples(model=model, inp_size=[64, 64], grid_size=10, cmap='bone',
                                     postfix='nsclc_' + str(args.seed) + '_K_' + str(model.num_clusters))

    tf.keras.backend.set_value(model.use_t, np.array([0.0]))
    rec, z_sample, p_z_c, p_c_z, risk_scores, lambdas = model.predict((x_test, y_test), batch_size=args.batch_size)
    risk_scores = np.squeeze(risk_scores)
    yy = np.argmax(p_c_z, axis=-1)
    if args.dsa:
        yy = km_dsa.predict(z_sample[:, 0, :])
    acc = utils.cluster_acc(y_test[:, 2], yy)
    nmi = normalized_mutual_info_score(y_test[:, 2], yy)
    ari = adjusted_rand_score(y_test[:, 2], yy)
    ci = cindex(t=y_test[:, 0], d=y_test[:, 1], scores_pred=risk_scores)
    t_pred_med = risk_scores * np.log(2) ** (1 / model.weibull_shape)
    rae_nc = RAE(t_pred=t_pred_med[y_test[:, 1] == 1], t_true=y_test[y_test[:, 1] == 1, 0],
                 cens_t=1 - y_test[y_test[:, 1] == 1, 1])
    rae_c = RAE(t_pred=t_pred_med[y_test[:, 1] == 0], t_true=y_test[y_test[:, 1] == 0, 0],
                cens_t=1 - y_test[y_test[:, 1] == 0, 1])

    # NOTE: this can be slow, comment it out unless really necessary!
    if args.eval_cal:
        t_sample = utils.sample_weibull(scales=risk_scores, shape=model.weibull_shape)
        cal = calibration(predicted_samples=t_sample, t=y_test[:, 0], d=y_test[:, 1])
    else:
        cal = np.nan

    f.write(
        "Test (w/o t)   |   Accuracy: %.3f, NMI: %.3f, ARI: %.3f. CI: %.3f, RAE (nc.): %.3f, RAE (c.): %.3f, CAL: %.3f.\n" % (
            acc, nmi, ari, ci, rae_nc, rae_c, cal))
    tf.keras.backend.set_value(model.use_t, np.array([1.0]))

    if args.data == 'lung1':
        np.savetxt(fname="preds_lung1_" + str(args.seed) + ".csv",
                   X=np.stack((t_pred_med, y_test[:, 0], y_test[:, 1]), axis=1))
    elif args.data == 'nsclc':
        np.savetxt(fname="preds_nsclc_" + str(args.seed) + ".csv",
                   X=np.stack((t_pred_med, y_test[:, 0], y_test[:, 1]), axis=1))
    elif args.data == 'basel':
        np.savetxt(fname="preds_basel_" + str(args.seed) + ".csv",
                   X=np.stack((t_pred_med, y_test[:, 0], y_test[:, 1]), axis=1))

    f.close()
    print(str(acc))
    print(str(nmi))
    print(str(ari))
    print(str(ci))
    print("(" + str(rae_nc) + "; " + str(rae_c) + ")")
