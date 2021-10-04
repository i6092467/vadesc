"""
Runs the VaDeSC model.
"""
import argparse
from pathlib import Path
import yaml
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import os
from models.losses import Losses

from train import run_experiment

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

# Project-wide constants:
ROOT_LOGGER_STR = "GMM_Survival"
LOGGER_RESULT_FILE = "logs.txt"
CHECKPOINT_PATH = 'models/Ours'

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main():
    project_dir = Path(__file__).absolute().parent
    print(project_dir)

    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--data',
                        default='mnist',
                        type=str,
                        choices=['mnist', 'sim', 'support', 'flchain', 'hgg', 'hemo', 'lung1', 'nsclc',
                                 'nsclc_features', 'basel'],
                        help='the dataset (mnist, sim, support, flchain, hgg, hemo, lung1, nsclc, basel)')
    parser.add_argument('--num_epochs',
                        default=1000,
                        type=int,
                        help='the number of training epochs')
    parser.add_argument('--batch_size',
                        default=256,
                        type=int,
                        help='the mini-batch size')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='the learning rate')
    parser.add_argument('--decay',
                        default=0.00001,
                        type=float,
                        help='the decay')
    parser.add_argument('--decay_rate',
                        default=0.9,
                        type=float,
                        help='the decay rate for the learning rate schedule')
    parser.add_argument('--epochs_lr',
                        default=10,
                        type=int,
                        help='the number of epochs before dropping down the learning rate')
    parser.add_argument('--lrs',
                        default=False,
                        type=bool,
                        help='specifies if the learning rate schedule is to be used')
    parser.add_argument('--weibull_shape',
                        default=1.0,
                        type=float,
                        help='the Weibull shape parameter (global)')
    parser.add_argument('--no-survival',
                        dest='survival',
                        action='store_false',
                        help='specifies if the survival model should not be included')
    parser.add_argument('--dsa',
                        dest='dsa',
                        action='store_true',
                        help='specifies if the deep survival analysis with k-means shuld be run')
    parser.add_argument('--dsa_k',
                        default=1,
                        type=int,
                        help='number of clusters in deep survival analysis with k-means')
    parser.add_argument('--eval-cal',
                        default=False,
                        type=bool,
                        help='specifies if the calibration needs to be evaluated')
    parser.set_defaults(survival=True)

    # Other parameters
    parser.add_argument('--runs',
                        default=1,
                        type=int,
                        help='the number of runs, the results will be averaged')
    parser.add_argument('--results_dir',
                        default=os.path.join(project_dir, 'models/experiments'),
                        type=lambda p: Path(p).absolute(),
                        help='the directory where the results will be saved')
    parser.add_argument('--results_fname',
                        default='',
                        type=str,
                        help='the name of the .txt file with the results')
    parser.add_argument('--pretrain', default=False, type=bool,
                        help='specifies if the autoencoder should be pretrained')
    parser.add_argument('--epochs_pretrain', default=10, type=int,
                        help='the number of pretraining epochs')
    parser.add_argument('--save_model', default=False, type=bool,
                        help='specifies if the model should be saved')
    parser.add_argument('--ex_name', default='', type=str, help='the experiment name')
    parser.add_argument('--config_override', default='', type=str, help='the override file name for config.yml')
    parser.add_argument('--seed', default=42, type=int, help='random number generator seed')
    parser.add_argument('--eager',
                        default=False,
                        type=bool,
                        help='specifies if the TF functions should be run eagerly')
    args = parser.parse_args()

    data_name = args.data +'.yml'
    config_path = project_dir / 'configs' / data_name

    # Check for config override
    if args.config_override is not "":
        config_path = Path(args.config_override)

    with config_path.open(mode='r') as yamlfile:
        configs = yaml.safe_load(yamlfile)

    losses = Losses(configs)
    if args.data == "MNIST":
        loss = losses.loss_reconstruction_binary
    else:
        loss = losses.loss_reconstruction_mse

    run_experiment(args, configs, loss)


if __name__ == "__main__":
    main()
