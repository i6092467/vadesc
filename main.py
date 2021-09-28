import argparse
from pathlib import Path
import yaml
import logging
import tensorflow as tf
import tensorflow_probability as tfp
import os
from models.losses import Losses

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers
tfk = tf.keras

from train import run_experiment

# project-wide constants:
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

    # parameters of the model
    parser.add_argument('--data',
                        default='mnist',
                        type=str,
                        choices=['mnist', 'sim', 'support', 'flchain', 'hgg', 'hemo', 'lung1', 'nsclc',
                                 'nsclc_features', 'basel'],
                        help='specify the data (mnist, sim, support, flchain, hgg, hemo, lung1, nsclc, basel)')
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
    parser.add_argument('--dsa',
                        dest='dsa',
                        action='store_true',
                        help='')
    parser.add_argument('--dsa_k',
                        default=1,
                        type=int,
                        help='')
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
                        default=os.path.join(project_dir, 'models/experiments'),
                        type=lambda p: Path(p).absolute(),
                        help='specify the folder where the results get saved')
    parser.add_argument('--results_fname',
                        default='',
                        type=str,
                        help='specify the name of the .txt file with the final results')
    parser.add_argument('--pretrain', default=False, type=bool,
                        help='True to pretrain the autoencoder.')
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
    parser.add_argument('--eager',
                        default=False,
                        type=bool,
                        help='specify if the TF functions should be run eagerly')
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

    if args.dsa:
        configs['training']['num_clusters'] = 1
        print('\n' * 2)
        print("RUNNING DSA WITH K-MEANS POST HOC")
        print(configs)
        print('\n' * 2)

    run_experiment(args, configs, loss)


if __name__ == "__main__":
    main()
