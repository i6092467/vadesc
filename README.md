# Variational Deep Survival Clustering

This repository holds the code for clustering survival data using the **<u>va</u>riational <u>de</u>ep <u>s</u>urvival <u>c</u>lustering (VaDeSC)** model, a novel probabilistic approach to cluster survival data in a variational deep clustering setting. This method employs a deep generative model to uncover the underlying distribution of both the explanatory variables and the potentially censored survival times.

### Requirements

All required libraries are included in the conda environment specified by `environment.yml`. To install and activate it, follow the instructions below:

```
conda env create -f environment.yml         # install dependencies
conda activate Survival_Cluster_Analysis    # activate environment
```

### Usage

File `main.py` trains and evaluates the VaDeSC model. It accepts following arguments:

```
--data {mnist,sim,support,flchain,hgg,hemo,lung1,nsclc,nsclc_features,basel}
                        the dataset
  --num_epochs NUM_EPOCHS
                        the number of training epochs
  --batch_size BATCH_SIZE
                        the mini-batch size
  --lr LR               the learning rate
  --decay DECAY         the decay
  --decay_rate DECAY_RATE
                        the decay rate for the learning rate schedule
  --epochs_lr EPOCHS_LR
                        the number of epochs before dropping down the learning rate
  --lrs LRS             specifies if the learning rate schedule is to be used
  --weibull_shape WEIBULL_SHAPE
                        the Weibull shape parameter (global)
  --no-survival         specifies if the survival model should not be included
  --dsa                 specifies if the deep survival analysis with k-means should be run
  --dsa_k DSA_K         number of clusters in deep survival analysis with k-means
  --eval-cal EVAL_CAL   specifies if the calibration needs to be evaluated
  --runs RUNS           the number of runs, the results will be averaged
  --results_dir RESULTS_DIR
                        the directory where the results will be saved
  --results_fname RESULTS_FNAME
                        the name of the .txt file with the results
  --pretrain PRETRAIN   specifies if the autoencoder should be pretrained
  --epochs_pretrain EPOCHS_PRETRAIN
                        the number of pretraining epochs
  --save_model SAVE_MODEL
                        specifies if the model should be saved
  --ex_name EX_NAME     the experiment name
  --config_override CONFIG_OVERRIDE
                        the override file name for config.yml
  --seed SEED           random number generator seed
  --eager EAGER         specifies if the TF functions should be run eagerly
```
Folder `/configs` contains `.yml` files which further sepcify the configuration of the model for each experiment. Folder `/bin` contains shell scripts for running clustering and time-to-event prediction experiments:
- **Synthetic**: `run_vadesc_sim`
- **survMNIST**: `run_vadesc_mnist`
- **SUPPORT**: `run_vadesc_support`
- **FLChain**: `run_vadesc_flchain`
- **HGG**: `run_vadesc_hgg`
- **Hemodialysis**: `run_vadesc_hemo`
- **NSCLC**: `run_vadesc_nsclc`

The VaDeSC model is implemented in `/models/model.py`. Encoder and decoder architectures are specified in `/models/networks.py`. Data loaders are provided in the `/datasets` folder. HGG, Hemodialysis, and NSCLC datasets are not included in the repository due to the medical confidentiality. Code for *post hoc* explanations of cluster assignments is available in the `/post_hoc_explanations` folder.

### Acknowledgements

- survMNIST code is based on [Sebastian Pölsterl's tutorial](https://github.com/sebp/survival-cnn-estimator)
- SUPPORT and FLChain datasets and utility functions for data preprocessing were taken from Chapfuwa *et al.*'s [SCA repository](https://github.com/paidamoyo/survival_cluster_analysis) and Nagpal *et al.*'s [DCM repository](https://github.com/chiragnagpal/deep_cox_mixtures)
