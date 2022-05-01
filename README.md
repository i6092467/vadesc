# A Deep Variational Approach to Clustering Survival Data

This repository holds the official code for the paper "[*A Deep Variational Approach to Clustering Survival Data*](https://openreview.net/forum?id=RQ428ZptQfU)", accepted at [ICLR 2022](https://openreview.net/group?id=ICLR.cc/2022/Conference). A short explanation of the method is provided in this [talk](https://slideslive.com/38955419), poster can be viewed [here](https://github.com/i6092467/vadesc/blob/main/documents/VaDeSC_Poster_ICLR_2022.pdf).

### 🔎 Contents

- [Motivation](#🦸‍♀motivation)  
- [Method](#method)
- [Requirements](#requirements)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [Maintainers](#maintainers)
- [References](#references)
- [Citation](#citation)

<p align="center">
<img src="https://user-images.githubusercontent.com/32577028/153908787-9961771f-8745-4dac-8e92-23076c0ecf94.png" width="1000" />  
  </p>
<p>
    <em> [left] Cluster-specific Kaplan–Meier curves and corresponding centroid CT images, computed
by averaging all samples assigned to each cluster by VaDeSC on the NSCLC data. [right] CT images generated by (i) sampling
latent representations from the Gaussian mixture learnt by VaDeSC and (ii) decoding representations using the decoder network.</em>
</p>
<img align="right" src="https://user-images.githubusercontent.com/32577028/153891639-ecc481c3-3652-47b9-aa31-3d2880d53858.png" width="300" />  

### 🦸‍♀ Motivation

In this work, we study the problem of clustering survival data — a challenging and so far under-explored task. The figure on the right schematically depicts this clustering problem: here, the overall patient population consists of three groups characterised by different associations between the covariates and survival, resulting in disparate clinical conditions. The survival distributions do not need to differ between clusters: compare groups 1 and 3. The identification of such patient subpopulations could, for example, facilitate a better understanding of a disease and a more personalised disease management. 

### 💡 Method

<img align="right" src="https://user-images.githubusercontent.com/32577028/153889328-bd437e81-551b-4ef7-af24-85d6adbc2c01.png" width="300" />

We introduce a novel semi-supervised probabilistic approach, the **<u>va</u>riational <u>de</u>ep <u>s</u>urvival <u>c</u>lustering (VaDeSC)**, to discovers groups of patients characterised by different generative mechanisms of survival outcome. 
The proposed method employs a deep generative model to uncover the underlying distribution of both the explanatory variables and censored survival times. The figure on the right provides
a schematic summary of the proposed approach: the input vector <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{x}"> is mapped to a latent representation <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{z}"> using
a VAE regularised by a Gaussian mixture prior. The survival density function is given by a mixture of Weibull
distributions with cluster-specific parameters <img src="https://render.githubusercontent.com/render/math?math=\boldsymbol{\beta}">. The parameters of the Gaussian mixture and the Weibull distributions are then optimised jointly using both the explanatory input variables and survival outcomes.

### 📝 Requirements

All required libraries are included in the conda environment specified by [`environment.yml`](https://github.com/i6092467/vadesc/blob/main/environment.yml). To install and activate it, follow the instructions below:

```
conda env create -f environment.yml         # install dependencies
conda activate Survival_Cluster_Analysis    # activate environment
```

### 🔨 Usage

File [`main.py`](https://github.com/i6092467/vadesc/blob/main/main.py) trains and evaluates the VaDeSC model. It accepts following arguments:

```
  --data {mnist,sim,support,flchain,hgg,hemo,lung1,nsclc,nsclc_features,basel}
                        the dataset
  --num_epochs NUM_EPOCHS
                        the number of training epochs
  --batch_size BATCH_SIZE
                        the mini-batch size
  --lr LR               the learning rate
  --decay DECAY         the exponential decay of the learning rate
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
Folder [`/configs`](https://github.com/i6092467/vadesc/tree/main/configs) contains `.yml` files which further sepcify the configuration of the model for each experiment. Folder [`/bin`](https://github.com/i6092467/vadesc/tree/main/bin) contains shell scripts for running clustering and time-to-event prediction experiments:
- **Synthetic**: [`run_vadesc_sim`](https://github.com/i6092467/vadesc/blob/main/bin/run_vadesc_sim)
- **survMNIST**: [`run_vadesc_mnist`](https://github.com/i6092467/vadesc/blob/main/bin/run_vadesc_mnist)
- **SUPPORT**: [`run_vadesc_support`](https://github.com/i6092467/vadesc/blob/main/bin/run_vadesc_support)
- **FLChain**: [`run_vadesc_flchain`](https://github.com/i6092467/vadesc/blob/main/bin/run_vadesc_flchain)
- **HGG**: [`run_vadesc_hgg`](https://github.com/i6092467/vadesc/blob/main/bin/run_vadesc_hgg)
- **Hemodialysis**: [`run_vadesc_hemo`](https://github.com/i6092467/vadesc/blob/main/bin/run_vadesc_hemo)
- **NSCLC**: [`run_vadesc_nsclc`](https://github.com/i6092467/vadesc/blob/main/bin/run_vadesc_nsclc)

The VaDeSC model is implemented in [`/models/model.py`](https://github.com/i6092467/vadesc/blob/main/models/model.py). Encoder and decoder architectures are specified in [`/models/networks.py`](https://github.com/i6092467/vadesc/blob/main/models/networks.py). Data loaders are provided in the [`/datasets`](https://github.com/i6092467/vadesc/tree/main/datasets) folder. HGG, Hemodialysis, and NSCLC datasets are not included in the repository due to the medical confidentiality. Code for *post hoc* explanations of cluster assignments is available in the [`/post_hoc_explanations`](https://github.com/i6092467/vadesc/tree/main/posthoc_explanations) folder.

**🆘 Feeling lost?** Follow [this Jupyter notebook](https://github.com/i6092467/vadesc/blob/main/notebooks/example.ipynb) to get started!

### 🤝 Acknowledgements

- survMNIST code is based on [Sebastian Pölsterl's tutorial](https://github.com/sebp/survival-cnn-estimator)
- SUPPORT and FLChain datasets and utility functions for data preprocessing were taken from Chapfuwa *et al.*'s [SCA repository](https://github.com/paidamoyo/survival_cluster_analysis) and Nagpal *et al.*'s [DCM repository](https://github.com/chiragnagpal/deep_cox_mixtures)

### 📭 Maintainers

- Laura Manduchi ([laura.manduchi@inf.ethz.ch](mailto:laura.manduchi@inf.ethz.ch))
- Ričards Marcinkevičs ([ricards.marcinkevics@inf.ethz.ch](mailto:ricards.marcinkevics@inf.ethz.ch))

### 📚 References

Below are some references helpful for understanding our method:
- E. Bair and R. Tibshirani. Semi-supervised methods to predict patient survival from gene expression data. *PLoS Biology*, 2(4):e108, 2004.
- N. Dilokthanakul, P. A. M. Mediano, M. Garnelo, M. C. H. Lee, H. Salimbeni, K. Arulkumaran, and M. Shanahan. Deep unsupervised clustering with Gaussian mixture variational autoencoders, 2016. *arXiv:1611.02648*.
- Z. Jiang, Y. Zheng, H. Tan, B. Tang, and H. Zhou. Variational deep embedding: An unsupervised and generative approach to clustering. In *Proceedings of the 26th International Joint Conference on Artificial Intelligence*, pp. 1965–1972. AAAI Press, 2017.
- S. Liverani, L. Leigh, I. L. Hudson, and J. E. Byles. Clustering method for censored and collinear survival data. *Computational Statistics*, 2020.
- P. Chapfuwa, C. Li, N. Mehta, L. Carin, and R. Henao. Survival cluster analysis. In *Proceedings of the ACM Conference on Health, Inference, and Learning*, pp. 60–68. Association for Computing Machinery, 2020.
- C. Nagpal, X. R. Li, and A. Dubrawski. Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks. *IEEE Journal of Biomedical and Health Informatics*, 2021.

### 🙂 Citation

```
@inproceedings{ManduchiMarcinkevics2022,
  title={A Deep Variational Approach to Clustering Survival Data},
  author={Laura Manduchi and Ri{\v{c}}ards Marcinkevi{\v{c}}s and Michela C. Massi 
          and Thomas Weikert and Alexander Sauter and Verena Gotta 
          and Timothy M{\"u}ller and Flavio Vasella and Marian C. Neidert 
          and Marc Pfister and Bram Stieltjes and Julia E Vogt},
  booktitle={International Conference on Learning Representations},
  year={2022},
}
```
