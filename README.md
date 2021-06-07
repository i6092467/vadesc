# Variational Deep Survival Clustering

This repository holds the code for clustering survival data using the **<u>va</u>riational <u>de</u>ep <u>s</u>urvival <u>c</u>lustering (VaDeSC)** model, a novel probabilistic approach to cluster survival data in a variational deep clustering setting. This method employs a deep generative model to uncover the underlying distribution of both the explanatory variables and the potentially censored survival times.

*Schematic summary of VaDeSC*:
<p align="center">
  <img align="middle" src="https://github.com/i6092467/vadesc/blob/main/figures/survival_clustering_schematic.png" alt="schematic summary of VaDeSC" width="400"/>
</p>

### Requirements

All required libraries are included in the conda environment specified by `environment.yml`. To install and activate it, follow the instructions below:

```
conda env create -f environment.yml  # install dependencies
conda activate Survival_Cluster_Analysis  # activate environment
```

### Usage

Folder `/bin` contains shell scripts for running clustering and time-to-event prediction experiments:
- **Synthetic**: `run_vadesc_sim`
- **survMNIST**: `run_vadesc_mnist`
- **SUPPORT**: `run_vadesc_support`
- **FLChain**: `run_vadesc_flchain`
- **HGG**: `run_vadesc_hgg`
- **Hemodialysis**: `run_vadesc_hemo`

The VaDeSC model is implemented in `/models/Ours/model.py`. Data loaders are provided in the `/datasets` folder. HGG and Hemodialysis datasets are not included in the repository due to the medical confidentiality. Code for *post hoc* explanations of cluster assignments is available in the `/post_hoc_explanations` folder.

### Acknowledgements

- survMNIST code is based on Sebastian Pölsterl's tutorial: [https://github.com/sebp/survival-cnn-estimator](https://github.com/sebp/survival-cnn-estimator)
- SUPPORT and FLChain datasets and utility functions for data preprocessing were taken from Paidamoyo Chapfuwa's SCA repository: [https://github.com/paidamoyo/survival_cluster_analysis](https://github.com/paidamoyo/survival_cluster_analysis)

### Maintainers
- Laura Manduchi ([laura.manduchi@inf.ethz.ch](mailto:laura.manduchi@inf.ethz.ch))
- Ričards Marcinkevičs ([ricards.marcinkevics@inf.ethz.ch](mailto:ricards.marcinkevics@inf.ethz.ch))
- Michela C. Massi
