# How Powerful are Spectral Graph Neural Networks

This repository is based on the official implementation of the model in the [following paper](https://arxiv.org/abs/2205.11172v1):

Xiyuan Wang, Muhan Zhang: How Powerful are Spectral Graph Neural Networks. ICML 2022

```{bibtex}
@article{JacobiConv,
  author    = {Xiyuan Wang and
               Muhan Zhang},
  title     = {How Powerful are Spectral Graph Neural Networks},
  journal   = {ICML},
  year      = {2022}
}
```

#### Requirements
Tested combination: Python 3.9.6 + [PyTorch 1.9.0](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 2.0.3](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) + [PyTorch Sparse 0.6.12](https://github.com/rusty1s/pytorch_sparse)

Other required python libraries include: numpy, scikit-learn, optuna, seaborn etc.

Tested on

```bash
optuna                3.0.3
numpy                 1.19.5
scikit-learn          0.24.2
seaborn               0.12.0
```

### Usage

#### Data preparation

Create a symbolic or copy to the network directory, i.e `.../JacobiConv/data`

```
ln -s <DATA_DIR> data
```

#### Experiment running

In order to launch experiments run `./run_real_world.sh` in the root directory.

To setup the experiment edit `run_real_world.sh`.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.

