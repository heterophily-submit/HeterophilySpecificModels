# Graph Neural Networks with Heterophily

Jiong Zhu, Ryan Rossi, Anup Rao, Tung Mai, Nedim Lipka, Nesreen K Ahmed, and Danai Koutra. 2021. *Graph Neural Networks with Heterophily*. To Appear In *Proceedings of the AAAI Conference on Artificial Intelligence*.

[[Paper]](https://arxiv.org/abs/2009.13566)

## Requirements

This repository is based on [CPGNN](https://github.com/GemsLab/CPGNN) original repository.

### Basic Requirements

- **Python** >= 3.7 (tested on 3.8)
- **signac**: this package utilizes [signac](https://signac.io) to manage experiment data and jobs. signac can be installed with the following command:

  ```bash
  pip install signac==1.1 signac-flow==0.7.1 signac-dashboard
  ```

  Note that the latest version of signac may cause incompatibility.
- **numpy** (tested on 1.18.5)
- **scipy** (tested on 1.5.0)
- **networkx** >= 2.4 (tested on 2.4)
- **scikit-learn** (tested on 0.23.2)

### For `CPGNN`

- **TensorFlow** >= 2.0 (tested on 2.2)

Note that it is possible to use `CPGNN` without `signac` and `scikit-learn` on your own data and experimental framework.

### For baselines

We also include the code for the baseline methods in the repository. These code are mostly the same as the reference implementations provided by the authors, *with our modifications* to add interoperability with our experimental pipeline. For the requirements to run these baselines, please refer to the instructions provided by the original authors of the corresponding code, which could be found in each folder under `/baselines`.

As a general note, TensorFlow 1.15 can be used for all code requiring TensorFlow 1.x; for PyTorch, it is usually fine to use PyTorch 1.6; all code should be able to run under Python >= 3.7. In addition, the [basic requirements](#basic-requirements) must also be met.

## Usage

#### Data preparation

Create a symbolic or copy to the network directory, i.e `.../H2GCN/h2gcn/data`

```
ln -s <DATA_DIR> data
```

#### Experiment running

In order to launch experiments run `./run_experiments.sh` in the `/h2gcn` directory.

To setup the experiment edit `run_experiments.sh`.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.

## Contact

Please contact Jiong Zhu (jiongzhu@umich.edu) in case you have any questions.

## Citation

Please cite the original paper if you make use of this code in your own work:

```bibtex
@inproceedings{zhu2021graph,
  title={Graph Neural Networks with Heterophily},
  author={Zhu, Jiong and Rossi, Ryan A and Rao, Anup and Mai, Tung and Lipka, Nedim and Ahmed, Nesreen K and Koutra, Danai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={12},
  pages={11168--11176},
  year={2021}
}
```
