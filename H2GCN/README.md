# Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs

Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. 2020. *Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs*. Advances in Neural Information Processing Systems 33 (2020).

[[Paper]](https://arxiv.org/abs/2006.11468)
[[Poster]](https://www.jiongzhu.net/assets/files/F20-Jiong-H2GCN-NeurIPS-Poster.pdf)
[[Slides]](https://www.jiongzhu.net/assets/files/F20-Jiong-H2GCN-NeurIPS-Talk.pdf)

## Updates

- Oct. 2021: We additionally provide our synthetic datasets `syn-cora` and `syn-products` in a more straight-forward `npz` format; see README in folder `/npz-datasets` for more details.

- Aug. 2021: In a [blog post](https://www.jiongzhu.net/revisiting-heterophily-GNNs/), we revisit the problem of heterophily for GNNs and discuss the reasons behind seemly different takeaways in light of recent works in this area.

## Requirements

This repository is based on [H2GCN](https://github.com/GemsLab/H2GCN) original repository.

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

### For `H2GCN`

- **TensorFlow** >= 2.0 (tested on 2.2)

Note that it is possible to use `H2GCN` without `signac` and `scikit-learn` on your own data and experimental framework.

### For baselines

We also include the code for the baseline methods in the repository. These code are mostly the same as the reference implementations provided by the authors, *with our modifications* to add JK-connections, interoperability with our experimental pipeline, etc. For the requirements to run these baselines, please refer to the instructions provided by the original authors of the corresponding code, which could be found in each folder under `/baselines`.

As a general note, TensorFlow 1.15 can be used for all code requiring TensorFlow 1.x; for PyTorch, it is usually fine to use PyTorch 1.6; all code should be able to run under Python >= 3.7. In addition, the [basic requirements](#basic-requirements) must also be met.

## Usage

#### Data preparation

Create a symbolic or copy to the network directory, i.e `.../CPGNN/cpgnn/data`

```
ln -s <DATA_DIR> data
```

#### Experiment running

In order to launch experiments run `./run_experiments.sh` in the `/cpgnn` directory.

To setup the experiment edit `run_experiments.sh`.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.

## Contact

Please contact Jiong Zhu (jiongzhu@umich.edu) in case you have any questions.

## Citation

Please cite the paper if you make use of this code in your own work:

```bibtex
@article{zhu2020beyond,
  title={Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs},
  author={Zhu, Jiong and Yan, Yujun and Zhao, Lingxiao and Heimann, Mark and Akoglu, Leman and Koutra, Danai},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
