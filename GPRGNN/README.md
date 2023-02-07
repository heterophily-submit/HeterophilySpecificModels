# GPRGNN

This is the modified version of the source code for our ICLR2021 paper: [Adaptive Universal Generalized PageRank Graph Neural Network](https://openreview.net/forum?id=n6jl7fLxrP). (See also the [ArXiv version](https://arxiv.org/pdf/2006.07988.pdf) for the latest update on typos) 


<p align="center">
  <img src="https://github.com/jianhao2016/GPRGNN/blob/master/figs/workflow.png" width="600">
</p>

Hidden state feature extraction is performed by a neural networks using individual node features propagated via GPR. Note that both the GPR weights <img src="https://render.githubusercontent.com/render/math?math=\gamma_k"> and parameter set <img src="https://render.githubusercontent.com/render/math?math=\{\theta\}"> of the neural network are learned simultaneously in an end-to-end fashion (as indicated in red).


The learnt GPR weights of the GPR-GNN on real world datasets. Cora is homophilic while Texas is heterophilic (Here, H stands for the level of homophily defined in the main text, Equation (1)). An interesting trend may be observed: For the heterophilic case the weights alternate from positive to negative with dampening amplitudes. The shaded region corresponds to a 95% confidence interval.


<p align="center">
  <img src="https://github.com/jianhao2016/GPRGNN/blob/master/figs/Different_gamma_upated_H.png" width="600">
</p>

# Requirements:
```
pytorch
pytorch-geometric
numpy
```

# Usage

### Data preparation

Create a symbolic or copy to the network directory, i.e `.../GPRGNN/data`

```
ln -s <DATA_DIR> data
```

## Experiment running

In order to launch experiments run `./Reproduce_GPRGNN.sh` in the `src` directory.

To setup the experiment edit `Reproduce_GPRGNN.sh`.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.


# Citation
Please cite the paper if you use this code in your own work:
```latex
@inproceedings{
chien2021adaptive,
title={Adaptive Universal Generalized PageRank Graph Neural Network},
author={Eli Chien and Jianhao Peng and Pan Li and Olgica Milenkovic},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=n6jl7fLxrP}
}
```

Feel free to email us(jianhao2@illinois.edu, ichien3@illinois.edu) if you have any further questions. 



