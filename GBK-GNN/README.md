# GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily

## Abstract

Graph Neural Networks (GNNs) are widely used on a variety of graph-based machine learning tasks. For node-level tasks, GNNs have strong power to model the homophily property of graphs (i.e., connected nodes are more similar) while their ability to capture heterophily property is often doubtful. This is partially caused by the design of the feature transformation with the same kernel for the nodes in the same hop and the followed aggregation operator. One kernel cannot model the similarity and the dissimilarity (i.e., the positive and negative correlation) between node features simultaneously even though we use attention mechanisms like Graph Attention Network (GAT), since the weight calculated by attention is always a positive value. In this paper, we propose a novel GNN model based on a bi-kernel feature transformation and a selection gate. Two kernels capture homophily and heterophily information respectively, and the gate is introduced to select which kernel we should use for the given node pairs. We conduct extensive experiments on various datasets with different homophily-heterophily properties. The experimental results show consistent and significant improvements against state-of-the-art GNN methods.

## Environmental Preparation

### Requirements

- pytorch 1.7.0
- cuda 10.2
- torch-geometric 1.7.2
- torch-scatter 2.0.5
- torch-sparse 0.6.8

### Install Scripts

Before install torch-geometric, make sure you have appropriate CUDA and pytorch versions. You can use the following scripts to install the dependencies.

You can directly run `install_requirements.sh` file to install environment.
Or install manually as follow:

#### Step 1: Ensure that at least PyTorch 1.4.0 is installed:

```bash
python -c "import torch; print(torch.__version__)"
>>> 1.7.0
```

#### Step 2: Find the CUDA version PyTorch was installed with:

```bash
python -c "import torch; print(torch.version.cuda)"
>>> 10.2
```

#### Step 3: Install the relevant packages:

```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

where `${CUDA}` and `${TORCH}` should be replaced by the specific CUDA version (`cpu`, `cu92`, `cu101`, `cu102`, `cu110`, `cu111`) and PyTorch version (`1.4.0`, `1.5.0`, `1.6.0`, `1.7.0`, `1.7.1`, `1.8.0`, `1.8.1`, `1.9.0`), respectively.

## Run code

### Data preparatation

Data from pytorch-geometric will be downloaded to `./data` directory
or can be copied there via symbolic link

For new datasets make symbolic link to the root in a following way:

```
ln -s <DATA_DIR> new_data
```

The method is implemented on GCN, GraphSage, GAT and GAT2.

In order to run experiments edit and launch `run_train.sh`. 

Do not set `--aug` flag.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.
