# GBK-GNN: Gated Bi-Kernel Graph Neural Networks for Modeling Both Homophily and Heterophily

## Network description (from the original repo)

Graph Neural Networks (GNNs) are widely used on a variety of graph-based machine learning tasks. For node-level tasks, GNNs have strong power to model the homophily property of graphs (i.e., connected nodes are more similar) while their ability to capture heterophily property is often doubtful. This is partially caused by the design of the feature transformation with the same kernel for the nodes in the same hop and the followed aggregation operator. One kernel cannot model the similarity and the dissimilarity (i.e., the positive and negative correlation) between node features simultaneously even though we use attention mechanisms like Graph Attention Network (GAT), since the weight calculated by attention is always a positive value. In this paper, we propose a novel GNN model based on a bi-kernel feature transformation and a selection gate. Two kernels capture homophily and heterophily information respectively, and the gate is introduced to select which kernel we should use for the given node pairs. We conduct extensive experiments on various datasets with different homophily-heterophily properties. The experimental results show consistent and significant improvements against state-of-the-art GNN methods.

## Run code

### Data preparation

Create a symbolic or copy to the network directory, i.e `.../GBK-GNN/data`

```
ln -s <DATA_DIR> data
```


### Experiments

The method is implemented on GCN, GraphSage, GAT and GAT2.

In order to run experiments edit and launch `run_train.sh`. 

Do not set `--aug` flag.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.
