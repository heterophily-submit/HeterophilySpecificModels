# FSGNN

This is a copy of implementation of FSGNN. 
For more details, please refer to [the paper](https://arxiv.org/abs/2105.07634).
This work is further extended into our [second](https://arxiv.org/abs/2111.06748) paper.
Experiments were conducted with following setup:  
Pytorch: 1.6.0  
Python: 3.8.5  
Cuda: 10.2.89
Trained on NVIDIA V100 GPU.

**Results from the original paper**

| **Dataset** | **3-hop Accuracy(%)** | **8-hop Accuracy(%)** | **16-hop Accuracy(%)** | **32-hop Accuracy(%)** |
| :-------------- | :---------------------: | :---------------------: | :--------------------: | :--------------------: |
| Cora            | 87\.73                  |       **87\.93**        | 87\.91                 | 87\.83                 |
| Citeseer        | 77\.19                  | 77\.40                  | **77\.46**             | **77\.46**             |
| Pubmed          | 89\.73                  |       **89\.75**        | 89\.60                 | 89\.63                 |
| Chameleon       | 78\.14                  | 78\.27                  | 78\.36                 | **78\.53**             |
| Wisconsin       |       **88\.43**        | 87\.84                  | 88\.04                 | 88\.24                 |
| Texas           |       **87\.30**   |       **87\.30**        | 86\.76                 | 86\.76                 |
| Cornell         | 87\.03                  | 87\.84                  | 86\.76                 | **88\.11**             |
| Squirrel        | 73\.48                  | 74\.10                  | 73\.95                 | **74\.15**             |
| Actor           | 35\.67                  |       **35\.75**        | 35\.25                 | 35\.22                 |
| Actor(no-norm)  | 37\.63                  | **37\.75**              | 37\.67                 | 37\.62                 |

In addition, we include model accuracy of Actor dataset without using hop-normalization, as model shows higher accuracy in this setting.

**Results with considering homophily/heterophily assumption in datasets**

| **Dataset** | **3-hop Accuracy(%)** | **8-hop Accuracy(%)** |
| :---------- | :-------------------: | :-------------------: |
| Cora        | 87\.61                | 88\.23                |
| Citeseer    | 77\.17                | 77\.35                |
| Pubmed      | 89\.70                | 89\.78                |
| Chameleon   | 78\.93                | 78\.95                |
| Wisconsin   | 88\.24                | 87\.65                |
| Texas       | 87\.57                | 87\.57                |
| Cornell     | 87\.30                | 87\.30                |
| Squirrel    | 73\.86                | 73\.94                |
| Actor       | 35\.38                | 35\.62                |

# Usage

### Data preparation

Due to memory limitations refer to the [original repository](https://github.com/sunilkmaurya/FSGNN) and download 
`data`, `new_data` from there. 

Data from pytorch-geometric will be downloaded to `./data` directory
or can be copied there via symbolic link

For new datasets make symbolic link to the root in a following way:

```
ln -s <DATA_DIR> new_data
```

### Experiments

In order to launch experiments run `./run_train_fsgnn.sh` in the root directory.

To setup the experiment edit `run_train_fsgnn.sh`.

