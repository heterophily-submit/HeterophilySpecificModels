# GloGNN

This repository is based on the official repository of ICML 2022 paper *Finding Global Homophily in Graph Neural Networks When Meeting Heterophily*.

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.10.0

- torch-geometric==2.0.2

- networkx==2.3

- scipy==1.5.4

- numpy==1.19.2

- sklearn==0.0

- matplotlib==3.1.1

- pandas==1.1.5

You can simply run 

```python
pip install -r requirements.txt
```

## Repository structure
We select some important files for detailed description.

```python
|-- small-scale # experiments for 9 small-scale datasets
    |-- data/ # 3 old datasets, including cora, citeseer, and pubmed
    |-- new-data/ # 6 new datasets, including texas, wisconsin, cornell, actor, squirrel, and chameleon
    |-- splits/ # splits for 6 new datasets
    |-- sh/ # all run shs
    |-- main.py  # the main code
    |-- main_z.py  # obtains coefficient matrix z
    |-- main_h.py # obtains final layer embedding h
```

## Run code

### Data preparation

Create a symbolic or copy to the `small-scale` directory, i.e `.../glognn/small-scale/data`

```
ln -s <DATA_DIR> data
```

### Experiments

In order to run experiments edit and launch `run_glognn.sh`. 

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.

## Attribution

Parts of this code are based on the following repositories:

- [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale)

- [PYGCN](https://github.com/tkipf/pygcn)

- [WRGAT](https://github.com/susheels/gnns-and-local-assortativity/tree/main/struc_sim)


## Citation

If you find this code working for you, please cite:

```python
@article{li2022finding,
  title={Finding Global Homophily in Graph Neural Networks When Meeting Heterophily},
  author={Li, Xiang and Zhu, Renyu and Cheng, Yao and Shan, Caihua and Luo, Siqiang and Li, Dongsheng and Qian, Weining},
  journal={arXiv preprint arXiv:2205.07308},
  year={2022}
}
```
