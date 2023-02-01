# FAGCN
Code of [Beyond Low-frequency Information in Graph Convolutional Networks](http://shichuan.org/doc/102.pdf)

# Usage

### Installation

Requirements are given in `requirements.txt`

### Data preparation

Create a symbolic or copy to the network directory, i.e `.../FAGCN/data`

```
ln -s <DATA_DIR> data
```

### Experiments

In order to launch experiments run `run_train.sh` in `src` directory, specifying  visible `CUDA DEVICES` if necessary.

To compute averaged stats use `src/parse_results.py` with --result_path `<path_to_experiments>`.

# Reference
If you make advantage of the FAGCN model in your research, please cite the following in your manuscript:

```
@inproceedings{fagcn2021,
  title={Beyond Low-frequency Information in Graph Convolutional Networks},
  author={Deyu Bo and Xiao Wang and Chuan Shi and Huawei Shen},
  booktitle = {{AAAI}},
  publisher = {{AAAI} Press},
  year      = {2021}
}
```
