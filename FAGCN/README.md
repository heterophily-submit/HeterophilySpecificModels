# FAGCN
Code of [Beyond Low-frequency Information in Graph Convolutional Networks](http://shichuan.org/doc/102.pdf)

# Usage

### Installation

Requirements are given in `requirements.txt`

### Data preparation

Due to the lack of space we do not copy the saved features for FAGCN. 
Please refer to the original repository `https://github.com/bdy9527/FAGCN` and download and 
extract the [archived features](https://github.com/bdy9527/FAGCN/blob/main/FAGCN.zip).

For new datasets make symbolic link to the root in a following way:

```
ln -s <DATA_DIR> new_data
```

### Experiments

In order to launch experiments run `run_train.sh` in `src` directory.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.

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
