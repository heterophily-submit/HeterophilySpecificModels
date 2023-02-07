# FSGNN

This repositoty is based on the implementation of [FSGNN](https://github.com/sunilkmaurya/FSGNN). 
For more details, please refer to [the paper](https://arxiv.org/abs/2105.07634).
This work is further extended into our [second](https://arxiv.org/abs/2111.06748) paper.

# Usage

### Data preparation

Create a symbolic or copy to the network directory, i.e `.../FSGNN/data`

```
ln -s <DATA_DIR> data
```

### Experiments

In order to launch experiments run `./run_train_fsgnn.sh` in the root directory.

To setup the experiment edit `run_train_fsgnn.sh`.

To compute averaged stats use `parse_results.py` with --result_path `<path_to_experiments>`.

