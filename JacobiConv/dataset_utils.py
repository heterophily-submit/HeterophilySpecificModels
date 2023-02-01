# copied from https://github.com/ivam-he/BernNet
# load the real-world dataset from the `"BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation" paper
import torch
import numpy as np
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.datasets import WikipediaNetwork, Actor, WebKB


DATASET_LIST = [
    'squirrel_directed', 'chameleon_directed',
    'squirrel_filtered_directed', 'chameleon_filtered_directed',
    'roman_empire', 'minesweeper', 'questions', 'amazon_ratings', 'workers', 'sbm_counter'
]


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

# GPRGNN
def random_planetoid_splits(
    data,
    num_classes,
    percls_trn=20,
    val_lb=500,
    Flag=0
):
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0), device=index.device)]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=data.num_nodes)
        val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat(
            [i[percls_trn:percls_trn + val_lb] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn + val_lb:] for i in indices],
                               dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=data.num_nodes)
        val_mask = index_to_mask(val_index, size=data.num_nodes)
        test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return train_mask, val_mask, test_mask


def load_custom_data(data_path, to_undirected: bool = True):
    npz_data = np.load(data_path)
    # convert graph to bidirectional
    if to_undirected:
        edges = np.concatenate((npz_data['edges'], npz_data['edges'][:, ::-1]), axis=0)
    else:
        edges = npz_data['edges']
    
    data = Data(
        x=torch.from_numpy(npz_data['node_features']),
        y=torch.from_numpy(npz_data['node_labels']),
        edge_index=torch.from_numpy(edges).T,
        train_mask=torch.from_numpy(npz_data['train_masks']).T,
        val_mask=torch.from_numpy(npz_data['val_masks']).T,
        test_mask=torch.from_numpy(npz_data['test_masks']).T,
    )
    return [data]


def DataLoader(name):
    if name == 'actor':
        return Actor(root='./data/actor', transform=T.NormalizeFeatures())
    if name == 'squirrel':
        return WikipediaNetwork(root='./data', name='squirrel', transform=T.NormalizeFeatures())
    if name == 'chameleon':
        return WikipediaNetwork(root='./data', name='chameleon', transform=T.NormalizeFeatures())
    if name in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root='./data/',  name=name, transform=T.NormalizeFeatures())
    if name in DATASET_LIST:
        root_path = './data'
        dataset = load_custom_data(f'{root_path}/{name}.npz', to_undirected='directed' not in name)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset
    