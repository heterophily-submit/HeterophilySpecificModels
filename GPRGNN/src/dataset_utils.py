#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import os
import torch
import pickle
import numpy as np
import os.path as osp
import torch_geometric.transforms as T


from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor, WebKB


DATASET_LIST = [
    'squirrel_directed', 'chameleon_directed',
    'squirrel_filtered_directed', 'chameleon_filtered_directed',
    'roman_empire', 'minesweeper', 'questions', 'amazon_ratings', 'tolokers', 'sbm_counter'
]


class dataset_heterophily(InMemoryDataset):
    
    def __init__(self, root='data/', name=None,
                 p2raw=None,
                 train_percent=0.01,
                 transform=None, pre_transform=None):

        existing_dataset = ['chameleon', 'film', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(
                f'name of hypergraph dataset must be one of: {existing_dataset}')
        else:
            self.name = name

        self._train_percent = train_percent

        if (p2raw is not None) and osp.isdir(p2raw):
            self.p2raw = p2raw
        elif p2raw is None:
            self.p2raw = None
        elif not osp.isdir(p2raw):
            raise ValueError(
                f'path to raw hypergraph dataset "{p2raw}" does not exist!')

        if not osp.isdir(root):
            os.makedirs(root)

        self.root = root

        super(dataset_heterophily, self).__init__(
            root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.train_percent = self.data.train_percent

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        file_names = [self.name]
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        p2f = osp.join(self.raw_dir, self.name)
        with open(p2f, 'rb') as f:
            data = pickle.load(f)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


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
    return data


class SingleGraphDataset(Dataset):

    def __init__(self, data):
        self.data = data
        self.num_classes = len(torch.unique(data.y))
        self.num_features = data.x.shape[-1]

    def __len__(self):
        return 1

    def __getitem__(self, idx: int) -> Data:
        if idx != 0:
            raise ValueError("Invalid index")
        else:
            return self.data


def DataLoader(name):

    if name in ['chameleon', 'squirrel']:
        transform = T.NormalizeFeatures()
        preProcDs = WikipediaNetwork(root=f"../pyg_data", 
            name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
        dataset = WikipediaNetwork(root=f"../pyg_data", 
            name=name, geom_gcn_preprocess=True, transform=transform)
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
        return dataset, data
    elif name in ['film']:
        dataset = Actor(root='../pyg_data/actor', transform=T.NormalizeFeatures())
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(root='../pyg_data/', name=name, transform=T.NormalizeFeatures())
    elif name in DATASET_LIST:
        data = load_custom_data(f'../data/{name}.npz', to_undirected='directed' not in name)
        dataset = SingleGraphDataset(data)
        return dataset, data
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset, dataset[0]
