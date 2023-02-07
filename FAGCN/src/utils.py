import numpy as np
import scipy.sparse as sp
import torch
import dgl

import os
from dgl.data import *
from sklearn.metrics import roc_auc_score


DATASET_LIST = [
    'squirrel_directed', 'chameleon_directed',
    'squirrel_filtered_directed', 'chameleon_filtered_directed',
    'roman_empire', 'minesweeper', 'questions', 'amazon_ratings', 'workers'
]


def preprocess_data(
    dataset, 
    train_ratio, 
    splits_file_path: str = None, 
    remove_zero_in_degree_nodes: bool = False
):

    if dataset in DATASET_LIST:
        npz_data = np.load(f'../data/{dataset}.npz')

        assert splits_file_path is not None
        split_id = int(splits_file_path)
        # convert graph to bidirectional
        if 'directed' not in dataset:
            edge = np.concatenate((npz_data['edges'], npz_data['edges'][:, ::-1]), axis=0)
        else:
            edge = npz_data['edges']
        labels = npz_data['node_labels']
        features = npz_data['node_features']

        train_mask = npz_data['train_masks'][split_id]
        val_mask   = npz_data['val_masks'][split_id]
        test_mask  = npz_data['test_masks'][split_id]

        valid_ids = (np.arange(len(labels)), np.unique(edge[:, 1]))[remove_zero_in_degree_nodes]

        U = [e[0] for e in edge if e[0] in valid_ids]
        V = [e[1] for e in edge if e[0] in valid_ids]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        train = np.flatnonzero(train_mask)
        val = np.flatnonzero(val_mask)
        test = np.flatnonzero(test_mask)

        nclass = len(np.unique(labels))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        return g, nclass, features, labels, train, val, test
    else:
        raise ValueError(f'dataset {dataset} not supported in dataloader')


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean().item()

@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy())   
