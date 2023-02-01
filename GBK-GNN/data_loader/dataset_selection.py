import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.datasets import Actor, WikipediaNetwork, WebKB

from utils.transform import zero_in_degree_removal
from utils.statistic import compute_smoothness, split_dataset


DATASET_LIST = [
    'squirrel_directed', 'chameleon_directed',
    'squirrel_filtered_directed', 'chameleon_filtered_directed',
    'squirrel_filtered', 'chameleon_filtered',
    'roman_empire', 'minesweeper', 'questions', 'amazon_ratings', 'workers'
]


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


def get_dataset(dataset):
    if dataset == 'actor':
        return Actor(root='./pyg_data/actor')
    if dataset == 'squirrel':
        return WikipediaNetwork(root='./pyg_data', name='squirrel')
    if dataset == 'chameleon':
        return WikipediaNetwork(root='./pyg_data', name='chameleon')
    if dataset in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root='./pyg_data', name=dataset)
    if dataset in DATASET_LIST:
        return load_custom_data(
            f'./data/{dataset}.npz', 
            to_undirected='directed' not in dataset
        )
    raise ValueError("Unknown dataset")


class DatasetSelection:
    def __init__(
        self, 
        dataset_name, 
        split, 
        task_type="NodeClasification",
        remove_zero_degree_nodes: bool = False
    ):
        task2str = {
            "NodeClasification": "node_",
            "EdgeClasification": "edge_",
            "GraphClasification": "graph_"
        }
     
        dataset = get_dataset(dataset_name)      

        self.dataset = {"graph": []}
        smoothness = num_class = num_node = num_edge = 0
        for i in range(len(dataset)):
            num_node += dataset[i].x.shape[0]
            num_edge += dataset[i].edge_index.shape[1]
            if (dataset[i].y.shape == torch.Size([1]) and task_type == "NodeClasification"):
                dataset[i].y.data = dataset[i].x.argmax(dim=1)
                num_class = max(dataset[i].x.shape[1], num_class)
            else:
                if (len(dataset[i].y.shape) != 1):
                    num_class = max(dataset[i].y.shape[1], num_class)
                    dataset[i].y.data = dataset[i].y.argmax(dim=1)
                else:
                    num_class = max(max(dataset[i].y + 1), num_class)
            if not hasattr(dataset[i], 'train_mask'):
                data_tmp = dataset[i]
                data_tmp.train_mask, data_tmp.test_mask, data_tmp.val_mask = split_dataset(
                    dataset[i].x.shape[0], split)
                self.dataset["graph"].append(data_tmp)
                
            self.dataset["graph"].append(dataset[i])
            smoothness += compute_smoothness(dataset[i]) * \
                dataset[i].x.shape[0]


        if (type(num_class) != type(1)):
            num_class = num_class.numpy()

        smoothness /= num_node
        self.dataset['num_node'] = num_node
        self.dataset['num_edge'] = num_edge
        self.dataset['num_node_features'] = dataset[0].x.shape[1]
        self.dataset['smoothness'] = smoothness
        self.dataset['num_' + task2str[task_type] + 'classes'] = num_class
        self.dataset['num_classes'] = num_class

    def get_dataset(self):
        return self.dataset
