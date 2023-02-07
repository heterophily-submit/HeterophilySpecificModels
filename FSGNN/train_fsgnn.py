import torch
import random
import argparse
import numpy as np

from copy import deepcopy
from torch_geometric.utils import to_dense_adj, add_self_loops

from models import FSGNN
from utils.metrics import accuracy, roc_auc
from utils.transform import zero_in_degree_removal
from training.engine_fsgnn import train_step, val_step
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, WikipediaNetwork, WebKB


DATASET_LIST = [
    'squirrel_directed', 'chameleon_directed',
    'squirrel_filtered_directed', 'chameleon_filtered_directed',
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


def get_dataset(args):
    if args.dataset == 'actor':
        return Actor(root='./pyg_data/actor')
    if args.dataset == 'squirrel':
        return WikipediaNetwork(root='./pyg_data', name='squirrel')
    if args.dataset == 'chameleon':
        return WikipediaNetwork(root='./pyg_data', name='chameleon')
    if args.dataset in ['cornell', 'texas', 'wisconsin']:
        return WebKB(root='./pyg_data', name=args.dataset)
    if args.dataset in DATASET_LIST:
        return load_custom_data(
            f'./data/{args.dataset}.npz', 
            to_undirected='directed' not in args.dataset
        )
    raise ValueError("Unknown dataset")


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
    parser.add_argument('--steps', type=int, default=1500, help='Number of epochs to train.')
    parser.add_argument('--log-freq', type=int, default=100, help='Logging frequency.')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of layers.')
    parser.add_argument('--hidden-dim', type=int, default=64, help='hidden dimensions.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--dataset', default='actor', help='dataset')
    parser.add_argument('--layer-norm', type=int, default=1, help='layer norm')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--feat-type', type=str, default='all', help='Type of features to be used')
    parser.add_argument('--transform', action='store_true')
    args = parser.parse_args()
    return args


def run_on_split(
    features, 
    labels,
    list_mat, 
    train_mask, 
    val_mask, 
    test_mask, 
    args
):
    num_features = features.shape[1]
    num_labels = len(torch.unique(labels))
    
    model = FSGNN(  
        nfeat=num_features,
        nlayers=len(list_mat),
        nhidden=args.hidden_dim,
        nclass=num_labels,
        dropout=args.dropout,
        layer_norm=args.layer_norm,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metric = accuracy if len(torch.unique(labels)) > 2 else roc_auc

    best = -torch.inf
    best_params = None
    bad_counter = 0
    for step in range(args.steps):
        loss_train, metric_train = train_step(model, optimizer, labels, list_mat, train_mask, metric, device=args.device)
        loss_val, metric_val = val_step(model, labels, list_mat, val_mask, metric, device=args.device)

        if step % args.log_freq == 0:
            print(f'Train metric {metric_train:.3f} / Val acc {metric_val:.3f}')

        if metric_val > best:
            best = metric_val
            bad_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    # load best params
    model.load_state_dict(best_params)
    loss_test, metric_test = val_step(model, labels, list_mat, test_mask, metric, device=args.device)
    # return test accuracy
    return metric_test


if __name__ == '__main__':
    args = parse_args()
    # fix seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    # get dataset
    dataset = get_dataset(args)
    data = dataset[0].to(args.device)

    features = data.x
    labels = data.y
    # get adjacency matrix and its powers
    adj = to_dense_adj(data.edge_index)[0]
    adj_i = to_dense_adj(add_self_loops(data.edge_index)[0])[0]

    list_mat = []
    list_mat.append(features)
    no_loop_mat = features
    loop_mat = features

    for ii in range(args.num_layers):
        no_loop_mat = torch.spmm(adj, no_loop_mat)
        loop_mat = torch.spmm(adj_i, loop_mat)
        list_mat.append(no_loop_mat)
        list_mat.append(loop_mat)

    # Select X and self-looped features 
    if args.feat_type == "homophily":
        select_idx = [0] + [2 * ll for ll in range(1, args.num_layers + 1)]
        list_mat = [list_mat[ll] for ll in select_idx]

    #Select X and no-loop features
    elif args.feat_type == "heterophily":
        select_idx = [0] + [2*ll - 1 for ll in range(1, args.num_layers + 1)]
        list_mat = [list_mat[ll] for ll in select_idx]

    num_splits = data.train_mask.shape[1]
    test_accs = []
    for i in range(num_splits):
        print(f'Split [{i+1}/{num_splits}]')
        train_mask, val_mask, test_mask = \
            data.train_mask[:, i], data.val_mask[:, i], data.test_mask[:, i]
        test_acc = run_on_split(features, labels, list_mat, train_mask, val_mask, test_mask, args)
        print(f'Test accuracy {test_acc:.3f}')
        test_accs.append(100 * test_acc)

    print(f'Test accuracy {np.mean(test_accs):.2f} +- {np.std(test_accs):.2f}')
