#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F

from dataset_utils import DataLoader
from sklearn.metrics import roc_auc_score
from GNN_models import *
from tqdm import tqdm


@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean().item()

@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy())   


def RunExp(args, dataset, data, Net, split_id):
    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, data = appnp_net.to(device), data.to(device)

    train_mask, val_mask, test_mask = \
        data.train_mask[:, split_id], data.val_mask[:, split_id], data.test_mask[:, split_id]

    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)[train_mask]
        loss = F.nll_loss(out, data.y[train_mask])
        loss.backward()
        optimizer.step()
        del out

    @torch.no_grad()
    def evaluate(model, data, metric: accuracy):
        model.eval()
        # predict on whole data
        logits = model(data)
        stats = {}
        for partition, mask in zip(['val', 'test'], [val_mask, test_mask]):
            loss = F.nll_loss(logits[mask], data.y[mask]).item()
            metric_value = metric(logits[mask], data.y[mask])
            stats[f'{partition}/loss'] = loss
            stats[f'{partition}/metric'] = metric_value
            
        return stats

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    metric_fn = accuracy if dataset.num_classes > 2 else roc_auc

    best_epoch = 0
    best_val_metric = 0
    test_metric = 0
    patience = 0

    for epoch in range(args.epochs):
        train(model, optimizer, data)

        stats = evaluate(model, data, metric_fn)

        if stats['val/metric'] > best_val_metric:
            best_val_metric = stats['val/metric']
            test_metric = stats['test/metric']
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha
            Gamma_0 = Alpha
            # set patience to 0
            patience = 0

        else:
            patience += 1
        if patience >= args.early_stopping:
            break

    return test_metric, best_val_metric, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', 
                        type=str, 
                        choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN'],
                        default='GPRGNN')

    args = parser.parse_args()

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN

    dname = args.dataset
    dataset, data = DataLoader(dname)

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print(f'Number of classes: {dataset.num_classes}')
    print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []
    assert RPMAX <= data.train_mask.shape[1] 

    for RP in tqdm(range(RPMAX)):
        test_acc, best_val_acc, Gamma_0 = RunExp(args, dataset, data, Net, RP)
        Results0.append([test_acc, best_val_acc, Gamma_0])

    test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.2f} +- {test_acc_std:.2f}')
