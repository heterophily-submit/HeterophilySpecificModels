import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utils.statistic import *
from utils.metric import accuracy, roc_auc, compute_correct_num, compute_sigma_acc
from data_loader import data_loaders
from hyperparameters_setting import *
import numpy as np
import random
import pandas as pd
from collections import defaultdict as ddt
from torch_geometric.utils import add_remaining_self_loops
import os


def train(args, model, optimizer):
    model.train()
    dataset = args.dataset
    device = args.device
    if len(dataset['graph'][0].train_mask.shape) != 1:
        train_mask = dataset['graph'][0].train_mask[:, args.split_id]
    else:
        train_mask = dataset['graph'][0].train_mask

    optimizer.zero_grad()
    loss = 0

    # choose metric
    if dataset['num_classes'] > 2:
        metric = accuracy
    else:
        metric = roc_auc

    for i in range(len(dataset['graph'])):
        data = dataset['graph'][i].to(device)
        out = model(data)
        regularizer_list = []
        if args.aug == True:
            out, sigma_list = out
            loss_function = CrossEntropyLoss()
            edge_train_mask = data.train_mask[data.edge_index[0]
                                              ] * data.train_mask[data.edge_index[1]]
            if len(edge_train_mask.shape) == 2:
                edge_train_mask = torch.unbind(edge_train_mask, dim=1)[0]
            for sigma in sigma_list:
                sigma = sigma[edge_train_mask]
                sigma_ = sigma.clone()
                for i in range(len(sigma)):
                    sigma_[i] = 1 - sigma[i]
                sigma = torch.cat(
                    (sigma_.unsqueeze(1), sigma.unsqueeze(1)), 1)
                regularizer = loss_function(
                    sigma.cuda(), torch.tensor(args.similarity, dtype=torch.long).cuda()[edge_train_mask])
                regularizer_list.append(regularizer)

            loss += F.nll_loss(out[train_mask],
                               data.y[train_mask]) + args.lamda * sum(regularizer_list)
        else:
            loss += F.nll_loss(out[train_mask],
                               data.y[train_mask])
        if len(dataset['graph']) == 1:  # single graph
            metric_train = metric(out[train_mask], data.y[train_mask])
        else:
            metric_train += metric(out[train_mask], data.y[train_mask]) * len(data.y)
    if not len(dataset['graph']) == 1:
        metric_train = metric_train / dataset['num_node']
    loss.backward()
    optimizer.step()
    return loss, metric_train


def test(args, model, mask_type="test"):
    model.eval()
    dataset = args.dataset
    device = args.device

    # choose metric
    if dataset['num_classes'] > 2:
        metric = accuracy
    else:
        metric = roc_auc

    if mask_type == "test":
        if len(dataset['graph'][0].test_mask.shape) != 1:  # multiple splits
            mask = dataset['graph'][0].test_mask[:, args.split_id]
        else:
            mask = dataset['graph'][0].test_mask
    if mask_type == "val":
        if len(dataset['graph'][0].val_mask.shape) != 1:  # multiple splits
            mask = dataset['graph'][0].val_mask[:, args.split_id]
        else:
            mask = dataset['graph'][0].val_mask

    assert len(dataset['graph']) == 1
    data = dataset['graph'][0].to(device)
    if args.aug == True:
        out, _ = model(data)[0].max(dim=1)
        sigma0 = model(data)[1][0].tolist()
        sigma_acc = compute_sigma_acc(sigma0, args.similarity)
        print('Sigma Accuracy: {:.4f}'.format(sigma_acc))
    else:
        out = model(data)

    metric_test = metric(out[mask], data.y[mask])
    return metric_test


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


def main():
    args = HyperparameterSetting("node classification").get_args()
    set_random_seed(args.seed)
    experiment_ans = ddt(lambda: [])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    data_name = args.dataset_name
    model_name = args.model_type
    args.dataset = data_loaders.DataLoader(args).dataset

    experiment_ans["datasetName"].append(data_name)
    experiment_ans["nodeNum"].append(args.dataset["num_node"])
    experiment_ans["edgeNum"].append(args.dataset["num_edge"])
    experiment_ans["nodeFeaturesDim"].append(
        args.dataset["num_node_features"])
    experiment_ans["nodeClassification"].append(
        args.dataset["num_node_classes"])
    experiment_ans["smoothness"].append(
        compute_smoothness(args.dataset["graph"][0]))

    if model_name != "GraphSage":
        edge_index, _ = add_remaining_self_loops(
            args.dataset["graph"][0].edge_index, None, 1, args.dataset["num_node"])
    else:
        edge_index = args.dataset["graph"][0].edge_index
    args.similarity = compute_cosine_similarity(
        args.dataset, edge_index, "label")
    set_random_seed(args.seed)

    model = MODEL_CLASSES[args.model_type](args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    counter: int = args.patience
    for epoch in range(args.epoch):
        loss, train_acc = train(args, model, optimizer)
        val_acc = test(args, model, mask_type="val")
        tmp_test_acc = test(args, model, mask_type="test")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
            # reset counter
            counter = args.patience
        else:
            counter -= 1
        
        if counter <= 0:
            print(f'Early stopping on epoch {epoch}.')
            break

        if (epoch + 1) % args.log_interval == 0:
            print('epoch: {:03d}'.format(epoch + 1),
                  'loss: {:.4f}'.format(loss),
                  'train_acc: {:.4f}'.format(train_acc),
                  'val_acc: {:.4f}'.format(val_acc),
                  'test_acc: {:.4f}'.format(tmp_test_acc),
                  'final_test_acc: {:.4f}'.format(test_acc)
                  )

    print('*' * 10)
    print('Final_test_acc: {:.4f}'.format(test_acc))

    model_train_res = model_name + "TrainAcc"
    model_val_res = model_name + "ValAcc"
    model_test_res = model_name + "TestAcc"
    experiment_ans[model_train_res].append(train_acc)
    experiment_ans[model_val_res].append(best_val_acc)
    experiment_ans[model_test_res].append(test_acc)
    if args.aug:
        experiment_ans["sigma_acc"].append(
            test(args, model, mask_type="test", return_sigma_acc=True)[1])

    df = pd.DataFrame(experiment_ans)

    if not os.path.exists(f'./saved/log/{args.dataset_name}'):
        os.makedirs(f'./saved/log/{args.dataset_name}')
    df.to_csv(
        f'./saved/log/{args.dataset_name}/lr{args.lr}_weightdecay{args.weight_decay}_lamda{args.lamda}_dataset{args.dataset_name}_model{args.model_type}_split{args.split}_aug{args.aug}.csv')

    method = 'gbk' if args.aug else 'baseline'
    if not os.path.exists(f'./saved/model/{args.dataset_name}/{method}/{args.split}/{args.model_type}'):
        os.makedirs(
            f'./saved/model/{args.dataset_name}/{method}/{args.split}/{args.model_type}')
    torch.save(model.state_dict(
    ), f'./saved/model/{args.dataset_name}/{method}/{args.split}/{args.model_type}/{args.model_path}')


if __name__ == "__main__":
    main()
