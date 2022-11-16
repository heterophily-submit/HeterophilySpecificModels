import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import accuracy


__all__ = ['train_step', 'val_step']


def train_step(
    model,
    optimizer,
    labels,
    list_mat,
    mask,
    metric = accuracy,
    device: str = 'cpu'
):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat)
    loss_train = F.cross_entropy(output[mask], labels[mask].to(device))
    acc_train = metric(output[mask], labels[mask].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train, acc_train


def val_step(
    model,
    labels,
    list_mat,
    mask,
    metric = accuracy,
    device: str = 'cpu'
):
    model.eval()
    with torch.no_grad():
        output = model(list_mat)
        loss_val = F.cross_entropy(output[mask], labels[mask].to(device))
        acc_val = metric(output[mask], labels[mask].to(device))
        return loss_val, acc_val
