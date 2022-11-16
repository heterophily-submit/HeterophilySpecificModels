import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter

__all__ = [
    "FSGNN",
    "FSGNN_Large"
]


class FSGNN(nn.Module):

    def __init__(
        self, 
        nfeat, 
        nlayers, 
        nhidden, 
        nclass, 
        dropout,
        layer_norm: bool = False,
    ):
        super(FSGNN, self).__init__()
        self.fc2 = nn.Linear(nhidden * nlayers,nclass)
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.layer_norm = layer_norm

    def forward(self, list_mat):
        mask = self.sm(self.att)
        list_out = list()
        for ind, mat in enumerate(list_mat):
            tmp_out = self.fc1[ind](mat)
            if self.layer_norm == True:
                tmp_out = F.normalize(tmp_out, p=2, dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)
            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)

        return out


class FSGNN_Large(nn.Module):

    def __init__(
        self,
        nfeat,
        nlayers,
        nhidden,
        nclass,
        dp1,
        dp2,
        layer_norm: bool = True
    ):
        super(FSGNN_Large,self).__init__()
        self.wt1 = nn.ModuleList([nn.Linear(nfeat,int(nhidden)) for _ in range(nlayers)])
        self.fc2 = nn.Linear(nhidden*nlayers,nhidden)
        self.fc3 = nn.Linear(nhidden,nclass)
        self.dropout1 = dp1 
        self.dropout2 = dp2 
        self.act_fn = nn.ReLU()
        self.att = nn.Parameter(torch.ones(nlayers))
        self.sm = nn.Softmax(dim=0)
        self.layer_norm = layer_norm


    def forward(self, list_adj, st=0, end=0):

        mask = self.sm(self.att)
        mask = torch.mul(len(list_adj),mask)

        list_out = list()
        for ind, mat in enumerate(list_adj):
            mat = mat[st:end,:].cuda()
            tmp_out = self.wt1[ind](mat)
            if self.layer_norm == True:
                tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[ind],tmp_out)

            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)

        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout1,training=self.training)
        out = self.fc2(out)

        out = self.act_fn(out)
        out = F.dropout(out,self.dropout2,training=self.training)
        out = self.fc3(out)

        return out

