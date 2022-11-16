import torch

from torch_geometric.data import Data


def zero_in_degree_removal(data):
    edge_index = data.edge_index
    # keep only with non-zero incoming edges
    valid_ids = torch.unique(edge_index[1])
    node_mask = torch.zeros(len(data.y), dtype=torch.bool)
    node_mask[valid_ids] = True
    valid_mask = edge_index[0].clone().apply_(lambda x: x in valid_ids).bool()
    valid_edges = torch.masked_select(data.edge_index, valid_mask).view(2, -1)
    return Data(
        x=data.x,
        y=data.y,
        edge_index=valid_edges,
        train_mask=(data.train_mask & node_mask[:, None]),
        val_mask=(data.val_mask & node_mask[:, None]),
        test_mask=(data.test_mask & node_mask[:, None]),
    )
