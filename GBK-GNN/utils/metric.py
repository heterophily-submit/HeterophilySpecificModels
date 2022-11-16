import torch

from sklearn.metrics import roc_auc_score

@torch.no_grad()
def accuracy(pr_logits, gt_labels):
    return (pr_logits.argmax(dim=-1) == gt_labels).float().mean().item()

@torch.no_grad()
def roc_auc(pr_logits, gt_labels):
    return roc_auc_score(gt_labels.cpu().numpy(), pr_logits[:, 1].cpu().numpy()) 

def compute_correct_num(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def compute_sigma_acc(sigma, ground_truth, bound=0.1):
    count = 0
    assert len(sigma) == len(ground_truth)
    for i in range(len(sigma)):
        if sigma[i] - ground_truth[i] <= bound:
            count += 1

    return count / len(sigma)
