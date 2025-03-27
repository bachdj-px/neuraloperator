import torch

def compute_relative_l2_norm(pred, gt):
    norm_dims = tuple(range(1, pred.dim()))
    diff_norm = torch.norm(
        pred - gt,
        p=2,
        dim=norm_dims,
        keepdim=False,
    )
    target_norm = torch.norm(
        gt,
        p=2,
        dim=norm_dims,
        keepdim=False,
    )
    return diff_norm / target_norm