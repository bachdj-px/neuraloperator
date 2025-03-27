
from neuralop.training import AdamW
import torch.nn as nn
import torch

def get_training_optimizer(model: nn.Module, config):
    optimizer = AdamW(
        model.parameters(),
        lr=config.opt.solution.learning_rate,
        weight_decay=config.opt.solution.weight_decay,
    )

    if config.opt.solution.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.opt.solution.gamma,
            patience=config.opt.solution.scheduler_patience,
            mode="min",
        )
    elif config.opt.solution.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.opt.solution.scheduler_T_max
        )
    elif config.opt.solution.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.opt.solution.step_size,
            gamma=config.opt.solution.gamma,
        )
    else:
        raise ValueError(f"Got {config.opt.solution.scheduler=}")
    
    return optimizer, scheduler


def get_residual_optimizer(residual_model, config):
    residual_optimizer = torch.optim.Adam(
        residual_model.parameters(),
        lr=config.opt.residual.learning_rate,
        weight_decay=config.opt.residual.weight_decay,
    )

    if config.opt.residual.scheduler == "ReduceLROnPlateau":
        residual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            residual_optimizer,
            factor=config.opt.residual.gamma,
            patience=config.opt.residual.scheduler_patience,
            mode="min",
        )
    elif config.opt.residual.scheduler == "CosineAnnealingLR":
        residual_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            residual_optimizer, T_max=config.opt.residual.scheduler_T_max
        )
    elif config.opt.residual.scheduler == "StepLR":
        residual_scheduler = torch.optim.lr_scheduler.StepLR(
            residual_optimizer,
            step_size=config.opt.solution.step_size,
            gamma=config.opt.solution.gamma,
        )
    else:
        raise ValueError(f"Got residual scheduler={config.opt.residual.scheduler}")
    return residual_optimizer, residual_scheduler