from neuralop.losses.data_losses import LpLoss

def get_losses(config):
    l2loss = LpLoss(d=2, p=2)

    if config.opt.solution.training_loss == "l2":
        train_loss_fn = l2loss
    else:
        raise ValueError(f"Got {config.opt.solution.training_loss=}")

    if config.opt.solution.testing_loss == "l2":
        test_loss_fn = l2loss
    else:
        raise ValueError(f"Got {config.opt.solution.testing_loss=}")
    
    return l2loss, train_loss_fn, test_loss_fn