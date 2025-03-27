from copy import deepcopy
from pathlib import Path
import random
import pickle
import sys

from neuralop.losses.data_losses import PointwiseQuantileLoss
from neuralop.models import UQNO
from neuralop.training.trainer import Trainer
from neuralop.training import setup
from neuralop import get_model
from torch.utils.data import DataLoader
import torch

from cfd_data_preprocessor import CFDDataProcessor
from pipeline_config import get_config
from pipeline_data import get_data
from loss import get_losses
from optimizer import get_training_optimizer, get_residual_optimizer
from residual_db import loader_to_residual_db
from uqno_data_preprocessor import UQNODataProcessor
from uq_utils import get_coverage


def train_solution_model(
    solution_model,
    solution_train_loader,
    test_loader,
    optimizer,
    scheduler,
    train_loss_fn,
    test_loss_fn,
    config,
    data_processor,
    device,
    is_logger,
):
    trainer = Trainer(
        model=solution_model,
        n_epochs=config.opt.solution.n_epochs,
        data_processor=data_processor,
        device=device,
        mixed_precision=config.opt.solution.amp_autocast,
        wandb_log=config.wandb.log,
        verbose=is_logger,
    )

    if config.opt.solution.n_epochs > 0:
        trainer.train(
            train_loader=solution_train_loader,
            test_loaders={"": test_loader},
            optimizer=optimizer,
            scheduler=scheduler,
            training_loss=train_loss_fn,
            # eval_losses={config.opt.testing_loss: test_loss_fn, 'drag': DragLoss},
            eval_losses={config.opt.solution.testing_loss: test_loss_fn},
            regularizer=None,
            save_every=20,
            # save_best="l2",
            save_dir=Path(config.save_dir) / "solution_ckpts",
        )

        eval_metrics = trainer.evaluate({"l2": test_loss_fn}, data_loader=test_loader, epoch=1) 
        print(f"Eval metrics = {eval_metrics}")

def train_residual_model(
    residual_model,
    residual_train_loader,
    residual_val_loader,
    residual_optimizer,
    residual_scheduler,
    quantile_loss,
    l2loss,
    config,
    residual_data_processor,
    device,
    is_logger,
):

    if config.opt.residual.n_epochs > 0:
        residual_trainer = Trainer(
            model=residual_model,
            n_epochs=config.opt.residual.n_epochs,
            data_processor=residual_data_processor,
            wandb_log=config.wandb.log,
            device=device,
            mixed_precision=config.opt.residual.amp_autocast,
            eval_interval=config.wandb.eval_interval,
            log_output=config.wandb.log_output,
            use_distributed=config.distributed.use_distributed,
            verbose=config.verbose and is_logger,
        )

        residual_trainer.train(
            train_loader=residual_train_loader,
            test_loaders={"test": residual_val_loader},
            optimizer=residual_optimizer,
            scheduler=residual_scheduler,
            regularizer=False,
            training_loss=quantile_loss,
            eval_losses={"quantile": quantile_loss, "l2": l2loss},
            save_best="test_quantile",
            save_dir=Path(config.save_dir) / "residual_ckpts",
        )




def main():
    config_path = "CONFIG_PATH" # Should be inside the config folder
    config = get_config(config_path)
    print(config)
    solution_model = get_model(config)

    # Set-up distributed communication, if using
    device, is_logger = setup(config)

    ########################################
    ############## Load data ###############
    ########################################

    (
        data_module,
        solution_train_db,
        solution_train_loader,
        residual_train_db,
        residual_train_loader_unprocessed,
        residual_calib_db,
        test_db,
        test_loader,
    ) = get_data(config)

    ########################################
    ####### Train the solution model #######
    ########################################

    optimizer, scheduler = get_training_optimizer(solution_model, config)

    l2loss, train_loss_fn, test_loss_fn = get_losses(config)

    if config.verbose and is_logger and config.opt.solution.n_epochs > 0:
        print("\n### MODEL ###\n", solution_model)
        print("\n### OPTIMIZER ###\n", optimizer)
        print("\n### SCHEDULER ###\n", scheduler)
        print("\n### LOSSES ###")
        print(f"\n * Train: {l2loss}")
        # print(f"\n * Test: {eval_losses}")
        print(f"\n### Beginning Training...\n")
        sys.stdout.flush()

    output_encoder = deepcopy(data_module.normalizers["press"]).to(device)
    data_processor = CFDDataProcessor(normalizer=output_encoder, device=device)

    # Train the solution model
    train_solution_model(
        solution_model,
        solution_train_loader,
        test_loader,
        optimizer,
        scheduler,
        train_loss_fn,
        test_loss_fn,
        config,
        data_processor,
        device,
        is_logger,
    )

    # load best-performing solution model
    solution_model = solution_model.from_checkpoint(
        save_folder=Path(config.save_dir) / "solution_ckpts", save_name="model"
    )
    solution_model = solution_model.to(device)

    # solution_rl2s = []
    # with torch.no_grad():
    #     for idx, sample in enumerate(test_loader):
    #         sample = data_processor.preprocess(sample)
    #         out = solution_model(**{k: sample.get(k) for k in ["in_p", "out_p", "f", "ada_in"]})
    #         out, sample = data_processor.postprocess(out, sample)
    #         solution_gt = sample["y"]
    #         print(out, solution_gt)
    #         rl2 = torch.norm(out - solution_gt) / torch.norm(solution_gt)
    #         solution_rl2s.append(rl2)
    #         print(f"Sample {idx} - L2 error: {rl2}")

    # print("Mean relative L2 norm of solution model: ", torch.mean(torch.stack(solution_rl2s)))

    ########################################
    ####### Train the residual model #######
    ########################################

    residual_model = deepcopy(solution_model)
    residual_model = residual_model.to(device)

    # Changed here to use PointwiseQuantileLoss with 0.9 alpha
    quantile_loss = PointwiseQuantileLoss(alpha=config.opt.alpha)

    residual_optimizer, residual_scheduler = get_residual_optimizer(residual_model, config)
    processed_residual_train_db, processed_residual_val_db, residual_data_processor = (
        loader_to_residual_db(
            solution_model, data_processor, residual_train_loader_unprocessed, data_module.constant, device
        )
    )

    residual_data_processor = residual_data_processor.to(device)
    residual_train_loader = DataLoader(
        processed_residual_train_db,
        batch_size=1,
        shuffle=True,
    )
    residual_val_loader = DataLoader(
        processed_residual_val_db,
        batch_size=1,
        shuffle=True,
    )
    train_residual_model(
        residual_model,
        residual_train_loader,
        residual_val_loader,
        residual_optimizer,
        residual_scheduler,
        quantile_loss,
        l2loss,
        config,
        residual_data_processor,
        device,
        is_logger,
    )

    # load best residual model
    residual_model = residual_model.from_checkpoint(
        save_name="best_model", save_folder=Path(config.save_dir) / "residual_ckpts"
    )
    residual_model = residual_model.to(device)

    # create full uqno and uqno data processor
    uqno = UQNO(base_model=solution_model, residual_model=residual_model)
    uqno_data_proc = UQNODataProcessor(
        base_data_processor=data_processor,
        resid_data_processor=residual_data_processor,
        device=device,
    )

    uqno_data_proc.eval()


    # list of (true error / uncertainty band), indexed by score
    val_ratio_list = []
    calib_loader = DataLoader(residual_calib_db, shuffle=False, batch_size=1)
    solution_rl2s, residual_rl2s = [], []
    num_calib_samples = len(residual_calib_db)
    sampled_indices = random.sample(range(num_calib_samples), min(num_calib_samples, 5))
    (Path(config.save_dir) / "for_plot").mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for idx, sample in enumerate(calib_loader):
            sample = uqno_data_proc.preprocess(sample)
            out = uqno(**{k: sample.get(k) for k in ["in_p", "out_p", "f", "ada_in"]})
            out, sample, g_hat = uqno_data_proc.postprocess(out, sample)  # .squeeze()
            # Eval solution
            solution_gt = sample["y"] + g_hat
   
            solution_rl2s.append(torch.norm(g_hat - solution_gt) / torch.norm(solution_gt))
            # Eval uncertainty
            residual_rl2s.append(torch.norm(out - sample["y"]) / torch.norm(sample["y"]))

            ratio = torch.abs(sample["y"]) / out
            val_ratio_list.append(ratio.squeeze().to("cpu"))

            # if idx in sampled_indices:
            #     item = {
            #         "target": solution_gt.cpu().numpy(),
            #         "pred": g_hat.cpu().numpy(),
            #         "uncertainty": out.cpu().numpy(),
            #         "error": sample["y"].cpu().numpy(),
            #         "idx": idx,
            #     }
            #     with open(Path(config.save_dir) / f"for_plot/item_{idx}.pkl", "wb") as f:
            #         pickle.dump(item, f)

            del sample, out, g_hat

    print("Mean relative L2 norm of solution model: ", torch.mean(torch.stack(solution_rl2s)))
    print("Mean relative L2 norm of residual model: ", torch.mean(torch.stack(residual_rl2s)))

    val_ratios = torch.stack(val_ratio_list)

    vr_view = val_ratios.view(val_ratios.shape[0], -1)

    (Path(config.save_dir) / "arrays").mkdir(parents=True, exist_ok=True)
    get_coverage(uqno_data_proc, val_ratios, config, calib_loader, test_loader, device, is_logger)


if __name__ == "__main__":
    main()
