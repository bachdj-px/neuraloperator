import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralop.models import UQNO

from uqno_data_preprocessor import UQNODataProcessor

def get_coeff_quantile_idx(alpha: float, delta: float, n_samples: int, n_gridpts: int):
    """
    get the index of (ranked) sigma's for given delta and t
    we take the min alpha for given delta
    delta is percentage of functions that satisfy alpha threshold in domain
    alpha is percentage of points in ball on domain
    return 2 idxs
    domain_idx is the k for which kth (ranked descending by ptwise |err|/quantile_model_pred_err)
    value we take per function
    func_idx is the j for which jth (ranked descending) value we take among n_sample functions
    Note: there is a min alpha we can take based on number of gridpoints, n and delta, we specify lower bounds lb1 and lb2
    t needs to be between the lower bound and alpha
    """
    lb = np.sqrt(-np.log(delta) / 2 / n_gridpts)
    t = (
        alpha - lb
    ) / 3 + lb  # if t too small, will make the in-domain estimate conservative
    # too large will make the across-function estimate conservative. so we find a moderate t value
    print(f"we set alpha (on domain): {alpha}, t={t}")
    percentile = alpha - t
    domain_idx = int(np.ceil(percentile * n_gridpts))
    print(f"domain index: {domain_idx}'th largest of {n_gridpts}")

    # get function idx
    function_percentile = (
        np.ceil((n_samples + 1) * (delta - np.exp(-2 * n_gridpts * t * t))) / n_samples
    )
    function_idx = int(np.ceil(function_percentile * n_samples))
    print(f"function index: {function_idx}'th largest of {n_samples}")
    return domain_idx, function_idx


def eval_coverage_bandwidth(uqno_model: UQNO, uqno_data_proc: UQNODataProcessor, test_loader: DataLoader, alpha: float, device: str = "cuda"):
    """
    Get percentage of instances hitting target-percentage pointwise coverage
    (e.g. pctg of instances with >1-alpha points being covered by quantile model)
    as well as avg band length
    """
    in_pred_list = []
    avg_interval_list = []

    uncertainty_preds = []
    solution_outputs = []
    gts = []
    true_errors = []

    with torch.no_grad():
        for _, sample in enumerate(test_loader):
            sample = {k: v.to(device) for k, v in sample.items() if torch.is_tensor(v)}
            unnorm_true_press = sample["unnorm_press"]
            gts.append(unnorm_true_press.to("cpu").numpy())
            sample = uqno_data_proc.preprocess(sample)
            out = uqno_model(**sample)
            uncertainty_pred, sample, unnorm_pred = uqno_data_proc.postprocess(
                out, sample
            )
            solution_outputs.append(unnorm_pred.to("cpu").numpy())
            uncertainty_preds.append(uncertainty_pred.to("cpu").numpy())
            pointwise_true_err = sample["y"]
            true_errors.append(pointwise_true_err.to("cpu").numpy())

            in_pred = (
                (torch.abs(pointwise_true_err) < torch.abs(uncertainty_pred))
                .float()
                .squeeze()
            )
            avg_interval = (
                torch.abs(uncertainty_pred.squeeze())
                .view(uncertainty_pred.shape[0], -1)
                .mean(dim=1)
            )
            avg_interval_list.append(avg_interval.to("cpu"))

            in_pred_flattened = in_pred.view(in_pred.shape[0], -1)
            in_pred_instancewise = (
                torch.mean(in_pred_flattened, dim=1) >= 1 - alpha
            )  # expected shape (batchsize, 1)
            in_pred_list.append(in_pred_instancewise.float().to("cpu"))

    in_pred = torch.cat(in_pred_list, axis=0)
    intervals = torch.cat(avg_interval_list, axis=0)
    mean_interval = torch.mean(intervals, dim=0)
    in_pred_percentage = torch.mean(in_pred, dim=0)
    print(
        f"{in_pred_percentage} of instances satisfy that >= {1 - alpha} pts drawn are inside the predicted quantile"
    )
    print(f"Mean interval width is {mean_interval}")
    return (
        mean_interval,
        in_pred_percentage,
        uncertainty_preds,
        solution_outputs,
        gts,
        true_errors,
    )


def get_coverage(uqno_data_proc: UQNODataProcessor, val_ratios: np.ndarray, config, calib_loader: DataLoader, test_loader: DataLoader, device: str = "cuda", is_logger: bool = True):
    for alpha in [0.05]:
        # for delta in [0.02, 0.05, 0.1]:
        for delta in [0.1]:
            # get quantile of domain gridpoints and quantile of function samples
            # Original is: darcy_discretization = train_db[0]['x'].shape[-1] ** 2 where  train_db[0]['x'] has shape [1000, 16, 16] for example
            discretization = calib_loader.dataset[0]["vertices"].shape[
                -2
            ]  # number of gridpoints
            try:
                domain_idx, function_idx = get_coeff_quantile_idx(
                    alpha,
                    delta,
                    n_samples=len(calib_loader),
                    n_gridpts=discretization,
                )
                if domain_idx <= 0:
                    raise ValueError(f"Domain index is {domain_idx}")
            except ValueError as e:
                # print(f"Failed with {alpha=} and {delta=}")
                print(e)
                continue

            val_ratios_pointwise_quantile = torch.topk(
                val_ratios.view(val_ratios.shape[0], -1), domain_idx + 1, dim=1
            ).values[:, -1]
            uncertainty_scaling_factor = torch.abs(
                torch.topk(val_ratios_pointwise_quantile, function_idx + 1, dim=0).values[
                    -1
                ]
            )  # if val_ratios_pointwise_quantile.shape[0] > 1 else function_idx
            print(f"scale factor: {uncertainty_scaling_factor}")

            uqno_data_proc.set_scale_factor(uncertainty_scaling_factor)

            uqno_data_proc.eval()
            print(f"------- for values {alpha=} {delta=} ----------")
            interval, percentage, uncertainty_preds, solution_outputs, gts, true_errors = (
                eval_coverage_bandwidth(test_loader=test_loader, alpha=alpha, device=device)
            )
            np.save(
                f"{config.save_dir}/arrays/test_set_uncertainty_preds_alpha_{alpha}_delta_{delta}.npy",
                np.array(uncertainty_preds),
            )
            np.save(
                f"{config.save_dir}/arrays/test_set_solution_outputs_alpha_{alpha}_delta_{delta}.npy",
                np.array(solution_outputs),
            )
            np.save(f"{config.save_dir}/arrays/test_set_gts_alpha_{alpha}_delta_{delta}.npy", np.array(gts))
            np.save(
                f"{config.save_dir}/arrays/test_set_true_errors_alpha_{alpha}_delta_{delta}.npy", np.array(true_errors)
            )
            np.save(
                f"{config.save_dir}/arrays/test_set_coverage_alpha_{alpha}_delta_{delta}.npy",
                np.array([percentage, interval]),
            )
            # print(f"-----------------------------------------")

            # interval, percentage, uncertainty_preds, solution_outputs, gts, true_errors = (
            #     eval_coverage_bandwidth(
            #         test_loader=calib_loader, alpha=alpha, device=device
            #     )
            # )
            # np.save(
            #     f"{config.save_dir}/arrays/calib_uncertainty_preds_alpha_{alpha}_delta_{delta}.npy",
            #     np.array(uncertainty_preds),
            # )
            # np.save(
            #     f"{config.save_dir}/arrays/calib_solution_outputs_alpha_{alpha}_delta_{delta}.npy",
            #     np.array(solution_outputs),
            # )
            # np.save(f"{config.save_dir}/arrays/calib_gts_alpha_{alpha}_delta_{delta}.npy", np.array(gts))
            # np.save(
            #     f"{config.save_dir}/arrays/calib_true_errors_alpha_{alpha}_delta_{delta}.npy",
            #     np.array(true_errors),
            # )