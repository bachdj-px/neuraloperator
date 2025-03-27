from neuralop.data.datasets import DictDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.data_processors import DataProcessor
import torch

def loader_to_residual_db(model, data_processor, loader, constant, device, train_val_split=True):
    """
    loader_to_residual_db converts a dataset of x: a(x), y: u(x) to
    x: a(x), y: G(a,x) - u(x) for use training the residual model.

    model : nn.Module
        trained solution model (frozen)
    data_processor: DataProcessor
        data processor used to train solution model
    loader: DataLoader
        data loader to convert to a dataloader of residuals
        must be drawn from the same distribution as the solution
        model's training distribution
    device: str or torch.device
    train_val_split: whether to split into a training and validation dataset, default True
    """
    error_list = []
    x_list = []
    model = model.to(device)
    model.eval()
    data_processor.eval()  # unnormalized y
    data_processor = data_processor.to(device)
    for idx, sample in enumerate(loader):
        sample = data_processor.preprocess(sample)
        out = model(**sample)
        out, sample = data_processor.postprocess(out, sample)  # unnormalize output

        # x_list.append({k: v.to("cpu") for k, v in sample.items() if k in ["in_p", "out_p", "f"]})
        x_list.append(sample)
        # x_list.append(sample['in_p'].to("cpu"))
        error = (
            (out - sample["y"]).detach().to("cpu")
        )  # detach, otherwise residual carries gradient of model weight
        # error is unnormalized here
        error_list.append(error)

        del sample, out
    errors = torch.cat(error_list, axis=0)
    # xs = torch.cat(x_list, axis=0) # check this

    residual_encoder = UnitGaussianNormalizer()
    residual_encoder.fit(errors)

    class CustomDataPrecessor(DataProcessor):
        def __init__(self, out_normalizer, device="cpu"):
            super().__init__()
            self.out_normalizer = out_normalizer
            self.device = device
            self.model = None

        def preprocess(self, data_dict, batched=True):
            y = data_dict["y"].to(self.device)
            for k in [
                "in_p",
                "out_p",
                "f",
                "inward_normals",
                "flow_normals",
                "vol_elm",
            ]:
                if data_dict[k].shape[0] == 1:
                    data_dict[k] = data_dict[k].squeeze(0)
                data_dict[k] = data_dict[k].to(self.device)
            if self.training:
                data_dict["y"] = self.out_normalizer.transform(y)
                
            data_dict["y"] = data_dict["y"].to(self.device)
            # print("Preprocessed y - should be normalized for the loss")
            # print(data_dict["y"])
            return data_dict

        def postprocess(self, output, data_dict):
            if not self.training:
                output = self.out_normalizer.inverse_transform(output)
            # print("Postprocessed pred - should not be changed and should be normalized")
            # print(output)
            return output, data_dict

        def to(self, device):
            self.device = device
            self.out_normalizer = self.out_normalizer.to(device)
            return self

        def forward(self, **data_dict):
            data_dict = self.preprocess(data_dict)
            output = self.model(data_dict["x"])
            output = self.postprocess(output)
            return output, data_dict

    # positional encoding and normalization already applied to X values
    residual_data_processor = CustomDataPrecessor(out_normalizer=residual_encoder, device=device)
    residual_data_processor.train()

    if train_val_split:
        num_samples = len(x_list)
        val_start = int(0.8 * num_samples)
        num_vertices = int(len(errors) / num_samples)

        xs = []
        for i in range(num_samples):
            sample_errors = errors[i * num_vertices : (i + 1) * num_vertices]
            sample = {k: v for k, v in x_list[i].items() if v is not None}
            sample.update({"y": sample_errors})
            xs.append(sample)

        residual_train_db = DictDataset(
            data_list=xs[:val_start], constant=constant
        )
        residual_val_db = DictDataset(
            data_list=xs[val_start:], constant=constant
        )
        # residual_train_db = TensorDataset(x=x_list[:val_start], y=errors[:val_start])
        # residual_val_db = TensorDataset(x=x_list[val_start:], y=errors[val_start:])
    else:
        residual_val_db = None
    return residual_train_db, residual_val_db, residual_data_processor