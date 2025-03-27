from neuralop.data.transforms.data_processors import DataProcessor
import torch

class CFDDataProcessor(DataProcessor):
    """
    Implements logic to preprocess data/handle model outputs
    to train an FNOGNO on the CFD car-pressure dataset
    """

    def __init__(self, normalizer, device="cuda"):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):
        # Turn a data dictionary returned by MeshDataModule's DictDataset
        # into the form expected by the FNOGNO

        in_p = sample["query_points"].squeeze(0).to(self.device)
        out_p = sample["centroids"].squeeze(0).to(self.device)

        f = sample["distance"].squeeze(0).to(self.device)

        weights = sample["triangle_areas"].squeeze(0).to(self.device)

        # Output data
        truth = sample["press"].squeeze(0).unsqueeze(-1)

        # Take the first 3682 vertices of the output mesh to correspond to pressure
        output_vertices = truth.shape[1]
        if out_p.shape[0] > output_vertices:
            out_p = out_p[:output_vertices, :]

        truth = truth.to(self.device)

        inward_normals = -sample["triangle_normals"].squeeze(0).to(self.device)
        flow_normals = torch.zeros((sample["triangle_areas"].shape[1], 3)).to(
            self.device
        )
        flow_normals[:, 0] = -1.0
        batch_dict = dict(
            in_p=in_p,
            out_p=out_p,
            f=f,
            y=truth,
            inward_normals=inward_normals,
            flow_normals=flow_normals,
            flow_speed=None,
            vol_elm=weights,
            reference_area=None,
        )

        sample.update(batch_dict)
        return sample

    def postprocess(self, out, sample):
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = self.normalizer.inverse_transform(sample["y"].squeeze(0))
            sample["y"] = y

        return out, sample

    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)
        return self

    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample