from neuralop.data.transforms.data_processors import DataProcessor

class UQNODataProcessor(DataProcessor):
    def __init__(
        self,
        base_data_processor: DataProcessor,
        resid_data_processor: DataProcessor,
        device: str = "cpu",
    ):
        """UQNODataProcessor converts tuple (G_hat(a,x), E(a,x)) and
        sample['y'] = G_true(a,x) into the form expected by PointwiseQuantileLoss

        y_pred = E(a,x)
        y_true = abs(G_hat(a,x) - G_true(a,x))

        It also preserves any transformations that need to be performed
        on inputs/outputs from the solution model.

        Parameters
        ----------
        base_data_processor : DataProcessor
            transforms required for base solution_model input/output
        resid_data_processor : DataProcessor
            transforms required for residual input/output
        device: str
            "cpu" or "cuda"
        """
        super().__init__()
        self.base_data_processor = base_data_processor
        self.residual_normalizer = resid_data_processor.out_normalizer

        self.device = device
        self.scale_factor = None

    def set_scale_factor(self, factor):
        self.scale_factor = factor.to(self.device)

    def wrap(self, model):
        self.model = model
        return self

    def to(self, device):
        self.device = device
        self.base_data_processor = self.base_data_processor.to(device)
        self.residual_normalizer = self.residual_normalizer.to(device)
        return self

    def train(self):
        self.base_data_processor.train()

    def eval(self):
        self.base_data_processor.eval()

    def preprocess(self, *args, **kwargs):
        """
        nothing required at preprocessing - just wrap the base DataProcessor
        """
        return self.base_data_processor.preprocess(*args, **kwargs)

    def postprocess(self, out, sample):
        """
        unnormalize the residual prediction as well as the output
        """
        self.base_data_processor.eval()
        g_hat, pred_uncertainty = out  # UQNO returns a tuple

        pred_uncertainty = self.residual_normalizer.inverse_transform(pred_uncertainty)
        # this is normalized

        g_hat, sample = self.base_data_processor.postprocess(
            g_hat, sample
        )  # unnormalize g_hat

        g_true = sample["y"]  # this is unnormalized in eval mode
        sample["y"] = g_true - g_hat  # both unnormalized

        if self.scale_factor is not None:
            pred_uncertainty = pred_uncertainty * self.scale_factor
        return pred_uncertainty, sample, g_hat

    def forward(self, **sample):
        # combine pre and postprocess for wrap
        sample = self.preprocess(sample)
        out = self.model(**sample)
        out, sample = self.postprocess(out, sample)
        return out, sample, None