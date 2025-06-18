import torch
import torch.nn as nn

from utils.model_utils import check_if_models_have_same_architecture


class Ensemble(nn.Module):
    def __init__(self, models, percentages=None):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.percentages = percentages

    def forward(self, x):
        probs = [torch.softmax(model(x), dim=-1) for model in self.models]
        if self.percentages is not None:
            probs = [probs[i] * self.percentages[i] for i in range(len(probs))]
        return torch.stack(probs).mean(0)


class VanillaAveraging(nn.Module):
    def __init__(self, models):
        super(VanillaAveraging, self).__init__()

        # initialize the model to be the first model
        self.model = models[0].copy_model()
        self.device = self.model.device

        # iterate over the remaining models
        for idx, model in enumerate(models[1:], start=2):
            if not check_if_models_have_same_architecture(self.model, model):
                raise ValueError(f'Model {idx} has a different architecture from the first model -- '
                                 f'cannot perform averaging, since VanillaAveraging requires '
                                 f'that all models have the same architecture')
            # add the weights of the model to the current model
            for key, value in model.state_dict().items():
                self.model.state_dict()[key].data.add_(value)

        # divide by the number of models to get the average
        for key in self.model.state_dict():
            self.model.state_dict()[key].data.div_(len(models))

    def forward(self, x):
        return self.model(x)
