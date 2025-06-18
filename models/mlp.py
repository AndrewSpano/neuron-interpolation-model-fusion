import torch
import torch.nn as nn

from models.base_model import BaseModel, WeightRepresentation, LevelRepresentation


class MLPNet(BaseModel):
    """A simple MLP with a few hidden layers."""

    def __init__(self, input_size, hidden_size, output_size, use_bias=True, use_activations=True,
                 train_indices=None, val_indices=None):
        super(MLPNet, self).__init__(train_indices, val_indices)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.use_activations = use_activations

        level1_layers = [nn.Flatten(start_dim=1), nn.Linear(input_size, hidden_size, bias=use_bias)]
        level2_layers = [nn.Linear(hidden_size, output_size, bias=use_bias)]
        if use_activations:
            level1_layers.append(nn.ReLU())
        else:
            level2_layers = [nn.ReLU()] + level2_layers

        self.level1 = nn.Sequential(*level1_layers)
        self.level2 = nn.Sequential(*level2_layers)

    def forward(self, x):
        x = self.level1(x)
        x = self.level2(x)
        return x

    def forward_until_level(self, x: torch.Tensor, level_idx: int = -1) -> torch.Tensor:
        x = x.view(x.shape[0], -1)

        # level 0
        x = self.level1(x)
        if level_idx == 0:
            return x

        # level 1
        x = self.fc2(x)
        return x

    def copy_model(self) -> 'MLPNet':
        """Creates a copy of the model with the same architecture,
            and the same weights as the current model."""
        model_copy = MLPNet(self.input_size, self.hidden_size, self.output_size,
                            self.use_bias, self.use_activations,
                            self.train_indices, self.val_indices)
        model_copy.load_state_dict(self.state_dict())
        return model_copy.to(self.device)

    @property
    def get_ordered_trainable_named_layers(self):
        """Returns the trainable layers of the model in the order they are applied."""
        fc1 = self.level1[1]
        fc2 = self.level2[-1]
        return [
            WeightRepresentation(name='fc1', weight=fc1.weight, layer=fc1),
            WeightRepresentation(name='fc2', weight=fc2.weight, layer=fc2)
        ]

    @property
    def get_ordered_trainable_named_levels(self) -> list[LevelRepresentation]:
        """Returns the trainable levels of the model in the order they are applied."""
        level1_name = 'fc1' + ('_with_relu' if self.use_activations else '')
        level2_name = ('relu_with_' if self.use_activations else '') + 'fc2'
        return [
            LevelRepresentation(name=level1_name, output_width=self.hidden_size, network=self.level1),
            LevelRepresentation(name=level2_name, output_width=self.output_size, network=self.level2)
        ]
