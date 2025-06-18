import torch
import torch.nn as nn

from typing import NamedTuple
from abc import ABC, abstractmethod

from utils.model_utils import is_conv_weight, module_is_trainable, reset_parameters


class WeightRepresentation(NamedTuple):
    """Representation of a trainable layer in a model."""
    name: str
    weight: torch.Tensor
    layer: nn.Module


class LevelRepresentation(NamedTuple):
    """Representation of a collections of layers in a model."""
    name: str
    output_width: int
    network: nn.Module


class BaseModel(nn.Module, ABC):
    """Base class for all models."""

    def __init__(self, train_indices: list[int] | None, val_indices: list[int] | None):
        """Initialize the model."""
        super(BaseModel, self).__init__()

        # dictionaries to map layer names to their inputs and outputs
        self.layer_inputs = {}
        self.layer_outputs = {}

        # dictionary to store the handles of the hooks
        self.handles = {}

        # store the train/val indices, so that we know which data was used to train/eval the model
        self.train_indices = train_indices
        self.val_indices = val_indices

    @property
    @abstractmethod
    def get_ordered_trainable_named_layers(self) -> list[WeightRepresentation]:
        """Returns the trainable layers of the model in the order they are applied."""
        pass

    @abstractmethod
    def forward_until_level(self, x: torch.Tensor, level_idx: int = -1) -> torch.Tensor:
        """Forwards the input tensor until the specified level. If level_idx is -1,
            it will forward the input throught the whole model."""
        pass

    @property
    @abstractmethod
    def get_ordered_trainable_named_levels(self) -> list[LevelRepresentation]:
        """Returns the trainable levels of the model in the order they are applied."""
        pass

    @abstractmethod
    def copy_model(self) -> 'BaseModel':
        """Creates a copy of the model with the same architecture,
            and the same weights as the current model."""
        pass

    def set_weight_at_layer(self, trainable_layer_idx, new_weight):
        """Set the weight of the specified layer to the new weight."""
        layer = self.get_ordered_trainable_named_layers[trainable_layer_idx].layer
        current_weight = layer.weight.data
        if is_conv_weight(current_weight):
            new_weight = new_weight.view_as(current_weight)
        with torch.no_grad():
            layer.weight.data = new_weight.to(self.device)

    def set_bias_at_layer(self, trainable_layer_idx, new_bias):
        """Set the bias of the specified layer to the new bias."""
        layer = self.get_ordered_trainable_named_layers[trainable_layer_idx].layer
        with torch.no_grad():
            layer.bias.data = new_bias.to(self.device)

    def set_unique_trainable_level(self, level_idx: int) -> None:
        """Make only the level layers trainable."""
        num_levels = len(self.get_ordered_trainable_named_levels)
        for i in range(num_levels):
            level = self.get_ordered_trainable_named_levels[i]
            is_right_level = (i == level_idx)
            for layer in level.trainable_layers:
                if module_is_trainable(layer):
                    for param in layer.parameters():
                        param.requires_grad = is_right_level

    def reset_parameters_of_level(self, level_idx: int) -> None:
        """Reset the parameters of the specified level."""
        level = self.get_ordered_trainable_named_levels[level_idx]
        for layer in level.trainable_layers:
            if module_is_trainable(layer):
                reset_parameters(layer)

    @property
    def device(self):
        return next(self.parameters()).device

    def register_hooks(self):
        """Register hooks for the model."""
        module_to_name = {
            weight_repr.layer: weight_repr.name
            for weight_repr in self.get_ordered_trainable_named_layers
        }

        def layer_hook(module, layer_in, layer_out):
            """Hook to store the input/output to a given layer."""
            module_name = module_to_name[module]
            self.layer_inputs[module_name] = layer_in[0].detach().clone()
            self.layer_outputs[module_name] = layer_out.detach().clone()

        # register input/output hooks
        for layer in self.get_ordered_trainable_named_layers:
            self.handles[layer.name] = layer.layer.register_forward_hook(layer_hook)

    def remove_hooks(self):
        """Remove all hooks from the model."""
        for handle in self.handles.values():
            handle.remove()
        self.handles.clear()
        self.layer_inputs.clear()
        self.layer_outputs.clear()

    def get_inputs_and_preactivations(self, layer_idx):
        """Get the inputs and preactivations of the specified layer."""
        layer_name = self.get_ordered_trainable_named_layers[layer_idx].name
        preactivations = self.layer_outputs[layer_name]
        inputs = self.layer_inputs[layer_name]
        return inputs, preactivations
