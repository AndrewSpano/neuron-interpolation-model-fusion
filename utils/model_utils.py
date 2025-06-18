import copy
import torch
import torch.nn as nn


def check_if_models_have_same_architecture(model1: nn.Module, model2: nn.Module) -> bool:
    # first make sure they stem from the same class -- models can be from different classes
    #    but have the same architecture -- I don't care --> convert them yourself :)
    if not model1.__class__ == model2.__class__:
        return False

    # check every named layer with weights, to see if they have the same shape
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2 or param1.shape != param2.shape:
            return False
    return True


def module_is_trainable(module: nn.Module) -> bool:
    return len(list(module.parameters())) > 0


def is_conv_weight(weight: torch.Tensor) -> bool:
    return len(weight.shape) == 4


def is_transformer_activation(activation: torch.Tensor) -> bool:
    return len(activation.shape) == 3


def is_conv_activation(activation: torch.Tensor) -> bool:
    return len(activation.shape) == 4


def reset_parameters(net: nn.Module) -> None:
    for layer in net.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def soft_reset_parameters(net: nn.Module, eps: float) -> None:
    # save current parameters
    current_params = {name: param.data.clone() for name, param in net.named_parameters()}

    # make a copy of the model and reset its parameters
    net_copy = copy.deepcopy(net)
    for layer in net_copy.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    # interpolate between current and reset parameters
    with torch.no_grad():
        for name, param in net.named_parameters():
            if name in current_params:
                reset_param = dict(net_copy.named_parameters())[name].data
                param.data.mul_(1 - eps).add_(eps * reset_param)


def disable_dropout(net: nn.Module) -> None:
    """Disable dropout layers in the model."""
    for module in net.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.eval()
