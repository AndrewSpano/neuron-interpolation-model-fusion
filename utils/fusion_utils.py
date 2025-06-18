import torch

from utils.model_utils import is_conv_weight
from utils.neuron_importance import Conductance, DeepLIFT, FeatureAblation


def get_num_layers(models):
    """Returns the *common* number of layers of all models, while
        ensuring that this number is the same for all models to be fused."""
    model0_num_layers = len(models[0].get_ordered_trainable_named_layers)
    for i, m in enumerate(models[1:]):
        if model0_num_layers != len(m.get_ordered_trainable_named_layers):
            raise ValueError(f'The number of layers of model 1 is different compared to model {i + 1}:'
                             f' {model0_num_layers} != {len(m.get_ordered_trainable_named_layers)}')
    return model0_num_layers


def get_num_levels(models):
    """Returns the *common* number of levels of all models, while
        ensuring that this number is the same for all models to be fused."""
    model0_num_levels = len(models[0].get_ordered_trainable_named_levels)
    for i, m in enumerate(models[1:]):
        if model0_num_levels != len(m.get_ordered_trainable_named_levels):
            raise ValueError(f'The number of levels of model 1 is different compared to model {i + 1}:'
                             f' {model0_num_levels} != {len(m.get_ordered_trainable_named_levels)}')
    return model0_num_levels


def initialize_activations(model, activation_dataset):
    """Precompute the activations of the given model."""
    model.remove_hooks()
    model.register_hooks()
    with torch.no_grad():
        x = activation_dataset
        model(x)


def normalize_activations(activations):
    """Normalize the activations to have unit L2 norm."""
    return activations / torch.norm(activations, dim=0, keepdim=True)


def normalize_activations_per_sample(activations):
    """Normalize the activations to have unit L2 norm per sample."""
    return activations / torch.norm(activations, dim=1, keepdim=True)


def get_layer_weight(model, layer):
    """Returns the weight of the specified layer of the given model.
        Note: If the weight is a convolutional weight, it is reshaped to a 2D matrix as follows:
            (out_channels, in_channels, kernel_height, kernel_width) ->
                (out_channels, in_channels, kernel_height * kernel_width)"""
    weight = model.get_ordered_trainable_named_layers[layer].weight.data.detach().clone()
    is_conv = is_conv_weight(weight)
    if is_conv:
        weight = weight.flatten(2)
    return weight, is_conv


def get_layer_width(model, layer_idx):
    """Returns the width of the specified layer of the given model.
        Note: For FC it's out_features, for conv it's out_channels, and in
            both cases it's the first dimensions of the weight vector."""
    weight = model.get_ordered_trainable_named_layers[layer_idx].weight.data
    return weight.shape[0]


def get_preactivations_of_layer(model, target_model, layer):
    """Returns the preactivations of the specified layer for the given model and target model."""
    # get the precomputed preactivations of model1
    _, act_model = model.get_inputs_and_preactivations(layer_idx=layer)

    # get the preactivations of model2
    _, act_target = target_model.get_inputs_and_preactivations(layer_idx=layer)

    # flatten the preactivations (needed for conv layers)
    is_conv_act = len(act_model.shape) == 4
    if is_conv_act:
        act_model = act_model.permute(1, 0, 2, 3).flatten(1).T
        act_target = act_target.permute(1, 0, 2, 3).flatten(1).T

    return act_model, act_target


def compute_neuron_importance_scores(method, model, idx, X, y, fusion_logic, batch_size=50):
    """Compute the neuron importance scores for the given layer of the specified model,
        using the specified neuron importance method."""
    assert fusion_logic in ['layer', 'level'], f'Invalid fusion logic: {fusion_logic}'

    # get the layer or level
    if fusion_logic == 'layer':
        layer = model.get_ordered_trainable_named_layers[idx].layer
        width = get_layer_width(model, idx)
    else:
        level = model.get_ordered_trainable_named_levels[idx]
        layer = level.network
        width = level.output_width

    # get the scores
    if method == 'uniform':
        scores = torch.ones(width)
    elif method == 'conductance':
        scores = Conductance.get_score(model, layer, width, X, y, batch_size=batch_size)
    elif method == 'deeplift':
        scores = DeepLIFT.get_score(model, layer, width, X, y, batch_size=batch_size)
    else:
        scores = FeatureAblation.get_score(model, layer, width, X, y, batch_size=batch_size)

    # convert to cpu, clean up
    scores_cpu = scores.cpu()
    return scores_cpu
