import ot
import time
import torch
import torch.nn as nn

from utils.model_utils import is_conv_weight
from utils.fusion_utils import (
    get_num_layers,
    initialize_activations,
    normalize_activations,
    get_layer_weight,
    get_layer_width,
    get_preactivations_of_layer,
    compute_neuron_importance_scores
)


def get_ot_matrix(cost, mu, nu):
    """Returns the optimal transport matrix between two sets of activations.
        Parameters:
            cost: (dim1, dim2),
            mu: (dim1,),
            nu: (dim2,)

        Returns:
            T: (dim1, dim2) matrix

        Note: The cost matrix is the distance matrix between the activations
            of the two models. The alpha and beta vectors are the neuron importance
            scores.
    """
    return ot.emd(mu.cpu(), nu.cpu(), cost.cpu())


class OTFusion:
    """Fuses two models using Optimal Transport (OT) Fusion."""

    def __init__(
            self,
            models: list[nn.Module],
            target_model: nn.Module
    ):
        """Class constructor used to initialize the OTFusion object.
            Parameters:
                models: list of models to be fused,
                target_model: initial estimate of the fused model.

            Note: The models to be fused should have the same number of
                layers as the target model.
        """
        self.models, self.target_model = models, target_model
        self.device = self.target_model.device
        self.alignment_strategy = None
        self.neuron_importance_scores = None
        self.target_neuron_importance_scores = None

    def _get_ground_metric(
            self,
            current_model: nn.Module,
            fused_model_hat: nn.Module,
            layer_idx: int,
            model_k_aligned_weight: torch.Tensor
    ) -> torch.Tensor:
        """Returns the ground metric dictated by the `mode` field of the alignment strategy,
            between the current model and the current estimate of the fused model.
            Note that `model_k_aligned_weight` will not be used for activation-based alignment."""
        if self.alignment_strategy['mode'] not in ['acts', 'wts']:
            raise ValueError('Only `acts` and `wts` modes are currently supported')

        # activation-based alignment
        if self.alignment_strategy['mode'] == 'acts':
            return self._get_preactivation_ground_metric(
                current_model,
                fused_model_hat,
                layer_idx
            )
        # weight-based alignment
        else:
            return self._get_weight_ground_metric(
                model_k_aligned_weight,
                fused_model_hat,
                layer_idx
            )

    def _check_num_layers(self):
        """Returns the *common* number of layers of all models, while
            ensuring that this number is the same for all models to be fused."""
        num_layers = get_num_layers(self.models)
        target_num_layers = len(self.target_model.get_ordered_trainable_named_layers)
        if num_layers != target_num_layers:
            raise ValueError(f'The number of layers of the models to be fused is different '
                             f'compared to the target model: {num_layers} != {target_num_layers}')
        return target_num_layers

    def _prepare_acts_based_alignment(self, mode='acts'):
        """Prepares stuff for the activations-based alignment strategy."""
        num_layers = self._check_num_layers()
        dataset = self.alignment_strategy[mode]['activations_dataset']
        labels = self.alignment_strategy[mode]['activation_labels']
        method = self.alignment_strategy[mode]['neuron_importance_method']
        # ToDo: maybe used .get(..., None) for line below
        rescale_min_importance = self.alignment_strategy[mode]['rescale_min_importance'] if 'rescale_min_importance' in self.alignment_strategy[mode] else None

        # compute the neuron importance scores of the models to be aligned
        self.neuron_importance_scores = [[] for _ in range(len(self.models))]
        start_time = time.time()
        for k, model in enumerate(self.models):
            for layer_idx in range(num_layers):
                scores = compute_neuron_importance_scores(method, model, layer_idx, dataset, labels, fusion_logic='layer')
                if rescale_min_importance is not None:
                    scores = scores - torch.min(scores)
                    scores = (scores / torch.max(scores)) * (1 - rescale_min_importance) + rescale_min_importance
                self.neuron_importance_scores[k].append(scores)

        # compute the neuron importance scores of the target model
        self.target_neuron_importance_scores = []
        for layer_idx in range(num_layers):
            scores = compute_neuron_importance_scores(method, self.target_model, layer_idx, dataset, labels, fusion_logic='layer')
            if rescale_min_importance is not None:
                scores = scores - torch.min(scores)
                scores = (scores / torch.max(scores)) * (1 - rescale_min_importance) + rescale_min_importance
            self.target_neuron_importance_scores.append(scores)

        print(f'Neuron Importance Score Time: {time.time() - start_time}')
        # initialize the activations of the models to be aligned
        if mode == 'acts':
            for model in self.models:
                self._initialize_activations(model)

    def _initialize_activations(self, model):
        """Precompute the activations of the given model."""
        initialize_activations(model, self.alignment_strategy['acts']['activations_dataset'])

    @staticmethod
    def _get_preactivation_cost(act1, act2):
        """Computes the ground metric between two sets of preactivations.
            Parameters:
                act1: (batch_size, dim1),
                act2: (batch_size, dim2)

            Returns:
                C: (dim1, dim2) matrix

            Note: The ground metric is the pairwise distance between the preactivations,
                for every pair of neurons. For every neuron, we stack in a vector of
                size `batch_size` the preactivation for each data point. The Euclidean
                distance is used as the distance metric between any 2 neurons.
        """
        # contrary to what the documentation says, we have to transpose the preactivations
        return torch.cdist(act1.T.cpu(), act2.T.cpu(), p=2)

    def _get_preactivation_ground_metric(self, model, fused_model_hat, layer):
        """Computes the transportation cost between the preactivations of the two models."""
        # get the preactivations for the specified layer
        act1, act2 = get_preactivations_of_layer(model, fused_model_hat, layer)
        # normalize the activations if specified
        if self.alignment_strategy['acts']['normalize_activations']:
            act1 = normalize_activations(act1)
            act2 = normalize_activations(act2)
        # compute the ground metric matrix
        return self._get_preactivation_cost(act1, act2)

    @staticmethod
    def _get_weight_ground_metric(aligned_weight, fused, layer):
        """Computes the transportation cost between the weights of the two models.
            Parameters:
                aligned_weight: if FC: (out, in), or if conv: (out, in, spatial_dim)
                fused: fused model whose `.fc_{layer}.weight` attribute has dimension:
                    if FC: (out_fused, in), or if conv: (out, in, H, W)
                layer: int

            Returns:
                C: (dim1, dim3) matrix

            Note: The ground metric is the pairwise distance between the weights of the two models.
                In the paper, `out` is n^{l}, `in` is m^{l-1}, and `out_fused` is m^l.
        """
        fused_weight = fused.get_ordered_trainable_named_layers[layer].weight
        if len(aligned_weight) > 2:
            aligned_weight = aligned_weight.flatten(1)
            fused_weight = fused_weight.flatten(1)
        return torch.cdist(aligned_weight, fused_weight, p=2)

    def _get_input_dim(self):
        """Returns the input dimension of the target model."""
        first_layer = self.target_model.get_ordered_trainable_named_layers[0].layer
        first_weight = first_layer.weight.data
        if is_conv_weight(first_weight):
            return first_layer.in_channels
        else:
            return first_layer.in_features

    def fuse_models(self, alignment_strategy, handle_skip=False, handle_bias=True):
        """Returns the fused model using the specified alignment strategy.
            Note that `self.models` are aligned with `self.target_model`."""
        [model.eval() for model in self.models]
        self.target_model.eval()
        self.alignment_strategy = alignment_strategy

        # handle skip connections (if needed)
        skip_T, skip_T_idx = {}, {}
        residual_T, residual_T_idx = {}, {}

        # get the number of layers and the input dimension
        num_layers = self._check_num_layers()
        input_dim = self._get_input_dim()

        # initial estimate of fused model: target model
        fused_hat = self.target_model.copy_model().eval()

        # some extra steps for activation-based alignment
        if self.alignment_strategy['mode'] == 'acts' or self.alignment_strategy['wts']['neuron_importance_method'] != 'uniform':
            self._prepare_acts_based_alignment(self.alignment_strategy['mode'])

        # initial marginal of the fused model, and initial OT matrix
        nu_prev = torch.ones(input_dim, device=self.device) / input_dim
        T_prev = [torch.diag(nu_prev) for _ in range(len(self.models))]

        # variable to handle switches from conv to fc layers
        prev_is_conv = False

        # for activation-based alignment, we can compute the activations just once from now
        if alignment_strategy['mode'] == 'acts':
            self._initialize_activations(fused_hat)

        # iterate through the layers
        for layer in range(num_layers):
            print(f'Fusing layer {layer+1}/{num_layers}')

            # initialize the fused weight matrix
            total_weight, is_conv = get_layer_weight(fused_hat, layer)

            if handle_bias:
                total_bias = fused_hat.get_ordered_trainable_named_layers[layer].layer.bias.data.clone().detach()

            # compute the marginal of the fused model for the current layer
            target_model_layer_width = get_layer_width(fused_hat, layer)
            if alignment_strategy['mode'] == 'acts' or self.alignment_strategy['wts']['neuron_importance_method'] != 'uniform':
                nu = self.target_neuron_importance_scores[layer].to(self.device)
                nu = nu / nu.sum()
            else:
                nu = torch.ones(target_model_layer_width, device=self.device) / target_model_layer_width

            # iterate through the models to be aligned with the fused model
            for k, model in enumerate(self.models):

                # get the weight of the current layer of the k-th model, as well as its width and marginal mu
                model_k_weight, _ = get_layer_weight(model, layer)
                model_k_layer_width = get_layer_width(model, layer)

                if handle_bias:
                    model_k_bias = model.get_ordered_trainable_named_layers[layer].layer.bias

                if alignment_strategy['mode'] == 'acts' or self.alignment_strategy['wts']['neuron_importance_method'] != 'uniform':
                    mu = self.neuron_importance_scores[k][layer].to(self.device)
                    mu = mu / mu.sum()
                else:
                    mu = torch.ones(model_k_layer_width, device=self.device) / model_k_layer_width

                # get the corrected transportation matrix
                T_prev_corrected = T_prev[k] @ torch.diag(1/nu_prev)

                # align the current weight to the weight of the previous layer of the fused model
                if is_conv:
                    # handle skip connections
                    if handle_skip:
                        # specific to ResNet: if input_channels != output_channels
                        if model_k_weight.shape[1] != model_k_weight.shape[0]:
                            # specific to ResNet: skip connections have 1x1 kernels
                            if not (model_k_weight.shape[2] == 1):
                                skip_T[k] = T_prev_corrected.clone()
                                skip_T_idx[k] = layer
                            else:
                                residual_T[k] = T_prev_corrected.clone()
                                residual_T_idx[k] = layer
                                T_prev_corrected = skip_T[k]
                        # if input_channels == output_channels and if we have saved a previous residual
                        else:
                            # then we can handle skips by averaging the two transport matrices
                            if k in residual_T and residual_T_idx[k] == layer - 1:
                                T_prev_corrected = (T_prev_corrected + residual_T[k]) / 2

                    # for conv layers, we repeat the previous T matrix, and perform batch matrix multiplication
                    spatial_dim = model_k_weight.shape[2]
                    T_prev_corrected_expanded = T_prev_corrected.unsqueeze(0).repeat(spatial_dim, 1, 1)
                    model_k_aligned_edges = model_k_weight.permute(2, 0, 1) @ T_prev_corrected_expanded
                    model_k_aligned_edges = model_k_aligned_edges.permute(1, 2, 0)
                else:
                    # handle switches from conv to fc layers
                    if prev_is_conv:
                        # unflatten the current linear weight
                        prev_out_channels = T_prev_corrected.shape[0]
                        unflattened_model_k_weight = model_k_weight.view(model_k_layer_width, prev_out_channels, -1)
                        # get the spacial dimension of the data after the last conv layer
                        conv_output_dim = unflattened_model_k_weight.shape[-1]
                        # permute to allow for batch matrix multiplication
                        unflattened_model_k_weight = unflattened_model_k_weight.permute(2, 0, 1)

                        # repeat T matrix along the spacial dimension, and perform batch matrix multiplication
                        T_prev_corrected_expanded = T_prev_corrected.unsqueeze(0).repeat(conv_output_dim, 1, 1)
                        model_k_aligned_edges = unflattened_model_k_weight @ T_prev_corrected_expanded
                        # permute back and flatten the output to mathe the shape of linear weights
                        model_k_aligned_edges = model_k_aligned_edges.permute(1, 2, 0).flatten(1)
                    else:
                        # for fc layers, we perform a simple matrix multiplication,
                        model_k_aligned_edges = model_k_weight @ T_prev_corrected
                # calculate ground metric and then the transportation matrix
                C = self._get_ground_metric(model, fused_hat, layer, model_k_aligned_edges)
                T = get_ot_matrix(C, mu, nu).to(self.device)
                T_prev[k] = T

                # get the corrected transportation matrix
                T_corrected = T.T @ torch.diag(1/mu)

                if layer == num_layers - 1:
                    T_corrected = torch.eye(model_k_layer_width).to(self.device)

                # aligned the weight of the current model to the current layer of the fused model
                if is_conv:
                    # the spatial dimensions get flattened along with the out channels of the previous conv layer
                    reshaped_aligned_edges = model_k_aligned_edges.flatten(1)
                    model_k_aligned_weight = T_corrected @ reshaped_aligned_edges
                else:
                    # for fc layers, we perform a simple matrix multiplication
                    model_k_aligned_weight = T_corrected @ model_k_aligned_edges

                # update the fused weight matrix, and make sure dimensions match (for conv layers)
                # Note: for conv layers, this will work only when the kernel sizes are identical (for a given layer)
                model_k_aligned_weight = model_k_aligned_weight.view_as(total_weight)
                total_weight += model_k_aligned_weight

                if handle_bias:
                    model_k_aligned_bias = T_corrected @ model_k_bias
                    model_k_aligned_bias = model_k_aligned_bias.view_as(total_bias)
                    total_bias += model_k_aligned_bias.detach()

            # set fc of the fused model to be the _average_ of the aligned weight layers
            avg_weight = total_weight / (len(self.models) + 1)
            fused_hat.set_weight_at_layer(layer, avg_weight)

            if handle_bias:
                avg_bias = total_bias / (len(self.models) + 1)
                fused_hat.set_bias_at_layer(layer, avg_bias)

            # update the previous marginal
            nu_prev = nu

            # update the switch variable
            prev_is_conv = is_conv

        # remove the hooks (if any)
        if alignment_strategy['mode'] == 'acts':
            for model in self.models:
                model.remove_hooks()
            self.target_model.remove_hooks()
            fused_hat.remove_hooks()

            # we can't pickle objects with hooks -- will just copy the model to itself
            fused_hat = fused_hat.copy_model()

        return fused_hat
