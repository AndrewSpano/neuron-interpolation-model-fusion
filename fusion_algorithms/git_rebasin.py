import torch

from scipy.optimize import linear_sum_assignment

from utils.model_utils import is_conv_weight
from utils.fusion_utils import (
    get_num_layers,
    initialize_activations,
    get_layer_weight,
    compute_neuron_importance_scores
)


class GitRebasin:

    neuron_importance_methods = [
        'uniform',
        'conductance',
        'deeplift',
        'featureablation'
    ]

    def __init__(self, model0, model1):
        self.model0 = model0
        self.model1 = model1
        self.neuron_importance_scores0 = None
        self.neuron_importance_scores1 = None

    def _prepare_acts_based_alignment(self, dataset, labels, method):
        """Prepares stuff for the activations-based alignment strategy."""
        num_layers = get_num_layers([self.model0, self.model1])

        # neuron importance scores
        self.neuron_importance_scores0 = []
        self.neuron_importance_scores1 = []
        for layer_idx in range(num_layers):
            # model 1
            scores1 = compute_neuron_importance_scores(method, self.model0, layer_idx, dataset, labels, fusion_logic='layer')
            self.neuron_importance_scores0.append(scores1)
            # model 2
            scores2 = compute_neuron_importance_scores(method, self.model1, layer_idx, dataset, labels, fusion_logic='layer')
            self.neuron_importance_scores1.append(scores2)

        # initialize the activations of the model to be aligned and the target model to be aligned to
        initialize_activations(self.model0, dataset)
        initialize_activations(self.model1, dataset)

    @staticmethod
    def _get_cost_matrix(acts0, acts1, s0, s1):
        """Get the cost matrix for the given layer."""
        cost_matrix = torch.zeros((acts0.shape[1], acts1.shape[1]))
        for i in range(acts0.shape[1]):
            for j in range(acts1.shape[1]):
                avg_acts = (acts0[:,i] * s0[i] + acts1[:,j] * s1[j]) / (s0[i] + s1[j])
                cost0 = s0[i] * torch.norm(avg_acts - acts0[:, i], p=2)
                cost1 = s1[j] * torch.norm(avg_acts - acts1[:, j], p=2)
                cost_matrix[i][j] = cost0 + cost1
        return cost_matrix.cpu()

    @staticmethod
    def _get_cost_matrix_vectorized(acts0, acts1, s0, s1):
        """Get the cost matrix for the given layer in a vectorized fashion."""
        # expand dimensions to prepare for broadcasting
        acts0_exp = acts0.unsqueeze(2)  # shape: (N, m, 1)
        acts1_exp = acts1.unsqueeze(1)  # shape: (N, 1, n)
        s0_exp = s0.unsqueeze(1)  # shape: (m, 1)
        s1_exp = s1.unsqueeze(0)  # shape: (1, n)

        # calculate the weighted average activations
        avg_acts = (acts0_exp * s0_exp + acts1_exp * s1_exp) / (s0_exp + s1_exp)  # shape: (N, m, n)

        # compute costs
        cost0 = s0_exp * torch.norm(avg_acts - acts0_exp, dim=0)  # shape: (m, n)
        cost1 = s1_exp * torch.norm(avg_acts - acts1_exp, dim=0)  # shape: (m, n)

        # combine costs to form the cost matrix
        cost_matrix = cost0 + cost1  # shape: (m, n)
        return cost_matrix.cpu()

    def _fuse_layer_two(self, layer, fused_hat):
        """Fuse the given layer using the Hungarian algorithm."""
        # get the width of the layer (number of neurons)
        num_layers = get_num_layers([self.model0, self.model1])

        # check whether the layer is a convolutional layer
        is_conv = is_conv_weight(fused_hat.get_ordered_trainable_named_layers[layer].weight)

        # neuron importance scores
        s0 = self.neuron_importance_scores0[layer]
        s1 = self.neuron_importance_scores1[layer]

        # get the preactivations of the current layer for both models
        acts0 = self.model0.get_inputs_and_preactivations(layer)[1]
        acts1 = self.model1.get_inputs_and_preactivations(layer)[1]

        # turn preactivations into activations for Git Re-basin; for our experiments, we only use relu
        acts0 = acts0 * (acts0 > 0)
        acts1 = acts1 * (acts1 > 0)
        if is_conv:
            acts0 = acts0.permute(0, 2, 3, 1).reshape(acts0.shape[0] * acts0.shape[2] * acts0.shape[3], -1)
            acts1 = acts1.permute(0, 2, 3, 1).reshape(acts1.shape[0] * acts1.shape[2] * acts1.shape[3], -1)

        acts0 = acts0.to(self.model0.device)
        acts1 = acts1.to(self.model1.device)
        s0 = s0.to(self.model0.device)
        s1 = s1.to(self.model1.device)

        # calculate the cost matrix vectorized
        cost_matrix = self._get_cost_matrix_vectorized(acts0, acts1, s0, s1)

        # solve the assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # for the last layer, fix the permutation_matrix to be identity
        if layer == num_layers - 1:
            col_indices = row_indices

        # create the permutation matrix and return it
        permutation_matrix = torch.zeros_like(cost_matrix)
        for i, j in zip(row_indices, col_indices):
            permutation_matrix[i][j] = 1

        return permutation_matrix

    def fuse(self, activations_dataset, activations_labels, neuron_importance_method, handle_skip=False):
        """Fuse the models using the Hungarian algorithm."""
        if neuron_importance_method not in self.neuron_importance_methods:
            raise ValueError(f'Invalid neuron importance method: {neuron_importance_method}')
        num_layers = get_num_layers([self.model0, self.model1])

        # handle skip connections (if needed)
        skip_permutation_matrix = None
        residual_permutation_matrix, residual_permutation_matrix_idx = None, None

        # prepare the activations-based alignment (neuron importance scores + preactivations)
        self._prepare_acts_based_alignment(activations_dataset, activations_labels, neuron_importance_method)

        # initialize the fused model to be the same as model 0
        fused_hat = self.model0.copy_model().eval()

        # perform the fusion layer by layer
        prev_is_conv = False
        prev_permutation_matrix = None
        for layer in range(num_layers):
            print(f'Fusing layer {layer + 1}/{num_layers}')

            # get the weights and biases of the current layer
            w0, is_conv = get_layer_weight(self.model0, layer)
            w1, is_conv = get_layer_weight(self.model1, layer)
            if is_conv:
                w0 = w0.flatten(1)
            b0 = self.model0.get_ordered_trainable_named_layers[layer].layer.bias
            b1 = self.model1.get_ordered_trainable_named_layers[layer].layer.bias

            # get the neuron importance scores of the current layer
            s0 = self.neuron_importance_scores0[layer].to(activations_dataset.device)
            s1 = self.neuron_importance_scores1[layer].to(activations_dataset.device)

            # get the permutation matrix for the current layer
            permutation_matrix = self._fuse_layer_two(layer, fused_hat).to(activations_dataset.device)

            # align the weights and biases of the current layer
            if layer == 0:
                if is_conv:
                    aligned_w1 = permutation_matrix @ w1.flatten(1)
                else:
                    aligned_w1 = permutation_matrix @ w1
            else:
                # handle conv layers
                if is_conv:
                    # handle skip connections
                    if handle_skip:
                        # specific to ResNet: if input_channels != output_channels
                        if w1.shape[1] != w1.shape[0]:
                            # specific to ResNet: skip connections have 1x1 kernels
                            if not (w1.shape[2] == 1):
                                skip_permutation_matrix = prev_permutation_matrix.clone()
                            else:
                                residual_permutation_matrix = prev_permutation_matrix.clone()
                                residual_permutation_matrix_idx = layer
                                prev_permutation_matrix = skip_permutation_matrix
                        # if input_channels == output_channels and if we have saved a previous residual
                        else:
                            # then we can handle skips by averaging the two transport matrices
                            if residual_permutation_matrix is not None and residual_permutation_matrix_idx == layer - 1:
                                prev_permutation_matrix = (prev_permutation_matrix + residual_permutation_matrix) / 2

                    # handle repeats for conv layers
                    prev_permute_w1 = torch.cat(w1.permute(0, 2, 1).unbind()) @ prev_permutation_matrix.T
                    prev_permute_w1 = torch.stack(prev_permute_w1.split(w1.shape[2]), dim=0).permute(0, 2, 1)
                    aligned_w1 = permutation_matrix @ prev_permute_w1.flatten(1)
                else:
                    # handle switches from conv to fc layers
                    if prev_is_conv:
                        reshaped_w1 = w1.reshape(w1.shape[0], prev_permutation_matrix.shape[0], -1)
                        prev_permute_w1 = torch.cat(reshaped_w1.permute(0, 2, 1).unbind()) @ prev_permutation_matrix.T
                        prev_permute_w1 = torch.stack(prev_permute_w1.split(reshaped_w1.shape[2]), dim=0).permute(0, 2, 1)
                        aligned_w1 = permutation_matrix @ prev_permute_w1.flatten(1)
                    # handle fc layers
                    else:
                        aligned_w1 = permutation_matrix @ w1 @ prev_permutation_matrix.T

            # align the neuron importance scores of the current layer
            s1 = permutation_matrix @ s1

            # fuse the weights and biases of the current layer
            new_W = (w0 * (s0  / (s0 + s1)).unsqueeze(1) + aligned_w1 * (s1  / (s0 + s1)).unsqueeze(1))

            # fuse biases if they exist
            new_b = None
            if b0 is not None:
                aligned_b1 = permutation_matrix @ b1
                new_b = (b0 * (s0  / (s0 + s1)) + aligned_b1 * (s1  / (s0 + s1)))

            # set the new weights
            fused_hat.set_weight_at_layer(layer, new_W)
            if new_b is not None:
                fused_hat.set_bias_at_layer(layer, new_b)

            # update the flag and previous permutation matrix
            prev_is_conv = is_conv
            prev_permutation_matrix = permutation_matrix

        # remove the hooks
        self.model0.remove_hooks()
        self.model1.remove_hooks()
        fused_hat.remove_hooks()

        # copy the model to itself because saving models if previous hooks is buggy
        fused_hat = fused_hat.copy_model()
        return fused_hat
