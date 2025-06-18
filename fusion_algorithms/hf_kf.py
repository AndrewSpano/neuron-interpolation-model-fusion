import time

import torch
import numpy as np
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from utils.model_utils import is_conv_weight
from utils.fusion_utils import (
    get_num_layers,
    initialize_activations,
    normalize_activations,
    get_layer_weight,
    get_layer_width,
    compute_neuron_importance_scores
)

COST_MATRIX_VECTORIZED_CHUNK_SIZE = 8

class LSFusion:
    fusion_methods = [
        'hungarian',
        'k_means'
    ]

    neuron_importance_methods = [
        'uniform',
        'conductance',
        'deeplift',
        'featureablation'
    ]

    # ToDo:
    #   1. Make `COST_MATRIX_VECTORIZED_CHUNK_SIZE` a class parameter
    #   2. Add an argument to measure execution times
    #   3. Improve KF fusion code (shorter functions, clearer code)
    #   4. Implement the feature ablation method
    def __init__(self, models):
        self.models = models
        self.neuron_importance_scores = None

    @property
    def device(self):
        """Return the device of the first model."""
        return self.models[0].device

    def _config_check(self, fusion_method, neuron_importance_method):
        """Check the configuration of the fusion method and neuron importance method."""
        if fusion_method not in self.fusion_methods:
            raise ValueError(f'Invalid fusion importance method: {fusion_method}')
        if neuron_importance_method not in self.neuron_importance_methods:
            raise ValueError(f'Invalid neuron importance method: {neuron_importance_method}')

    @staticmethod
    def _get_cost_matrix(X, acts0, acts1, w0, w1, s0, s1):
        """Get the cost matrix for the given layer."""
        cost_matrix = np.zeros((w0.shape[1], w1.shape[1]))
        weighted_acts0s = [X @ w0[:, i] * s0[i] for i in range(w0.shape)]
        weighted_acts1s = [X @ w1[:, i] * s1[i] for i in range(w1.shape)]
        for i in range(w0.shape[1]):
            for j in range(w1.shape[1]):
                act_pred = (weighted_acts0s[i] + weighted_acts1s[j]) / (s0[i] + s1[j])
                cost0 = s0[i] * torch.norm(act_pred - acts0[:, i], dim=0)
                cost1 = s1[j] * torch.norm(act_pred - acts1[:, j], dim=0)
                cost_matrix[i][j] = (cost0 + cost1).item()
        return cost_matrix

    @staticmethod
    def _get_cost_matrix_vectorized(X, acts0, acts1, w0, w1, s0, s1):
        """Get the cost matrix for the given layer."""
        # Reshape scales for broadcasting
        s0 = s0.view(-1, 1)  # shape (n0, 1)
        s1 = s1.view(1, -1)  # shape (1, n1)

        # compute the weighted combination of weights for all (i, j) pairs
        total_importance = s0 + s1  # shape (n0, n1)
        w_combined = (s0 * w0.unsqueeze(2) + s1 * w1.unsqueeze(1)) / total_importance  # shape (d, n0, n1)

        # predicted activations for all (i, j) pairs
        act_pred = torch.matmul(X, w_combined.view(X.shape[1], -1))  # shape (m, n0 * n1)
        act_pred = act_pred.view(X.shape[0], w0.shape[1], w1.shape[1])  # shape (m, n0, n1)

        # compute costs for acts0 and acts1
        cost0 = s0 * torch.norm(act_pred - acts0.unsqueeze(2), dim=0)  # shape (n0, n1)
        cost1 = s1 * torch.norm(act_pred - acts1.unsqueeze(1), dim=0)  # shape (n0, n1)

        # combine costs and return as a numpy array
        cost_matrix = cost0 + cost1  # shape (n0, n1)
        return cost_matrix.cpu().numpy()

    @staticmethod
    def _get_cost_matrix_vectorized_chunked(X, acts0, acts1, w0, w1, s0, s1, chunk_size=COST_MATRIX_VECTORIZED_CHUNK_SIZE):
        """Chunked version to save GPU memory."""
        device = X.device
        n0, n1 = w0.shape[1], w1.shape[1]
        cost_matrix = torch.zeros(n0, n1, device=device)

        for i_start in range(0, n0, chunk_size):
            for j_start in range(0, n1, chunk_size):
                # determine chunk ranges
                i_end = min(i_start + chunk_size, n0)
                j_end = min(j_start + chunk_size, n1)

                # compute cost for this chunk
                s0_chunk = s0[i_start:i_end].unsqueeze(1)  # shape (chunk, 1)
                s1_chunk = s1[j_start:j_end].unsqueeze(0)  # shape (1, chunk)

                total_importance = s0_chunk + s1_chunk
                w_combined = (s0_chunk * w0[:, i_start:i_end].unsqueeze(2) +
                              s1_chunk * w1[:, j_start:j_end].unsqueeze(1)) / total_importance

                act_pred = torch.matmul(X, w_combined.reshape(X.shape[1], -1))
                act_pred = act_pred.view(X.shape[0], i_end - i_start, j_end - j_start)

                cost0 = s0_chunk * torch.norm(act_pred - acts0[:, i_start:i_end].unsqueeze(2), dim=0)
                cost1 = s1_chunk * torch.norm(act_pred - acts1[:, j_start:j_end].unsqueeze(1), dim=0)

                cost_matrix[i_start:i_end, j_start:j_end] = cost0 + cost1

                del act_pred, w_combined, s0_chunk, s1_chunk, total_importance
                torch.cuda.empty_cache()

        return cost_matrix.cpu().numpy()

    @staticmethod
    def _get_cost_matrix_vectorized_chunked_v2(X, acts0, acts1, w0, w1, s0, s1, chunk_size=COST_MATRIX_VECTORIZED_CHUNK_SIZE):
        """Chunked version to save GPU memory."""
        device = X.device
        n0, n1 = w0.shape[1], w1.shape[1]
        cost_matrix = torch.zeros(n0, n1, device=device)

        act0s = []
        s0_chunks = []
        starts_0 = []
        ends_0 = []
        for i_start in range(0, n0, chunk_size):
            i_end = min(i_start + chunk_size, n0)
            s0_chunk = s0[i_start:i_end].unsqueeze(1)  # shape (chunk, 1)
            w0weighted = torch.cat([s0_chunk * w0[:, i_start:i_end].unsqueeze(2)] * (i_end-i_start), dim=-1)
            act0s.append(torch.matmul(X, w0weighted.reshape(X.shape[1], -1)))
            s0_chunks.append(s0_chunk)
            starts_0.append(i_start)
            ends_0.append(i_end)

        act1s = []
        s1_chunks = []
        starts_1 = []
        ends_1 = []
        for j_start in range(0, n1, chunk_size):
            j_end = min(j_start + chunk_size, n1)
            s1_chunk = s1[j_start:j_end].unsqueeze(0)  # shape (1, chunk)
            w1weighted = torch.cat([s1_chunk * w1[:, j_start:j_end].unsqueeze(1)] * (j_end-j_start), dim=-1)
            act1s.append(torch.matmul(X, w1weighted.reshape(X.shape[1], -1)))
            s1_chunks.append(s1_chunk)
            starts_1.append(j_start)
            ends_1.append(j_end)

        for i in range(len(s0_chunks)):
            for j in range(len(s1_chunks)):

                # compute cost for this chunk
                s0_chunk = s0_chunks[i]  # shape (chunk, 1)
                s1_chunk = s1_chunks[j]  # shape (1, chunk)

                i_start = starts_0[i]
                i_end = ends_0[i]

                j_start = starts_1[j]
                j_end = ends_1[j]

                total_importance = s0_chunk + s1_chunk
                act_pred = (act0s[i] + act1s[j]) / total_importance.flatten()
                act_pred = act_pred.view(X.shape[0], i_end - i_start, j_end - j_start)

                cost0 = s0_chunk * torch.norm(act_pred - acts0[:, i_start:i_end].unsqueeze(2), dim=0)
                cost1 = s1_chunk * torch.norm(act_pred - acts1[:, j_start:j_end].unsqueeze(1), dim=0)

                cost_matrix[i_start:i_end, j_start:j_end] = cost0 + cost1

                del act_pred, total_importance
                torch.cuda.empty_cache()

        return cost_matrix.cpu().numpy()

    def _fuse_layer_hungarian(self, layer_idx, fused_hat, norm_acts):
        """Fuse the given layer using the Hungarian algorithm."""
        # get the width of the layer (number of neurons)
        layer_width = get_layer_width(fused_hat, layer_idx)

        # check whether the layer is a convolutional layer
        is_conv = is_conv_weight(fused_hat.get_ordered_trainable_named_layers[layer_idx].weight)
        use_bias = fused_hat.get_ordered_trainable_named_layers[layer_idx].layer.bias is not None

        # get the post-activation of the previous layer (input to the current layer), and add the bias term
        X = fused_hat.get_inputs_and_preactivations(layer_idx)[0]

        # for conv layers, reshape the input into a matrix with unfold
        in_filters = None
        if is_conv:
            conv_layer = fused_hat.get_ordered_trainable_named_layers[layer_idx].layer
            in_filters = conv_layer.in_channels
            X = F.unfold(X, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding)
            X = X.permute(0, 2, 1).reshape(-1, conv_layer.in_channels * np.prod(conv_layer.kernel_size))

        # add the bias term if specified
        if use_bias:
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)

        # get the pseudo-inverse to later solve the LS problem
        start = time.time()
        # pinv = torch.from_numpy(np.linalg.pinv(X.cpu().numpy())).to(X.device)
        pinv = torch.linalg.pinv(X)
        print(f'Pseudo-inverse time: {time.time() - start}')

        # neuron importance scores
        s0 = self.neuron_importance_scores[0][layer_idx].clone().to(X.device)
        s1 = self.neuron_importance_scores[1][layer_idx].clone().to(X.device)

        # get the preactivations of the current layer for both models -- reshape for conv layers
        acts0 = self.models[0].get_inputs_and_preactivations(layer_idx)[1]
        acts1 = self.models[1].get_inputs_and_preactivations(layer_idx)[1]
        if is_conv:
            acts0 = acts0.permute(0, 2, 3, 1).reshape(X.shape[0], -1)
            acts1 = acts1.permute(0, 2, 3, 1).reshape(X.shape[0], -1)

        # normalize the activations
        if norm_acts:
            acts0 = normalize_activations(acts0)
            acts1 = normalize_activations(acts1)

        # compute the optimal LS solution for each neuron using broadcast
        w0 = pinv @ acts0
        w1 = pinv @ acts1

        # calculate the cost matrix
        # X = X.double()
        # acts0 = acts0.double()
        # acts1 = acts1.double()
        # w0 = w0.double()
        # w1 = w1.double()
        # s0 = s0.double()
        # s1 = s1.double()

        start = time.time()
        cost_matrix = self._get_cost_matrix_vectorized_chunked(X, acts0, acts1, w0, w1, s0, s1)
        print(f'Cost matrix time: {time.time() - start}')
        print()

        # solve the assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # For the last layer, fix the assignment to be identity
        if layer_idx == len(fused_hat.get_ordered_trainable_named_layers) - 1:
            col_indices = row_indices

        # create the new neurons
        current_weight, _ = get_layer_weight(fused_hat, layer_idx)
        new_weight = torch.zeros_like(current_weight)
        new_bias = torch.zeros(layer_width)
        for i, j in zip(row_indices, col_indices):
            # new weight: convex combination of the two weights from the Hungarian assignment
            _w0, _w1 = w0[:, i], w1[:, j]
            W = (s0[i] * _w0 + s1[j] * _w1) / (s0[i] + s1[j])

            # split weight and bias (if bias is used)
            if use_bias:
                new_bias[i] = W[-1]
                W = W[:-1]

            # reshape the weight for conv layers
            if is_conv:
                new_weight[i] = W.reshape(in_filters, -1)
            else:
                new_weight[i] = W

        # remove bias if not used
        if not use_bias:
            new_bias = None

        return new_weight, new_bias

    @staticmethod
    def _calculate_centroids(output_size, assignment, X, w, s):
        """Calculate the centroids of the assigned neurons for the k-means variant of the fusion procedure."""
        # get the assigned neurons for each centroid
        all_assigned_neurons = [[] for _ in range(output_size)]
        for i, x in enumerate(assignment):
            all_assigned_neurons[x].append(i)

        # compute the new centroids
        centroid_ws = []
        centroid_preactivations = []
        for i, assigned_neurons in enumerate(all_assigned_neurons):
            if len(assigned_neurons) == 0:
                continue

            # calculate the centroid of the assigned neurons for the current cluster
            centroid_w = (s[assigned_neurons] * w[:,assigned_neurons]).sum(dim=1)
            total_importance = s[assigned_neurons].sum()

            # normalize
            centroid_w /= total_importance
            centroid_preactivation = X @ centroid_w

            # save them
            centroid_ws.append(centroid_w)
            centroid_preactivations.append(centroid_preactivation)

        # convert the preactivations to a tensor and return
        centroid_preactivations = torch.stack(centroid_preactivations)
        return centroid_ws, centroid_preactivations

    @staticmethod
    def _calculate_centroids_v2(output_size, assignment, pinv, acts, s):
        """Calculate the centroids of the assigned neurons for the k-means variant of the fusion procedure."""
        # get the assigned neurons for each centroid
        all_assigned_neurons = [[] for _ in range(output_size)]
        for i, x in enumerate(assignment):
            all_assigned_neurons[x].append(i)

        # compute the new centroids
        centroid_ws = []
        centroid_preactivations = []
        for i, assigned_neurons in enumerate(all_assigned_neurons):
            if len(assigned_neurons) == 0:
                continue

            # calculate the centroid of the assigned neurons for the current cluster
            centroid_preactivation = (s[assigned_neurons] * acts[:, assigned_neurons]).sum(dim=1)
            total_importance = s[assigned_neurons].sum()

            # normalize
            centroid_preactivation /= total_importance
            centroid_w = pinv @ centroid_preactivation

            # save them
            centroid_ws.append(centroid_w)
            centroid_preactivations.append(centroid_preactivation)

        # convert the preactivations to a tensor and return
        centroid_preactivations = torch.stack(centroid_preactivations)
        return centroid_ws, centroid_preactivations

    def _fuse_layer_k_means(self, layer, fused_hat, norm_acts, num_k_means_tries=10, k_means_early_stop_minimum_improvement=0.05, k_means_early_stop_steps_check=3):
        """Fuse the given layer using the Hungarian algorithm."""
        # get the width of the layer (number of neurons)
        layer_width = get_layer_width(fused_hat, layer)

        # check whether the layer is a convolutional layer
        is_conv = is_conv_weight(fused_hat.get_ordered_trainable_named_layers[layer].weight)
        use_bias = fused_hat.get_ordered_trainable_named_layers[layer].layer.bias is not None
        is_last_layer = (layer == len(fused_hat.get_ordered_trainable_named_layers) - 1)

        # get the post-activation of the previous layer (input to the current layer), and add the bias term
        X = fused_hat.get_inputs_and_preactivations(layer)[0]

        # get the shape of the layer and the output size
        layer_shape = fused_hat.get_ordered_trainable_named_layers[layer].weight.shape
        output_size = layer_shape[0]

        # for conv layers, reshape the input into a matrix with unfold
        in_filters = None
        if is_conv:
            conv_layer = fused_hat.get_ordered_trainable_named_layers[layer].layer
            in_filters = conv_layer.in_channels
            X = F.unfold(X, kernel_size=conv_layer.kernel_size, stride=conv_layer.stride, padding=conv_layer.padding)
            X = X.permute(0, 2, 1).reshape(-1, in_filters * np.prod(conv_layer.kernel_size))

        # add the bias term if specified
        if use_bias:
            X = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)

        # get the pseudo-inverse to later solve the LS problem
        start = time.time()
        # pinv = torch.from_numpy(np.linalg.pinv(X.cpu().numpy())).to(X.device)
        pinv = torch.linalg.pinv(X)
        print(f'Pseudo-inverse time: {time.time() - start}')

        # neuron importance scores
        s = [self.neuron_importance_scores[model_idx][layer].clone().to(X.device) for model_idx in range(len(self.models))]

        # get the preactivations of the current layer for all models -- and adjust for conv layers

        # place activations and neuron importance scores in a single tensor
        acts = torch.cat(tuple([m.get_inputs_and_preactivations(layer)[1] for m in self.models]), dim=1)
        if is_conv:
            acts = acts.permute(0, 2, 3, 1).reshape(X.shape[0], -1)

        s = torch.cat(s, dim=0)

        # normalize the activations if specified
        if norm_acts:
            acts = normalize_activations(acts)

        # compute the optimal weight for each preactivation
        w = pinv @ acts

        # create the new neurons
        current_weight, _ = get_layer_weight(fused_hat, layer)
        new_weight = torch.zeros_like(current_weight)
        new_bias = torch.zeros(layer_width)

        # different handling for the last layer
        if is_last_layer:
            best_assignment = torch.cat([torch.arange(start=0, end=output_size) for _ in range(len(self.models))])
        else:
            # store the best assignment
            best_assignment = torch.cat([torch.arange(start=0, end=output_size) for _ in range(len(self.models))])
            best_total_cost = np.inf

            # try multiple times to find the best assignment
            start = time.time()
            time_centroid_initialization = 0
            iterations_centroid_initialization = 0
            time_centroid_covergence = 0
            iterations_centroid_covergence = 0
            for _ in range(num_k_means_tries):
                # initialize the assignment
                prev_assignment = torch.ones(w.shape[1]) * (-1)

                # initialize in the style of k-means++
                start_time_ci = time.time()
                initial_centroids = [int(np.random.randint(low=0, high=output_size))]
                initial_centroid_preactivations = [X @ w[:, initial_centroids[0]]]
                while len(initial_centroids) < output_size:
                    iterations_centroid_initialization += 1
                    # calculate the cost of each neuron to the nearest centroid
                    dist_matrix = torch.cdist(torch.stack(initial_centroid_preactivations), acts.T, p=2)
                    dist_to_closest_centroid, min_dist_centroid_idx = torch.min(dist_matrix, dim=0)

                    # choose the neuron with the maximum distance to the nearest centroid
                    max_dist_neuron_idx = torch.max(dist_to_closest_centroid, dim=0)[1]
                    initial_centroids.append(max_dist_neuron_idx)
                    initial_centroid_preactivations.append(X @ w[:, max_dist_neuron_idx])

                duration = time.time() - start_time_ci
                time_centroid_initialization += duration
                # assign each neuron to the nearest centroid
                dist_matrix = torch.cdist(torch.stack(initial_centroid_preactivations), acts.T, p=2)
                min_dist_centroid_idx = torch.min(dist_matrix, dim=0)[1]
                assignment = min_dist_centroid_idx.to(self.device)
                prev_assignment = prev_assignment.to(self.device)
                start_time_cc = time.time()
                num_loops = 0
                costs_log = []

                while (prev_assignment != assignment).any():
                    num_loops += 1
                    iterations_centroid_covergence += 1
                    prev_assignment = assignment

                    # calculate the centroids
                    centroid_ws, centroid_preactivations = self._calculate_centroids(output_size, assignment, X, w, s)
                    dist_matrix = torch.cdist(centroid_preactivations, acts.T, p=2)
                    dist_to_closest_centroid, min_dist_centroid_idx = torch.min(dist_matrix, dim=0)
                    assignment = min_dist_centroid_idx

                    current_cost = torch.sum(s * dist_to_closest_centroid)
                    if len(costs_log) >= k_means_early_stop_steps_check and current_cost >= costs_log[-k_means_early_stop_steps_check] * (1.0 - k_means_early_stop_minimum_improvement):
                        print("K-means Early Stopping at ", len(costs_log), " iterations since ", current_cost, " >= ", costs_log[-k_means_early_stop_steps_check], "*", (1.0 - k_means_early_stop_minimum_improvement))
                        break

                    costs_log.append(current_cost)

                    # handle the case where some centroids have no assigned neurons
                    for i in range(output_size):
                        if i not in assignment:
                            max_dist_neuron_idx = torch.max(s * dist_to_closest_centroid, dim=0)[1]
                            assignment[max_dist_neuron_idx] = i
                            dist_to_closest_centroid[max_dist_neuron_idx] = 0

                    # convert to device
                    assignment = assignment.to(self.device)
                    prev_assignment = prev_assignment.to(self.device)


                # calculate the total cost
                _, centroid_preactivations = self._calculate_centroids(output_size, assignment, X, w, s)
                dist_matrix = torch.cdist(centroid_preactivations, acts.T, p=2)
                dist_to_closest_centroid, _ = torch.min(dist_matrix, dim=0)
                total_cost = float(torch.sum(s * dist_to_closest_centroid))
                # change the assignment if the total cost is better
                if total_cost < best_total_cost:
                    best_assignment = assignment
                    best_total_cost = total_cost

                time_centroid_covergence += time.time() - start_time_cc

            print(f'K-means time: {(time.time() - start):.2f}, Centroid Initialization Time: {time_centroid_initialization:.2f}, Centroid Initialization Iterations: {iterations_centroid_initialization}, Centroid Convergence Time: {time_centroid_covergence:.2f}, Centroid Convergence Iterations: {iterations_centroid_covergence}')
        print()

        # create the new neurons
        centroid_ws, _ = self._calculate_centroids(output_size, best_assignment, X, w, s)
        for i, W in enumerate(centroid_ws):
            # split weight and bias (if bias is used)
            if use_bias:
                new_bias[i] = W[-1]
                W = W[:-1]
            # reshape the weight for conv layers
            if is_conv:
                new_weight[i] = W.reshape(in_filters, -1)
            else:
                new_weight[i] = W

        # remove bias if not used
        if not use_bias:
            new_bias = None

        return new_weight, new_bias

    def fuse(
            self,
            activation_dataset,
            activation_labels,
            fusion_method='hungarian',
            neuron_importance_method='uniform',
            norm_acts=False,
            norm_neuron_importance=True
    ):
        """Fuse the models using the Hungarian algorithm."""
        self._config_check(fusion_method, neuron_importance_method)
        num_layers = get_num_layers(self.models)

        # compute the neuron importance scores
        start_time = time.time()
        self.neuron_importance_scores = [[] for _ in range(len(self.models))]
        for k, model in enumerate(self.models):
            for layer_idx in range(num_layers):
                scores = compute_neuron_importance_scores(neuron_importance_method, model, layer_idx,
                                                          activation_dataset, activation_labels, fusion_logic='layer')
                if norm_neuron_importance:
                    scores = scores / torch.sum(scores)
                self.neuron_importance_scores[k].append(scores)
        print(f'Neuron Importance Score Time: {time.time() - start_time}')

        # initialize the activations
        for m in self.models:
            m.to(activation_dataset.device)
            initialize_activations(m, activation_dataset)

        # initialize the fused model
        fused_hat = self.models[0].copy_model().eval()

        # perform the fusion layer by layer
        for layer_idx in range(num_layers):
            print(f'Fusing layer {layer_idx + 1}/{num_layers}')
            # initialize activations for the fused model
            initialize_activations(fused_hat, activation_dataset)

            # fuse the layer (hungarian or k-means)
            if fusion_method == 'hungarian':
                new_W, new_b = self._fuse_layer_hungarian(layer_idx, fused_hat, norm_acts)
            else:
                new_W, new_b = self._fuse_layer_k_means(layer_idx, fused_hat, norm_acts)

            # set the new weights
            fused_hat.set_weight_at_layer(layer_idx, new_W)
            if new_b is not None:
                fused_hat.set_bias_at_layer(layer_idx, new_b)

        # remove the hooks
        for m in self.models:
            m.remove_hooks()
        fused_hat.remove_hooks()

        # copy the model to itself because saving models that have/had hooks is buggy
        fused_hat = fused_hat.copy_model()
        return fused_hat
