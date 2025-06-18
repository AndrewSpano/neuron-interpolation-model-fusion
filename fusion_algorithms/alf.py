import time
import torch
import torch.nn as nn
import numpy as np

from tqdm import trange
from dataclasses import dataclass
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader

from models.base_model import BaseModel
from utils.model_utils import (
    is_conv_activation,
    is_transformer_activation,
    reset_parameters,
    soft_reset_parameters,
    disable_dropout
)
from utils.fusion_utils import (
    get_num_levels,
    normalize_activations_per_sample,
    compute_neuron_importance_scores
)

from scipy.optimize import linear_sum_assignment



@dataclass
class ALFusionConfig:
    """Configuration class for the ALFusion class."""

    X: torch.Tensor
    y: torch.Tensor | None
    optimizer_per_level: list[int, str]
    lr_per_level: list[float]
    weight_decay_per_level: list[float]
    epochs_per_level: list[int]
    use_scheduler: bool = False
    neuron_importance_method: str = 'uniform'
    normalize_neuron_importances: bool = False
    align_skip_connections: bool = False
    train_batch_size: int = 32
    max_patience_epochs: int = 15
    normalize_acts_for_kmeans: bool = False
    num_kmeans_runs: int = 1
    init_kmeans_with_2nd_acts: bool = True
    use_weighted_loss: bool = True
    cls_head_weights: torch.Tensor | None = None
    use_kd_on_head: bool = False
    kd_temperature: float = 2.0
    skip_levels: list[int] = None
    reset_level_parameters: list[int] = None
    weight_perturbation_per_level: list[float] | None = None
    verbose: bool = True


class ActivationDataset(Dataset):
    """A torch dataset aimed to represent inputs and activations
        of specific levels as the target. `X` is a tensor representing
        an input dataset, of shape [B, num_features], and `T` represents
        a target tensor of shape [B, target_width_of_level]."""

    def __init__(self, X: torch.Tensor, T: torch.Tensor) -> None:
        self.X = X
        self.T = T

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]


class ALFusion:
    """Activation Learning Fusion (ALFusion) class."""

    neuron_importance_methods = [
        'uniform',
        'conductance',
        'deeplift',
        'featureablation'
    ]

    def __init__(
        self,
        models: list[BaseModel]
    ) -> None:
        self.models = models
        self.ni_scores: list[list[torch.Tensor]] = []
        self.weight_change_per_level: list[float] = []

    @property
    def device(self) -> torch.device:
        """Return the device of the first model."""
        return self.models[0].device

    def _config_check(self, fusion_config: ALFusionConfig):
        """Check the configuration of the fusion method and neuron importance method."""
        neuron_importance_method = fusion_config.neuron_importance_method
        if neuron_importance_method not in self.neuron_importance_methods:
            raise ValueError(f'Invalid neuron importance method: {neuron_importance_method}')
        
    def _flatten_acts(self, acts: torch.Tensor) -> torch.Tensor:
        """Flatten the activations tensor."""
        if len(acts.shape) == 2:
            return acts  # MLP-like: [B, num_neurons]
        elif len(acts.shape) == 3:
            hidden_dim = acts.shape[1]
            return acts.permute(0, 2, 1).reshape(-1, hidden_dim)  # (Permuted) Transformer-like: [B, seqlen, num_neurons]
        elif len(acts.shape) == 4:
            channels = acts.shape[1]
            return acts.permute(0, 3, 2, 1).reshape(-1, channels)  # CNN-like: [B, channels, height, width]
        else:
            raise ValueError(f'Invalid shape for activations: {acts.shape}. Expected a 2D, 3D or 4D tensor.')
        
    def _unflatten_target(self, T: torch.Tensor, target_width: int, height: int, width: int, seqlen: int) -> torch.Tensor:
        """Unflatten the target tensor."""
        if height != 0 and width != 0:
            return T.reshape(-1, width, height, target_width).permute(0, 3, 2, 1)  # CNN-like: [B, channels, height, width]
        elif seqlen != 0:
            return T.reshape(-1, seqlen, target_width).permute(0, 2, 1)  # (Permuted) Transformer-like: [B, hidden_dim, seqlen]
        else:
            return T  # MLP-like: [B, num_neurons]

    def _match_T_to_X(self, X, T, W, chunk_size=64):
        """Match the target matrix T to the input matrix X using the Hungarian algorithm."""
        X = self._flatten_acts(X)
        B, num_neurons = X.shape
        _, num_targets = T.shape
        cost_matrix = torch.zeros((num_neurons, num_targets), device=X.device)
        for i_start in range(0, num_neurons, chunk_size):
            i_end = min(i_start + chunk_size, num_neurons)
            X_chunk = X[:, i_start:i_end]  # [B, chunk_size]
            dists = torch.norm(X_chunk.unsqueeze(2) - T.unsqueeze(1), dim=0)  # (B, chunk_size, num_targets) --> (chunk_size, num_targets)
            cost_matrix[i_start:i_end, :] = dists * W

        row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        aligned_T = torch.zeros_like(T)
        for r, c in zip(row_indices, col_indices):
            aligned_T[:, r] = T[:, c]

        return aligned_T
        
    @staticmethod
    def _calculate_centroids(output_size, assignment, acts, weights=None):
        """Calculate the centroids of the assigned neurons for the k-means variant of the fusion procedure."""
        # get the assigned neurons for each centroid
        all_assigned_neurons = [[] for _ in range(output_size)]
        for i, x in enumerate(assignment):
            all_assigned_neurons[x].append(i)

        # compute the new centroids
        centroids = []
        for i, assigned_neurons in enumerate(all_assigned_neurons):
            if len(assigned_neurons) == 0:
                continue
            
            if weights is not None:
                # compute the weighted mean
                weights_assigned_neurons = weights[assigned_neurons]
                weights_assigned_neurons = weights_assigned_neurons / torch.sum(weights_assigned_neurons)
                centroids.append(torch.sum(acts[assigned_neurons] * weights_assigned_neurons[:, None], axis=0))
            else:
                centroids.append(torch.mean(acts[assigned_neurons], axis=0))

        # convert the preactivations to a tensor and return
        centroids = torch.stack(centroids)
        return centroids

    def _k_means_v2(
        self,
        level_idx: int,
        target_width: int,
        fusion_config: ALFusionConfig,
        level_X: torch.Tensor
    ) -> torch.Tensor:
        """GPU-optimized k-means for the (pre-)activations of the given level."""
        X = fusion_config.X
        time_centroid_covergence = 0
        iterations_centroid_covergence = 0
        k_means_early_stop_steps_check = 3
        k_means_early_stop_minimum_improvement = 0

        # for conv/transformer layers
        seqlen, height, width = 0, 0, 0

        # get the activations at `level_idx` for each model
        with torch.no_grad():
            acts = [m.forward_until_level(X, level_idx) for m in self.models]

        # decide whether to match the resulting centroids to the input, after running k-means
        match_T_to_level_X = fusion_config.align_skip_connections and (acts[0].shape == level_X.shape)

        # check the shape of activations to reshape accordingly later
        if is_conv_activation(acts[0]):
            height, width = acts[0].shape[2], acts[0].shape[3]
        elif is_transformer_activation(acts[0]):
            seqlen = acts[0].shape[2]

        # flatten the activations
        acts = [self._flatten_acts(act) for act in acts]
        norm_acts = [normalize_activations_per_sample(act) for act in acts]

        # concatenate the activations
        acts = torch.cat(tuple(acts), dim=1)
        norm_acts = torch.cat(tuple(norm_acts), dim=1)

        # decide which activations to use for k-means (normalized or not)
        kmeans_acts = norm_acts if fusion_config.normalize_acts_for_kmeans else acts

        # get sample weight for the activations, which are the neuron importance scores
        ni_scores_of_level = [self.ni_scores[k][level_idx] for k in range(len(self.models))]
        sample_weight = torch.cat(ni_scores_of_level).to(self.device)  # [sum(output_widths_of_level_of_models)]

        # run k-means to group neurons
        best_total_cost, best_assignment = float('inf'), None
        for _ in range(fusion_config.num_kmeans_runs):

            # initialize the centroids (by the individual neuros from either the 2nd or 1st model)
            if fusion_config.init_kmeans_with_2nd_acts:
                centroid_preactivations = kmeans_acts[:, target_width:2*target_width].T
            else:
                centroid_preactivations = kmeans_acts[:, :target_width].T

            # assign each neuron to the nearest centroid
            dist_matrix = torch.cdist(centroid_preactivations, kmeans_acts.T, p=2)
            min_dist_centroid_idx = torch.min(dist_matrix, dim=0)[1]
            assignment = min_dist_centroid_idx.to(self.device)
            prev_assignment = torch.ones(len(assignment)) * (-1)
            prev_assignment = prev_assignment.to(self.device)
            start_time_cc = time.time()
            num_loops = 0
            costs_log = []

            while (prev_assignment != assignment).any():
                num_loops += 1
                iterations_centroid_covergence += 1
                prev_assignment = assignment

                # calculate the centroids
                centroid_preactivations = self._calculate_centroids(target_width, assignment, kmeans_acts.T, weights=sample_weight)
                dist_matrix = torch.cdist(centroid_preactivations, kmeans_acts.T, p=2)
                dist_to_closest_centroid, min_dist_centroid_idx = torch.min(dist_matrix, dim=0)
                assignment = min_dist_centroid_idx

                # handle the case where some centroids have no assigned neurons
                counts = torch.bincount(assignment, minlength=target_width)
                unassigned_clusters = set(range(target_width)) - set(assignment.tolist())
                while unassigned_clusters:
                    i = unassigned_clusters.pop()  # Get the first unassigned cluster
                    max_dist_neuron_idx = torch.max(dist_to_closest_centroid, dim=0)[1]
                    if counts[assignment[max_dist_neuron_idx].item()] == 1:
                        dist_to_closest_centroid[max_dist_neuron_idx] = float('-inf')  # Mark as assigned
                        unassigned_clusters.add(i)
                        continue

                    counts[assignment[max_dist_neuron_idx].item()] -= 1
                    assignment[max_dist_neuron_idx] = i
                    dist_to_closest_centroid[max_dist_neuron_idx] = float('-inf')  # Mark as assigned

                dist_to_closest_centroid = torch.where(dist_to_closest_centroid == float('-inf'), torch.tensor(0.0, device=dist_to_closest_centroid.device), dist_to_closest_centroid)  # Replace -inf with 0

                # convert to device
                assignment = assignment.to(self.device)
                prev_assignment = prev_assignment.to(self.device)

                current_cost = torch.sum(dist_to_closest_centroid * sample_weight)
                if len(costs_log) >= k_means_early_stop_steps_check and current_cost >= costs_log[-k_means_early_stop_steps_check] * (1.0 - k_means_early_stop_minimum_improvement):
                    print("K-means Early Stopping at ", len(costs_log), " iterations since ", current_cost, " >= ", costs_log[-k_means_early_stop_steps_check], "*", (1.0 - k_means_early_stop_minimum_improvement))
                    break
                costs_log.append(current_cost)

            # calculate the total cost
            centroid_preactivations = self._calculate_centroids(target_width, assignment, kmeans_acts.T, weights=sample_weight)
            dist_matrix = torch.cdist(centroid_preactivations, kmeans_acts.T, p=2)
            dist_to_closest_centroid, _ = torch.min(dist_matrix, dim=0)
            total_cost = float(torch.sum(dist_to_closest_centroid * sample_weight))

            # change the assignment if the total cost is better
            if total_cost < best_total_cost:
                best_assignment = assignment
                best_total_cost = total_cost

            time_centroid_covergence += time.time() - start_time_cc

        # final target matrix -- align it to the input if needed (for skip connections)
        centroid_preactivations = self._calculate_centroids(target_width, best_assignment, acts.T, weights=sample_weight)
        T = centroid_preactivations.T
        if match_T_to_level_X:
            total_importances = torch.zeros(T.shape[1], device=self.device)
            for i in range(best_assignment.shape[0]):
                total_importances[best_assignment[i]] += sample_weight[i]
            T = self._match_T_to_X(level_X, T, total_importances)

        # reshape the target matrix to match the activations
        T = self._unflatten_target(T, target_width, height, width, seqlen)
        return T, best_assignment

    def _k_means(
        self,
        level_idx: int,
        target_width: int,
        fusion_config: ALFusionConfig
    ) -> torch.Tensor:
        """Run k-means on the activations of the given level. This is slow as it's not optimized for GPU, and should be avoided.
            Also it's deprecated."""
        # for conv/transformer layers
        seqlen, height, width = 0, 0, 0

        # get the activations at `level_idx` for each model
        X = fusion_config.X
        with torch.no_grad():
            acts = [m.forward_until_level(X, level_idx) for m in self.models]

        # check the shape of activations to reshape accordingly later
        if is_conv_activation(acts[0]):
            height, width = acts[0].shape[2], acts[0].shape[3]
        elif is_transformer_activation(acts[0]):
            seqlen = acts[0].shape[2]

        # flatten the activations (and normalize if needed)
        acts = [self._flatten_acts(act) for act in acts]
        acts = torch.cat(acts, dim=1).cpu().numpy().T  # [sum(output_widths_of_level_of_models) x B]

        # run k-means on the activations to get the target matrix T
        ni_scores_of_level = [self.ni_scores[k][level_idx] for k in range(len(self.models))]
        sample_weight = torch.cat(ni_scores_of_level).cpu().detach().numpy()  # [sum(output_widths_of_level_of_models)]
        kmeans = KMeans(n_clusters=target_width, random_state=42, n_init=fusion_config.num_kmeans_runs)
        kmeans.fit(acts, sample_weight=sample_weight)
        T = torch.from_numpy(kmeans.cluster_centers_).T  # [B x target_width]
        T = self._unflatten_target(T, target_width, height, width, seqlen)

        # get the assignment
        assignment = torch.zeros(acts.shape[0], dtype=torch.int)
        for i, x in enumerate(kmeans.labels_):
            assignment[i] = x

        return T, assignment

    def _hungarian(
        self,
        level_idx: int,
        target_width: int,
        fusion_config: ALFusionConfig
    ) -> torch.Tensor:
        """Run k-means on the activations of the given level."""
        # get the activations at `level_idx` for each model
        X = fusion_config.X
        with torch.no_grad():
            acts = [m.forward_until_level(X, level_idx) for m in self.models]

        height, width, seqlen = 0, 0, 0
        # check the shape of activations to reshape accordingly later
        if is_conv_activation(acts[0]):
            height, width = acts[0].shape[2], acts[0].shape[3]
        elif is_transformer_activation(acts[0]):
            # seqlen = acts[0].shape[1]
            seqlen = acts[0].shape[2]

        acts = [self._flatten_acts(act) for act in acts]

        if fusion_config.normalize_acts:
            acts = [normalize_activations_per_sample(act) for act in acts]

        assert len(acts) == 2 and acts[0].shape == acts[1].shape

        # acts = torch.cat(acts, dim=1).cpu().numpy().T  # [sum(output_widths_of_level_of_models) x B]

        # run k-means on the activations to get the target matrix T
        ni_scores_of_level = [self.ni_scores[k][level_idx] for k in range(len(self.models))]
        #sample_weight = torch.cat(ni_scores_of_level).cpu().detach().numpy()  # [sum(output_widths_of_level_of_models)]

        cost_matrix = np.zeros((acts[0].shape[1], acts[1].shape[1]))

        for i in range(acts[0].shape[1]):
            for j in range(acts[1].shape[1]):
                act_pred = (ni_scores_of_level[0][i] * acts[0][:, i] + ni_scores_of_level[1][j] * acts[1][:, j]) / (ni_scores_of_level[0][i] + ni_scores_of_level[1][j])
                cost0 = ni_scores_of_level[0][i] * torch.norm(act_pred - acts[0][:, i], dim=0)
                cost1 = ni_scores_of_level[1][j] * torch.norm(act_pred - acts[1][:, j], dim=0)
                cost_matrix[i][j] = (cost0 + cost1).item()

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        num_levels = get_num_levels(self.models)
        if level_idx == num_levels - 1:
            col_indices = row_indices

        T = torch.zeros_like(acts[0])
        assignment = torch.zeros(acts[0].shape[1] + acts[1].shape[1], dtype=torch.int)
        for i, j in zip(row_indices, col_indices):
            T[:,i] = (ni_scores_of_level[0][i] * acts[0][:, i] + ni_scores_of_level[1][j] * acts[1][:, j]) / (ni_scores_of_level[0][i] + ni_scores_of_level[1][j])
            assignment[i] = i
            assignment[acts[0].shape[1] + j] = i
        print(assignment)
        T = self._unflatten_target(T, target_width, height, width, seqlen)
        return T, assignment

    def _last_level_grouping(self, num_classes: int, fusion_config: ALFusionConfig):
        """For the output layer, we want to have a 1-1 matching for output classes."""
        all_model_logits = []
        with torch.no_grad():
            for i, m in enumerate(self.models):
                logits = m(fusion_config.X)
                if fusion_config.cls_head_weights is not None:
                    logits = logits * fusion_config.cls_head_weights[i]
                if fusion_config.use_kd_on_head:
                    logits = (logits / fusion_config.kd_temperature).softmax(dim=1)
                all_model_logits.append(logits)
        all_model_logits = torch.cat(tuple(all_model_logits), dim=1)  # [B x (num_classes * num_models)]
        assignment = torch.cat([torch.arange(start=0, end=num_classes) for _ in range(len(self.models))])  # [num_classes * num_models]
        T = self._calculate_centroids(num_classes, assignment, all_model_logits.T).T
        return T, assignment
    
    def _criterion_vectorized(
        self,
        act_pred: torch.Tensor,
        act_true: torch.Tensor,
        cluster_weights: torch.Tensor
    ) -> torch.Tensor:
        se = (act_pred - act_true) ** 2
        loss = (se * cluster_weights).mean()
        return loss

    def _get_criterion(
        self,
        level_idx: int,
        target_width: int,
        assignment: torch.Tensor,
        fusion_config: ALFusionConfig,
        is_last_level: bool
    ) -> nn.Module:
        """Get the criterion for the fusion process."""
        # use MSE on the last layer if specified
        if not fusion_config.use_weighted_loss or (is_last_level and fusion_config.cls_head_weights is None):
            return nn.MSELoss().to(self.device)
        
        # use CE on the last layer if specified
        if is_last_level and fusion_config.use_kd_on_head:
            return nn.KLDivLoss(reduction='batchmean').to(self.device)

        # ToDo: maybe for the last layer use head weights if available?
        num_models = len(self.models)
        ni_scores = torch.cat([self.ni_scores[i][level_idx] for i in range(num_models)], dim=0)  # [sum(output_widths_of_level_of_models)]
        ni_scores = ni_scores.detach().to(self.device)

        # compute the cluster weights
        W = torch.zeros(target_width, ni_scores.shape[0], device=self.device)
        W[assignment, torch.arange(ni_scores.shape[0])] = ni_scores
        cluster_weights = W.sum(dim=1)  # shape: [target_width,]

        # define and return the loss function
        def weighted_loss(y_pred, y_true):
            return self._criterion_vectorized(y_pred, y_true, cluster_weights)
        return weighted_loss
    
    def _get_optimizer(
        self,
        level_idx: int,
        level_network: nn.Module,
        fusion_config: ALFusionConfig
    ) -> torch.optim.Optimizer:
        """Get the optimizer for the fusion process."""
        level_params = level_network.parameters()
        lr = fusion_config.lr_per_level[level_idx]
        weight_decay = fusion_config.weight_decay_per_level[level_idx]
        optim_name = fusion_config.optimizer_per_level[level_idx].lower()
        if fusion_config.verbose:
            print(f'Optimizer for level {level_idx}: {optim_name}. Learning rate: {lr}. Weight decay: {weight_decay}.')
        if optim_name == 'adam':
            return torch.optim.Adam(level_params, lr=lr, weight_decay=weight_decay)
        elif optim_name == 'sgd':
            return torch.optim.SGD(level_params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'Invalid optimizer: {optim_name}')

    def _get_level_input(
            self,
            level_idx: int,
            fused_hat: BaseModel,
            fusion_config: ALFusionConfig
    ) -> torch.Tensor:
        """Get the input for the given level_idx."""
        X = fusion_config.X
        if level_idx == 0:
            return X

        # get the output of the previous level
        fused_hat.eval()
        with torch.no_grad():
            acts = fused_hat.forward_until_level(X, level_idx - 1)
        return acts

    def _fuse_level_sgd(
        self,
        level_idx: int,
        fused_hat: BaseModel,
        fusion_config: ALFusionConfig
    ):
        """Fuse the given level_idx using SGD to match activations of levels."""
        level_X = self._get_level_input(level_idx, fused_hat, fusion_config)
        level = fused_hat.get_ordered_trainable_named_levels[level_idx]
        target_width = level.output_width
        network = level.network

        # track changes in weights
        weights_before = nn.utils.parameters_to_vector(network.parameters()).detach().clone()
    
        # get the target matrix T, and the assignment that produces it
        num_levels = get_num_levels(self.models)
        is_last_level = level_idx == num_levels - 1
        if level_idx < num_levels - 1:
            T, assignment = self._k_means_v2(level_idx, target_width, fusion_config, level_X)
        else:
            T, assignment = self._last_level_grouping(target_width, fusion_config)

        # create a train/val dataloaders
        dataset = ActivationDataset(level_X, T)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        train_dl = DataLoader(train_dataset, batch_size=fusion_config.train_batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=fusion_config.train_batch_size, shuffle=False)

        # randomize the weights of the current level
        if level_idx in (fusion_config.reset_level_parameters or []):
            reset_parameters(network)
        # or soft reset the weights
        elif fusion_config.weight_perturbation_per_level is not None and fusion_config.weight_perturbation_per_level[level_idx] > 0:
            soft_reset_parameters(network, fusion_config.weight_perturbation_per_level[level_idx])

        # train the current level
        network.train().float()
        disable_dropout(network)
        optimizer = self._get_optimizer(level_idx, network, fusion_config)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        # get the loss function
        criterion = self._get_criterion(level_idx, target_width, assignment, fusion_config, is_last_level)

        # train the model
        patience = fusion_config.max_patience_epochs
        best_val_loss = float('inf')
        num_epochs = fusion_config.epochs_per_level[level_idx]
        for epoch in range(num_epochs):
            train_loss = 0.0
            network.train(True)
            for batch in train_dl:
                x, y = batch
                x, y = x.to(self.device).float(), self._flatten_acts(y).to(self.device).float()
                optimizer.zero_grad()

                y_hat = network(x)
                y_hat = self._flatten_acts(y_hat)

                loss = criterion(y_hat, y)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            if fusion_config.use_scheduler:
                scheduler.step(train_loss)

            val_loss = 0.0
            network.eval()
            with torch.no_grad():
                for batch in val_dl:
                    x, y = batch
                    x, y = x.to(self.device).float(), self._flatten_acts(y).to(self.device).float()

                    y_hat = network(x)
                    y_hat = self._flatten_acts(y_hat)

                    loss = criterion(y_hat, y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_dl)
            avg_val_loss = val_loss / len(val_dl)

            if fusion_config.verbose:
                print(f'Epoch: {epoch + 1}/{num_epochs}.    Avg Train Loss: {avg_train_loss:.4f}.    Avg Val Loss: {avg_val_loss:.4f}.')
            if avg_val_loss >= best_val_loss:
                patience -= 1
                if patience == 0:
                    if fusion_config.verbose:
                        print('Early stopping')
                    break
            else:
                best_val_loss = avg_val_loss
                patience = fusion_config.max_patience_epochs

        # track the weight changes
        weights_after = nn.utils.parameters_to_vector(network.parameters()).detach().clone()
        weight_change = torch.norm(weights_after - weights_before).item()
        self.weight_change_per_level.append(weight_change)
        if fusion_config.verbose:
            print(f'Weight change for level {level_idx}: {weight_change:.4f}')

    def fuse(
            self,
            fusion_config: ALFusionConfig,
            fused_hat: BaseModel | None = None
    ) -> BaseModel:
        """Fuse the models using the Hungarian algorithm."""
        self._config_check(fusion_config)
        num_levels = get_num_levels(self.models)

        # compute the neuron importance scores
        start_time = time.time()
        self.ni_scores = [{} for _ in range(len(self.models))]
        self.weight_change_per_level = []
        for k, model in enumerate(self.models):
            for level_idx in range(num_levels):
                if level_idx in (fusion_config.skip_levels or []):
                    continue
                scores = compute_neuron_importance_scores(
                    fusion_config.neuron_importance_method,
                    model,
                    idx=level_idx,
                    X=fusion_config.X,
                    y=fusion_config.y,
                    fusion_logic='level'
                )
                if fusion_config.normalize_neuron_importances:
                    scores = scores / torch.sum(scores)
                self.ni_scores[k][level_idx] = scores
        if fusion_config.verbose:
            print(f'Neuron Importance Score Time: {time.time() - start_time:.2f} seconds')

        # initialize the fused model
        if fused_hat is None:
            fused_hat = self.models[0].copy_model()

        # perform the fusion level by level
        for level_idx in trange(num_levels, desc='Fusing levels'):
            if level_idx in (fusion_config.skip_levels or []):
                continue
            # fuse the levels -- note that the fusion happens in-place for `fused_hat`
            self._fuse_level_sgd(level_idx, fused_hat, fusion_config)

        return fused_hat
