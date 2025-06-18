import torch

from abc import ABC, abstractmethod
from captum.attr import (
    LayerConductance,
    LayerDeepLift,
    LayerFeatureAblation
)

EPSILON=1e-8


class NeuronImportanceScoreMethod(ABC):

    @staticmethod
    @abstractmethod
    def get_score(model, layer, output_width, X, y, batch_size=None):
        """
        Computes the importance scores for the neurons in the specified layer of a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model for which neuron importance scores are computed.
            layer (torch.nn.Module): The specific layer in the model whose neuron importance scores are to be calculated.
            output_width (int): The width of the output layer, used to determine the shape of the scores.
            X (torch.Tensor): Input tensor to the model, used for computing the scores. 
                                Shape should match the model's input requirements.
            y (torch.Tensor): A tensor containing the target class labels corresponding to the input data. 
            batch_size (int or None, optional): The number of inputs to process in each batch. 
                                                If None, all inputs are processed at once.

        Returns:
            torch.Tensor: A tensor of the same shape as the output of the specified layer, where each element represents 
                        the importance score of the corresponding neuron. If the layer output has multiple dimensions,
                        scores are aggregated across all dimensions except the first.

        Notes:
            - The method is abstract and must be implemented by subclasses.
            - The returned scores are normalized and adjusted with a small epsilon value to avoid zero scores.
        """
        pass

    @staticmethod
    def _accumulate_scores(method, output_width, X, y, batch_size, **kwargs):
        """Accumulate neuron importance scores for a given method."""
        N = X.shape[0]
        batch_size = batch_size or N
        scores = torch.zeros(output_width).cpu()
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            X_batch = X[i:end]
            y_batch = y[i:end]
            batch_scores = method.attribute(X_batch, target=y_batch, **kwargs).detach().abs().sum(0).cpu()
            scores += batch_scores.sum(dim=tuple(range(1, batch_scores.ndim))) if batch_scores.ndim > 1 else batch_scores
        return scores
    
    @staticmethod
    def _normalize_scores(scores, X):
        """Normalize the scores by dividing by the number of samples."""
        N = X.shape[0]
        return scores / N


class Conductance(NeuronImportanceScoreMethod):

    @staticmethod
    def get_score(model, layer, output_width, X, y, batch_size=None):
        # accumulate scores
        lc = LayerConductance(model, layer)
        scores = NeuronImportanceScoreMethod._accumulate_scores(lc, output_width, X, y, batch_size, n_steps=10)

        # normalize scores
        scores = NeuronImportanceScoreMethod._normalize_scores(scores, X)

        return scores.to(model.device) + EPSILON


class DeepLIFT(NeuronImportanceScoreMethod):

    @staticmethod
    def get_score(model, layer, output_width, X, y, batch_size=None):
        # accumulate scores
        dl = LayerDeepLift(model, layer)

        scores = NeuronImportanceScoreMethod._accumulate_scores(dl, output_width, X, y, batch_size)

        # normalize scores
        scores = NeuronImportanceScoreMethod._normalize_scores(scores, X)

        return scores.to(model.device) + EPSILON


class FeatureAblation(NeuronImportanceScoreMethod):

    @staticmethod
    def get_score(model, layer, output_width, X, y, batch_size=None):
        # accumulate scores
        fa = LayerFeatureAblation(model, layer)
        scores = NeuronImportanceScoreMethod._accumulate_scores(fa, output_width, X, y, batch_size)

        # normalize scores
        scores = NeuronImportanceScoreMethod._normalize_scores(scores, X)

        return scores.to(model.device) + EPSILON
