import torch
import warnings
import torch.nn as nn

from torchmetrics import ConfusionMatrix
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from models.vgg import VGG
from models.mlp import MLPNet
from models.resnet import BasicBlock, Bottleneck, ResNet
from models.ensemble_vanilla_averaging import Ensemble, VanillaAveraging
from models.vit import MultiHeadSelfAttention, InputEmbedder, TransformerEncoder, ViT
from models.vit_torch import PatchEmbedding, InputEmbedding, PermutedTransformerEncoder, SmallViT


def add_safe_globals():
    """Utility for loading saved-models out-of-the-box, without having to load the weights instead."""
    torch.serialization.add_safe_globals([
        MLPNet, VGG, BasicBlock, Bottleneck, ResNet, PatchEmbedding, InputEmbedding, PermutedTransformerEncoder, SmallViT,
        MultiHeadSelfAttention, InputEmbedder, TransformerEncoder, ViT,
        Ensemble, VanillaAveraging,
        nn.Linear, nn.Dropout, nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.Dropout2d,
        nn.Parameter, nn.TransformerEncoderLayer, nn.MultiheadAttention, nn.modules.linear.NonDynamicallyQuantizableLinear,
        nn.Dropout, nn.LayerNorm,
        nn.Sequential, nn.ModuleList,
        nn.ReLU, nn.GELU,
        nn.Flatten, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d,
        set
    ])


def eval_model(model, test_dataloader, device, num_classes=10):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    with torch.no_grad():
        # place all predictions and targets in the lists below
        stacked_outputs, stacked_targets = [], []
        for i, (data, target) in enumerate(test_dataloader):
            output = model(data.to(device)).cpu()
            stacked_outputs.append(output)
            stacked_targets.append(target)
        stacked_outputs, stacked_targets = torch.cat(stacked_outputs), torch.tensor(stacked_targets) if isinstance(stacked_targets[0], int) else torch.cat(stacked_targets)

        # compute the confusion matrix
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        confustion_matrix = confmat(torch.argmax(stacked_outputs, dim=1), stacked_targets)

        # calculate the loss
        loss = criterion(stacked_outputs, stacked_targets).item()

        # calculate the accuracy, f1 score
        point_predictions = stacked_outputs.argmax(dim=1)
        acc = accuracy_score(stacked_targets, point_predictions)
        f1 = f1_score(stacked_targets, point_predictions, average='macro')

        # calculate the roc_auc score
        stacked_targets_onehot = torch.zeros_like(stacked_outputs)  # one hot encode the targets
        stacked_targets_onehot.scatter_(1, stacked_targets.view(-1, 1), 1)
        stacked_outputs = torch.softmax(stacked_outputs, dim=1)
        # we can get annoying warnings for imbalanced datasets, so we ignore them
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            roc_auc = roc_auc_score(stacked_targets_onehot.numpy(), stacked_outputs.numpy(), average='weighted', multi_class='ovr')  # ovr == One vs Rest

        return {
            'loss': loss,
            'accuracy': acc,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confustion_matrix
        }
