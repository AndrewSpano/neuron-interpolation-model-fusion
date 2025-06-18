import os
import json
import time
import torch

from pathlib import Path
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from utils.general_utils import set_seed
from fusion_algorithms.hf_kf import LSFusion
from fusion_algorithms.otfusion import OTFusion
from fusion_algorithms.git_rebasin import GitRebasin
from utils.dataset_utils import get_mnist_dataset, get_cifar10_dataset, get_tiny_imagenet_dataset, sample_balanced_subset

import lightning as L


DATASETS_DIR = Path(__file__).parent/'datasets'
MNIST_ROOT = DATASETS_DIR/'mnist'
CIFAR10_ROOT = DATASETS_DIR/'CIFAR10'
CIFAR100_ROOT = DATASETS_DIR/'CIFAR100'


class CustomTrainingModule(L.LightningModule):
    def __init__(self, model, loss, num_classes, optimizer=None, lr_scheduler=None, use_batchnorm=True, linear_bias=True, savepath=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.savepath = savepath

        if optimizer is None:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return  { "optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler }

    def on_save_checkpoint(self, checkpoint):
        """Override to save the full PyTorch model along with the Lightning checkpoint."""
        if self.savepath is not None:
            torch.save(self.model, self.savepath)

def get_eval_test_dls():
    test_dls = {}
    cifar_transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load the dataset
    test_cifar10 = CIFAR10(
        CIFAR10_ROOT,
        train=False,
        download=not CIFAR10_ROOT.exists(),
        transform=cifar_transform
    )
    test_dls["CIFAR10"] = DataLoader(test_cifar10, batch_size=256, shuffle=False)

    mnist_transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
    ])
    test_mnist = MNIST(
        MNIST_ROOT,
        train=False,
        download=False,
        transform=mnist_transform
    )
    test_dls["MNIST"] = DataLoader(test_mnist, batch_size=256, shuffle=False)
    return test_dls

def eval_model(model, test_dataloader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        # place all predictions and targets in the lists below
        stacked_outputs, staked_targets = [], []
        for i, (data, target) in enumerate(test_dataloader):
            output = model(data.to(device)).cpu()
            stacked_outputs.append(output)
            staked_targets.append(target)
        stacked_outputs, staked_targets = torch.cat(stacked_outputs), torch.cat(staked_targets)

        # calculate the loss
        loss = criterion(stacked_outputs, staked_targets).item()

        # calculate the accuracy, f1 score
        point_predictions = stacked_outputs.argmax(dim=1)
        acc = accuracy_score(staked_targets, point_predictions)
        f1 = f1_score(staked_targets, point_predictions, average='weighted')

        # calculate the roc_auc score
        staked_targets_onehot = torch.zeros_like(stacked_outputs)  # one hot encode the targets
        staked_targets_onehot.scatter_(1, staked_targets.view(-1, 1), 1)
        roc_auc = roc_auc_score(staked_targets_onehot, stacked_outputs, average='weighted', multi_class='ovr')  # ovr == One vs Rest

        return {
            'loss': loss,
            'accuracy': acc,
            'f1': f1,
            'roc_auc': roc_auc
        }

def get_combined_validation_datasets(dataset_name, split_file_path, train_test_split=0.8):
    if dataset_name.lower() == "mnist":
        mnist_transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST(
            MNIST_ROOT,
            train=True,
            download=not MNIST_ROOT.exists(),
            transform=mnist_transform
        )
    elif dataset_name.lower() == "cifar10":
        cifar_transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CIFAR10(
            CIFAR10_ROOT,
            train=True,
            download=not CIFAR10_ROOT.exists(),
            transform=cifar_transform
        )
    elif dataset_name.lower() == "cifar100":
        cifar_transform = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CIFAR100(
            CIFAR100_ROOT,
            train=True,
            download=not CIFAR100_ROOT.exists(),
            transform=cifar_transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is unsupported.")

    with open(split_file_path, "r") as f:
        split_structure = json.load(f)

    final_indices = []
    for split_indices in split_structure.values():
        length = len(split_indices)
        border = round(length * train_test_split)
        final_indices.extend(split_indices[border:])
    
    return Subset(dataset, final_indices)

    


def run_fusion_suite(save_dir, device, combination_subset=None, seed=123):
    
    print(f'Using device: {device}')
    set_seed(seed)

    save_dir_path = Path(save_dir)
    models_paths = list(save_dir_path.glob('*.pth'))
    
    # get the data
    if combination_subset is not None:
        alignment_dataset = combination_subset
    elif 'cifar' in save_dir.lower():
        train_dataloaders, _ = get_cifar10_dataset(len(models_paths), False)
        alignment_dataset = train_dataloaders[0].dataset
    elif 'tiny-imagenet' in save_dir.lower():
        train_dataloaders, _ = get_tiny_imagenet_dataset(len(models_paths), False)
        alignment_dataset = train_dataloaders[0].dataset
    else:
        train_dataloaders, _ = get_mnist_dataset(len(models_paths), False)
        alignment_dataset = train_dataloaders[0].dataset
    original_dataset = alignment_dataset.dataset

    os.makedirs(save_dir_path/'fusion_results', exist_ok=True)

    # load pretrained models
    models = []
    for model_idx in range(len(models_paths)):
        weights_path = save_dir_path/f'model{model_idx}.pth'
        model = torch.load(weights_path, map_location=device, weights_only=True).eval()
        torch.save(model, save_dir_path/f'fusion_results/model{model_idx}.pth')
        models.append(model)
        print(f'Model {model_idx} loaded from {save_dir_path}. Model class: {model.__class__.__name__}.')

    # create a VanillaAveraging model and save it
    from models.ensemble_vanilla_averaging import VanillaAveraging, Ensemble
    vanilla_averaging_model = VanillaAveraging(models)
    torch.save(vanilla_averaging_model, save_dir_path/'fusion_results/vanilla-averaging.pth')

    # create an ensemble model and save it
    ensemble = Ensemble(models)
    torch.save(ensemble, save_dir_path/'fusion_results/ensemble.pth')

    # fuse the models and save the fused model
    print('-' * 50)
    print(f'Fusing the trained models')
    activation_dataset, activation_classes = sample_balanced_subset(original_dataset, alignment_dataset, batch_size=200)
    activation_dataset = activation_dataset.to(device)
    activation_classes = activation_classes.to(device)

    times = []
    if len(models) == 2: # For only 2-way fusion algos
        
        print("Hungarian Uniform Scores")
        start = time.time()
        fusion = LSFusion([models[0], models[1]])
        fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='hungarian', neuron_importance_method='uniform')
        times.append(time.time() - start)
        torch.save(fused_model, save_dir_path/'fusion_results/hungarian-uniform.pth')

        torch.cuda.empty_cache()

        print("Hungarian Conductance-based Scores")
        start = time.time()
        fusion = LSFusion([models[0], models[1]])
        fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='hungarian', neuron_importance_method='conductance')
        times.append(time.time() - start)
        torch.save(fused_model, save_dir_path/'fusion_results/hungarian-conductance.pth')
        
        torch.cuda.empty_cache()
        
        print("Hungarian DeepLIFT-based Scores")
        start = time.time()
        fusion = LSFusion([models[0], models[1]])
        fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='hungarian', neuron_importance_method='deeplift')
        times.append(time.time() - start)
        torch.save(fused_model, save_dir_path/'fusion_results/hungarian-deeplift.pth')

        torch.cuda.empty_cache()

        
        print("Git Rebasin Uniform Scores")
        git_rebasin = GitRebasin(models[0], models[1])
        start = time.time()
        fused_model = git_rebasin.fuse(activation_dataset, activation_classes, neuron_importance_method='uniform', handle_skip=True)
        torch.save(fused_model, save_dir_path/'fusion_results/git_rebasin-uniform.pth')
        times.append(time.time() - start)
        torch.cuda.empty_cache()
    else:
        times.extend([-1] * 4)
    
    print("K-Means Uniform Scores")
    start = time.time()
    fusion = LSFusion(models)
    fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='k_means', neuron_importance_method='uniform')
    times.append(time.time() - start)
    torch.save(fused_model, save_dir_path/'fusion_results/k_means-uniform.pth')
    torch.cuda.empty_cache()
    
    print("K-Means Conductance Scores")
    start = time.time()
    fusion = LSFusion(models)
    fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='k_means', neuron_importance_method='conductance')
    times.append(time.time() - start)
    torch.save(fused_model, save_dir_path/'fusion_results/k_means-conductance.pth')
    torch.cuda.empty_cache()
    
    print("K-Means DeepLIFT-based Scores")
    start = time.time()
    fusion = LSFusion(models)
    fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='k_means', neuron_importance_method='deeplift')
    times.append(time.time() - start)
    torch.save(fused_model, save_dir_path/'fusion_results/k_means-deeplift.pth')
    torch.cuda.empty_cache()

    models_to_align = models[:-1]
    model_to_align_with = models[-1]

    alignment_strategy = {
        'mode': 'acts',  # 'acts' or 'wts'
        'acts': {
            'activations_dataset': activation_dataset,
            'activation_labels': activation_classes,
            'neuron_importance_method': 'uniform',
            'normalize_activations': True
        },
        'wts': {}
    }

    print("OT Fusion Uniform Scores")
    start = time.time()
    fusion = OTFusion(models_to_align, model_to_align_with)
    fused_model = fusion.fuse_models(alignment_strategy, handle_skip=True)
    times.append(time.time() - start)
    torch.save(fused_model, save_dir_path/f'fusion_results/ot-uniform-{alignment_strategy["mode"]}.pth')
    torch.cuda.empty_cache()

    print("OT Fusion Conductance-based Scores")
    start = time.time()
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'conductance'
    fused_model = fusion.fuse_models(alignment_strategy)
    times.append(time.time() - start)
    torch.save(fused_model, save_dir_path/'fusion_results/ot-conductance.pth')
    torch.cuda.empty_cache()

    print("OT Fusion DeepLIFT-based Scores")
    start = time.time()
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'deeplift'
    fused_model = fusion.fuse_models(alignment_strategy)
    times.append(time.time() - start)
    torch.save(fused_model, save_dir_path/'fusion_results/ot-deeplift.pth')
    torch.cuda.empty_cache()



    print(f'Fused models saved to {save_dir_path}')

    return times