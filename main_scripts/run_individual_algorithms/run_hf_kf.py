"""
Example usage:

$ python3 main_scripts/run_individual_algorithms/run_hf_kfv2.py \
    --model_savedir example-base-models/non-iid/VGGs/CIFAR-10/split-by-2/seed1 \
    --plot_path plots/hfkfvggs.png
"""

import torch
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from utils.plot_utils import bar_plot
from utils.run_utils import eval_model
from utils.general_utils import set_seed
from utils.run_utils import add_safe_globals
from fusion_algorithms.hf_kf import LSFusion
from utils.dataset_utils import (
    get_mnist_dataset,
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_tiny_imagenet_dataset,
    sample_balanced_subset
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LSFusion with basic CLI argument support.")
    parser.add_argument('--model_savedir', type=str, required=True, help='Directory containing base models')
    parser.add_argument('--num_fusion_samples', type=int, default=400, help='Number of samples to use for fusion (default: 400)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: auto-detect CUDA)')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='Optional path to save evaluation figure')
    parser.add_argument('--normalize_acts_for_kmeans', action='store_true',
                        help='Whether to normalize activations before k-means fusion')
    parser.add_argument('--normalize_neuron_importance', action='store_true',
                        help='Whether to normalize neuron importance scores')
    return parser.parse_args()


def get_dataset_fn_and_classes(savedir):
    path_str = savedir.as_posix()
    if 'MNIST' in path_str:
        return get_mnist_dataset, 10
    elif 'Tiny-ImageNet' in path_str:
        return get_tiny_imagenet_dataset, 200
    elif 'CIFAR-100' in path_str:
        return get_cifar100_dataset, 100
    else:
        return get_cifar10_dataset, 10


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    model_savedir = Path(args.model_savedir)
    print(f'Using device: {device}')

    # load models
    model_paths = sorted(model_savedir.glob('model_*.pt'))
    models = [torch.load(p, weights_only=True).to(device) for p in model_paths]
    for i, model in enumerate(models):
        print(f'Model {i} loaded from {model_paths[i]}. Model class: {model.__class__.__name__}.')

    get_dataset_fn, num_classes = get_dataset_fn_and_classes(model_savedir)
    train_dls, _, test_dl = get_dataset_fn(len(models), iid_split=False)

    dataset = train_dls[0].dataset.dataset
    indices = models[0].train_indices + models[0].val_indices
    X, y = sample_balanced_subset(dataset, indices, batch_size=args.num_fusion_samples)
    X, y = X.to(device), y.to(device)

    print('-' * 50)
    print(f'Fusing the trained models with LSFusion Gradient Fusion')

    def run_and_save(method_name, fusion_type, norm_acts, norm_neuron_importance):
        print(f'{fusion_type.upper()}: {method_name.title()} Scores')
        fusion = LSFusion(models)
        fused_model = fusion.fuse(X, y,
                                  fusion_method=fusion_type,
                                  neuron_importance_method=method_name,
                                  norm_acts=norm_acts,
                                  norm_neuron_importance=norm_neuron_importance)
        fusion_abbrev = 'hf' if fusion_type == 'hungarian' else 'kf'
        out_name = f'{fusion_abbrev}-{len(models)}-{method_name}.pt'
        torch.save(fused_model, model_savedir / out_name)
        torch.cuda.empty_cache()

    if len(models) == 2:
        for method in ['uniform', 'conductance', 'deeplift']:
            run_and_save(method, 'hungarian',
                         norm_acts=False,
                         norm_neuron_importance=args.normalize_neuron_importance)

    for method in ['uniform', 'conductance', 'deeplift']:
        run_and_save(method, 'k_means',
                     norm_acts=args.normalize_acts_for_kmeans,
                     norm_neuron_importance=args.normalize_neuron_importance)

    print(f'Fused models saved to {model_savedir}')

    # evaluation + plotting
    all_model_paths = list(model_savedir.glob('model*.pt')) + list(model_savedir.glob('hf-*.pt')) + list(model_savedir.glob('kf-*.pt'))
    all_models = [torch.load(p, weights_only=True).to(device) for p in all_model_paths]
    model_names = [p.stem for p in all_model_paths]
    stats = [eval_model(m, test_dl, device=device, num_classes=num_classes) for m in tqdm(all_models, desc='Evaluating all models')]

    fig, axs = plt.subplots(2, 2, figsize=(max(25, len(model_names)*4), 20))
    bar_plot(axs[0, 0], 'loss', stats, model_names, is_01_metric=False)
    bar_plot(axs[0, 1], 'f1', stats, model_names)
    bar_plot(axs[1, 0], 'accuracy', stats, model_names)
    bar_plot(axs[1, 1], 'roc_auc', stats, model_names)

    if args.plot_path:
        fig.savefig(args.plot_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to {args.plot_path}')


if __name__ == '__main__':
    add_safe_globals()
    main()
