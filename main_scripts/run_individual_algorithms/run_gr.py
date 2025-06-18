import argparse
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from utils.plot_utils import bar_plot
from utils.general_utils import set_seed
from fusion_algorithms.git_rebasin import GitRebasin
from utils.run_utils import eval_model, add_safe_globals
from utils.dataset_utils import (
    get_mnist_dataset,
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_tiny_imagenet_dataset,
    sample_balanced_subset
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run GitRebasin fusion with CLI argument support.')
    parser.add_argument('--model_savedir', type=str, required=True, help='Directory containing base models')
    parser.add_argument('--num_fusion_samples', type=int, default=400, help='Number of samples to use for fusion (default: 400)')
    parser.add_argument('--iid_split', action='store_true', help='Use IID data split (default: non-IID)')
    parser.add_argument('--seed', type=int, default=3409, help='Random seed (default: 3409)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: auto)')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='Optional path to save evaluation plot after fusion')
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

    dataset_fn, _ = get_dataset_fn_and_classes(model_savedir)
    _, val_dls, test_dl = dataset_fn(2, args.iid_split, test_batch_size=128)

    model_paths = sorted(model_savedir.glob('model*.pt'))
    models = [torch.load(p, map_location=device, weights_only=True).eval().to(device) for p in model_paths]
    for p in model_paths:
        print(f'Model loaded from {p}')
    if len(models) > 2:
        print(f'Warning: More than 2 models found ({len(models)}); using the first two...')
    models = models[:2]

    print('-' * 50)
    print(f'Fusing models with GitRebasin (Hungarian Matching)...')

    base_dataset = val_dls[0].dataset.dataset
    indices = models[1].train_indices + models[1].val_indices
    X, y = sample_balanced_subset(base_dataset, indices, batch_size=args.num_fusion_samples)
    X, y = X.to(device), y.to(device)

    for method in ['uniform', 'conductance', 'deeplift']:
        print(f"Fusing with {method.title()} Scores")
        git_rebasin = GitRebasin(models[0], models[1])
        fused_model = git_rebasin.fuse(X, y, neuron_importance_method=method, handle_skip=False)
        fused_path = model_savedir/f'gr-{method}.pt'
        torch.save(fused_model, fused_path)
        print(f'Fused model saved to {fused_path}')
        torch.cuda.empty_cache()

    if args.plot_path:
        base_paths = sorted(model_savedir.glob('model*.pt'))
        fused_paths = sorted(model_savedir.glob('gr-*.pt'))
        all_model_paths = base_paths + fused_paths

        all_models = [torch.load(p, weights_only=True).to(device) for p in all_model_paths]
        model_names = [p.stem for p in all_model_paths]

        num_classes = 10 if 'MNIST' in model_savedir.as_posix() else 100
        stats = [eval_model(m, test_dl, device=device, num_classes=num_classes)
                 for m in tqdm(all_models, desc='Evaluating models')]

        fig, axs = plt.subplots(2, 2, figsize=(max(25, len(model_names) * 4), 20))
        bar_plot(axs[0, 0], 'loss', stats, model_names, is_01_metric=False)
        bar_plot(axs[0, 1], 'f1', stats, model_names)
        bar_plot(axs[1, 0], 'accuracy', stats, model_names)
        bar_plot(axs[1, 1], 'roc_auc', stats, model_names)
        fig.savefig(args.plot_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {args.plot_path}')


if __name__ == '__main__':
    add_safe_globals()
    main()
