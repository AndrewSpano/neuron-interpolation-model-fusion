import torch
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from utils.plot_utils import bar_plot
from utils.general_utils import set_seed
from utils.run_utils import add_safe_globals, eval_model
from fusion_algorithms.otfusion import OTFusion
from utils.dataset_utils import (
    get_mnist_dataset,
    get_cifar10_dataset,
    get_cifar100_dataset,
    get_tiny_imagenet_dataset,
    sample_balanced_subset
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run OT Fusion with minimal CLI arguments.')

    parser.add_argument('--model_savedir', type=str, required=True, help='Path to base models directory')
    parser.add_argument('--num_fusion_samples', type=int, default=400, help='Number of samples to use for fusion (default: 400)')
    parser.add_argument('--num_models', type=int, help='Number of models to fuse')
    parser.add_argument('--iid_split', action='store_true', help='Use IID split (default: non-IID)')
    parser.add_argument('--seed', type=int, default=3409, help='Random seed (default: 3409)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: auto)')
    parser.add_argument('--normalize_activations', action='store_true',
                        help='Normalize activations before fusion (default: False)')
    parser.add_argument('--plot_path', type=str, default=None,
                        help='Optional path to save evaluation plot')
    return parser.parse_args()


def load_models(model_savedir, num_models, device, iid_split):
    if iid_split:
        paths = list(model_savedir.glob('**/model*.pt'))
        models = []
        for path in paths[:num_models]:
            model = torch.load(path, weights_only=True).to(device)
            models.append(model)
            print(f'Model loaded from {path}. Class: {model.__class__.__name__}')
    else:
        models = []
        for idx in range(num_models):
            path = model_savedir / f'model_{idx}.pt'
            model = torch.load(path, weights_only=True, map_location=device)
            models.append(model)
            print(f'Model {idx} loaded from {path}. Class: {model.__class__.__name__}')
    return models


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

    if args.num_models is None:
        model_paths = sorted(model_savedir.glob('model_*.pt'))
        args.num_models = len(model_paths)
        print(f'No num_models specified, using all {args.num_models} models found in {model_savedir}')

    print(f'Using device: {device}')
    models = load_models(model_savedir, args.num_models, device, args.iid_split)

    dataset_fn, _ = get_dataset_fn_and_classes(model_savedir)
    _, val_dls, test_dl = dataset_fn(len(models), args.iid_split, test_batch_size=1000)
    num_classes = 10 if 'MNIST' in model_savedir.as_posix() else 100 if 'CIFAR-100' in model_savedir.as_posix() else 10

    base_dataset = val_dls[0].dataset.dataset
    indices = models[1].train_indices + models[1].val_indices
    X, y = sample_balanced_subset(base_dataset, indices, batch_size=args.num_fusion_samples)
    X, y = X.to(device), y.to(device)

    print('-' * 50)
    print('Running OT Fusion...')

    models_to_align = models[:-1]
    model_to_align_with = models[-1]

    def run_otfusion(mode, method, rescale=None):
        print(f"{mode.upper()} mode - {method.title()} Scores", f"(rescale min={rescale})" if rescale else "")
        fusion = OTFusion(models_to_align, model_to_align_with)
        strategy = {
            'mode': mode,
            mode: {
                'activations_dataset': X,
                'activation_labels': y,
                'neuron_importance_method': method,
                'normalize_activations': args.normalize_activations,
                'rescale_min_importance': rescale
            }
        }
        fused = fusion.fuse_models(strategy, handle_skip=True if mode == 'acts' else False)
        name = f'ot-{method}-{mode}' + (f'-{rescale}' if rescale else '') + '.pt'
        torch.save(fused, model_savedir / name)
        print(f"Saved to {model_savedir / name}")
        torch.cuda.empty_cache()

    for mode in ['acts', 'wts']:
        for method in ['uniform', 'conductance', 'deeplift']:
            run_otfusion(mode, method)
        for rescale_val in [0.1, 0.5]:
            run_otfusion(mode, 'conductance', rescale=rescale_val)

    print(f"All fused models saved to {model_savedir}")

    if args.plot_path:
        all_model_paths = list(model_savedir.glob('model*.pt')) + list(model_savedir.glob('ot-*.pt'))
        all_models = [torch.load(p, weights_only=True).to(device) for p in all_model_paths]
        model_names = [p.stem for p in all_model_paths]
        stats = [eval_model(m, test_dl, device=device, num_classes=num_classes)
                 for m in tqdm(all_models, desc='Evaluating models')]

        fig, axs = plt.subplots(2, 2, figsize=(max(25, len(model_names)*4), 20))
        bar_plot(axs[0, 0], 'loss', stats, model_names, is_01_metric=False)
        bar_plot(axs[0, 1], 'f1', stats, model_names)
        bar_plot(axs[1, 0], 'accuracy', stats, model_names)
        bar_plot(axs[1, 1], 'roc_auc', stats, model_names)

        fig.savefig(args.plot_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {args.plot_path}')


if __name__ == '__main__':
    add_safe_globals()
    main()
