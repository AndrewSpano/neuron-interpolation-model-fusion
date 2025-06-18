import torch
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

from utils.plot_utils import bar_plot
from utils.run_utils import eval_model
from utils.general_utils import set_seed
from utils.run_utils import add_safe_globals
from utils.fusion_utils import get_num_levels
from fusion_algorithms.alf import ALFusion, ALFusionConfig
from utils.dataset_utils import (
    get_mnist_dataset,
    get_cifar10_dataset,
    get_tiny_imagenet_dataset,
    get_cifar100_dataset,
    sample_balanced_subset,
    get_classifier_head_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ALFusion with full CLI config support.')

    parser.add_argument('--model_savedir', type=str, required=True, help='Directory containing base models')
    parser.add_argument('--num_models', type=int, required=False,
                        help='Number of models to fuse. Indices 0-{num_models-1} will be used. If not provided, '
                             'all models in the directory will be used.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the fusion on (default: auto-detect based on availability)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    parser.add_argument('--num_fusion_samples', type=int, required=True, help='Number of samples to use for fusion.')

    parser.add_argument('--save_dir', type=str, help='Directory to save fused models (default: same as model_savedir)')
    parser.add_argument('--plot_path', type=str, help='Optional path to save evaluation plot')

    parser.add_argument('--optimizer_per_level', type=str, default='adam',
                        help='Optimizer to use per level. Can be a comma-separated list of optimizers, or just a single '
                             'one to be used in all the levels.')
    parser.add_argument('--lr_per_level', type=str, default='0.001',
                        help='Learning rate per level. Can be a comma-separated list of learning rates, or just a single one.')
    parser.add_argument('--weight_decay_per_level', type=str, default='0.0001',
                        help='Weight decay per level. Can be a comma-separated list of weight decays, or just a single one.')
    parser.add_argument('--epochs_per_level', type=str, default='100',
                        help='Number of epochs per level. Can be a comma-separated list of epochs, or just a single one.')
    parser.add_argument('--weight_perturbation_per_level', type=str, default='1',
                        help='Weight perturbation per level. Can be a comma-separated list of perturbations, or just a single one. '
                        'Note that perturbed weights are computed as: (1 - eps) * original_weights + eps * random_weights.')

    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use a learning rate scheduler for the optimizer when minimizing the approximation error. '
                             'If not set, no scheduler will be used.')
    parser.add_argument('--normalize_neuron_importances', action='store_true',
                        help='Normalize neuron importances to sum to 1. If not set, they will be used as is.')
    parser.add_argument('--align_skip_connections', action='store_true',
                        help='Align skip connections in the fused model. If not set, skip connections will not be aligned.')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Batch size for training the fused model. Default is 32.')
    parser.add_argument('--max_patience_epochs', type=int, default=15,
                        help='Maximum number of epochs to wait for improvement before stopping training. Default is 15.')
    parser.add_argument('--normalize_acts_for_kmeans', action='store_true',
                        help='Normalize activations before running k-means clustering. If not set, activations will not be normalized.')
    parser.add_argument('--num_kmeans_runs', type=int, default=1,
                        help='Number of k-means runs to perform for clustering neurons. Default is 1.')
    parser.add_argument('--init_kmeans_with_2nd_acts', action='store_true',
                        help='Initialize k-means with the second set of activations. If not set, k-means will be initialized with the first.')
    parser.add_argument('--use_weighted_loss', action='store_true',
                        help='Use weighted MSE for the approximation error. If not set, plain MSE will be used. ')

    parser.add_argument('--run_last_layer_kd', action='store_true',
                        help='Use knowledge distillation on the last layer of the best model')
    parser.add_argument('--use_kl_kd_on_head', action='store_true',
                        help='Use KL divergence for knowledge distillation on the last layer. If not set, MSE will be used.')
    parser.add_argument('--kd_temperature', type=float, default=2.0,
                        help='Temperature for knowledge distillation. Default is 2.0.')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def parse_list_arg(arg_str, item_type=float):
    return [item_type(x.strip()) for x in arg_str.split(',')] if arg_str else None


def get_dataset_fn(model_savedir):
    path_str = model_savedir.as_posix()
    if 'MNIST' in path_str:
        return get_mnist_dataset, 10
    elif 'Tiny-ImageNet' in path_str:
        return get_tiny_imagenet_dataset, 200
    elif 'CIFAR-100' in path_str:
        return get_cifar100_dataset, 100
    else:
        return get_cifar10_dataset, 10


def load_models(model_savedir, device, num_models):
    models = []
    for i, path in enumerate(sorted(model_savedir.glob('**/model*.pt'))):
        if i >= num_models:
            break
        model = torch.load(path, weights_only=True).to(device)
        models.append(model)
        print(f'Model loaded from {path}. Class: {model.__class__.__name__}')
    is_iid = 'iid' in model_savedir.as_posix()
    return models, is_iid


def build_fusion_config(args, models, dataset, device):
    num_levels = get_num_levels(models)
    cls_head_weights = get_classifier_head_weights(models, dataset, num_classes=args.num_models)
    cls_head_weights = torch.from_numpy(cls_head_weights).to(device) if args.use_weighted_loss else None

    def expand_param(p, cast_type):
        val = parse_list_arg(getattr(args, p), cast_type)
        return val * num_levels if len(val) == 1 else val

    return ALFusionConfig(
        X=None,  # will be filled in after sampling
        y=None,
        optimizer_per_level=expand_param('optimizer_per_level', str),
        lr_per_level=expand_param('lr_per_level', float),
        weight_decay_per_level=expand_param('weight_decay_per_level', float),
        epochs_per_level=expand_param('epochs_per_level', int),
        weight_perturbation_per_level=expand_param('weight_perturbation_per_level', float),
        use_scheduler=args.use_scheduler,
        neuron_importance_method='uniform',
        normalize_neuron_importances=args.normalize_neuron_importances,
        align_skip_connections=args.align_skip_connections,
        train_batch_size=args.train_batch_size,
        max_patience_epochs=args.max_patience_epochs,
        normalize_acts_for_kmeans=args.normalize_acts_for_kmeans,
        num_kmeans_runs=args.num_kmeans_runs,
        init_kmeans_with_2nd_acts=args.init_kmeans_with_2nd_acts,
        use_weighted_loss=args.use_weighted_loss,
        cls_head_weights=cls_head_weights,
        use_kd_on_head=False,
        kd_temperature=-1,
        verbose=args.verbose
    )


def plot_eval_results(models, model_names, test_dl, device, num_classes, save_path):
    stats = [eval_model(m, test_dl, device=device, num_classes=num_classes) for m in tqdm(models, desc='Evaluating models')]
    fig, axs = plt.subplots(2, 2, figsize=(max(25, len(models) * 4), 20))
    bar_plot(axs[0, 0], 'loss', stats, model_names, is_01_metric=False)
    bar_plot(axs[0, 1], 'f1', stats, model_names)
    bar_plot(axs[1, 0], 'accuracy', stats, model_names)
    bar_plot(axs[1, 1], 'roc_auc', stats, model_names)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to {save_path}')


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    model_savedir = Path(args.model_savedir)
    save_dir = Path(args.save_dir) if args.save_dir else model_savedir
    if args.num_models is None:
        args.num_models = len(list(model_savedir.glob('**/model*.pt')))

    models, iid_split = load_models(model_savedir, device, args.num_models)
    dataset_fn, num_classes = get_dataset_fn(model_savedir)
    _, val_dls, test_dl = dataset_fn(len(models), iid_split, test_batch_size=128)

    base_dataset = val_dls[0].dataset.dataset
    X, y = sample_balanced_subset(base_dataset, batch_size=args.num_fusion_samples)
    X, y = X.to(device), y.to(device)

    fusion_config = build_fusion_config(args, models, base_dataset, device)
    fusion_config.X = X
    fusion_config.y = y

    for method in ['uniform', 'conductance', 'deeplift']:
        print(f'Running ALFusion with {method}')
        fusion_config.neuron_importance_method = method
        model_path = save_dir/f'alf-{args.num_models}-{method}.pt'
        fused_model = ALFusion(models).fuse(fusion_config)
        torch.save(fused_model, model_path)
        print(f'Saved fused model to {model_path}')
        del fused_model
        torch.cuda.empty_cache()

    if args.run_last_layer_kd:
        print('Running ALFusion with last layer knowledge distillation')
        fusion_config.use_kd_on_head = args.use_kl_kd_on_head
        fusion_config.kd_temperature = args.kd_temperature
        fusion_config.skip_levels = list(range(get_num_levels(models) - 1))
        model_path = save_dir/f'alf-{args.num_models}-logit-kd.pt'
        fused_model = ALFusion(models).fuse(fusion_config)
        torch.save(fused_model, model_path)
        print(f'Saved last layer KD model to {model_path}')
        del fused_model
        torch.cuda.empty_cache()

    if args.plot_path:
        all_model_paths = list(model_savedir.glob('model*.pt')) + list(save_dir.glob('alf-*.pt'))
        all_models = [torch.load(p, weights_only=True).to(device) for p in all_model_paths]
        model_names = [p.stem for p in all_model_paths]
        plot_eval_results(all_models, model_names, test_dl, device, num_classes, args.plot_path)


if __name__ == '__main__':
    add_safe_globals()
    main()
