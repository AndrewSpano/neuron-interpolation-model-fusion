import gc
import torch

from pathlib import Path

from models.vit import ViT
from utils.run_utils import add_safe_globals
from utils.fusion_utils import get_num_levels
from utils.dataset_utils import get_fusion_data
from fusion_algorithms.alf import ALFusionConfig
from utils.dataset_utils import get_classifier_head_weights
from main_scripts.experiments_code.fusion.fuse_all_utils import fuse_alf

from utils.dataset_utils import get_cifar100_dataset, sample_balanced_subset, get_cifar10_dataset, get_tiny_imagenet_dataset



def get_models(models_dir: Path, device) -> list[ViT]:
    # load pretrained models
    models = []
    for model_path in sorted(list(models_dir.glob('model*.pt'))):  # make sure to load models in order (model 0 has to be first!)
        model = torch.load(model_path, weights_only=True).to(device)
        models.append(model)
        print(f'Loaded {model_path} with model class: {model.__class__.__name__}.')

    return models


def get_alf_config(models: list[ViT], dataset_name: str, iid_split, device):

    num_classes = models[0].num_classes

    # use different params depending on the dataset
    if dataset_name == 'CIFAR-10':
        get_dataset_function = get_cifar10_dataset
    elif dataset_name == 'CIFAR-100':
        get_dataset_function = get_cifar100_dataset
    elif dataset_name == 'Tiny-ImageNet':
        get_dataset_function = get_tiny_imagenet_dataset
    

    _, val_dls, test_dl = get_dataset_function(len(models), iid_split, test_batch_size=1_000)

    # get the data
    original_dataset = val_dls[0].dataset.dataset
    indices_to_sample = models[0].train_indices + models[0].val_indices
    input_samples, labels = sample_balanced_subset(original_dataset, indices_to_sample, batch_size=6_000)

    # percentages of train samples per class, per model
    percentages = get_classifier_head_weights(models, original_dataset, num_classes=num_classes)
    percentages = torch.from_numpy(percentages).to(device)

    # per-level alf configs
    num_levels = get_num_levels(models)
    epochs_per_level = [100] * num_levels
    optimizer_per_level = ['adam'] * num_levels
    lr_per_level = [0.001] * num_levels
    weight_decay_per_level = [0.00001] * num_levels
    weight_perturbation_per_level = [1] * num_levels

    # fix last levels
    epochs_per_level[-1] = 100  # last level
    optimizer_per_level[-1] = 'adam'  # last level
    lr_per_level[-1] = 0.001  # last level

    # create fusion config
    # start fusing
    print('-' * 50)
    print(f'Fusing the trained models with Activation Gradient Fusion')
    fusion_config = ALFusionConfig(
        X=input_samples.to(device),
        y=labels.to(device),
        optimizer_per_level=optimizer_per_level,
        lr_per_level=lr_per_level,
        weight_decay_per_level=weight_decay_per_level,
        epochs_per_level=epochs_per_level,
        use_scheduler=False,
        align_skip_connections=False,
        init_kmeans_with_2nd_acts=False,
        normalize_acts_for_kmeans=True,
        neuron_importance_method='uniform',
        train_batch_size=32,
        max_patience_epochs=15,
        reset_level_parameters=list(range(num_levels)),
        weight_perturbation_per_level=weight_perturbation_per_level,
        use_weighted_loss=False,
        # cls_head_weights=percentages,
        # skip_levels=[0, 1, 2, 3, 4, 5, 6]
    )
    return fusion_config


def fuse_all_vits(
        base_dir: Path,
        iid_settings: list[bool],
        dataset_names: list[str],
        split_by: list[int],
        model_folders: list[int],
        num_groups: int = 0,
        group_size: int = 2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in dataset_names:
        for is_iid in iid_settings:
            for split_by in split_by:
                for mf in model_folders:
                    gc.collect()
                    torch.cuda.empty_cache()

                    # set up directory for current experiment
                    models_dir = base_dir/('iid' if is_iid else 'non-iid')/'ViTs'/dataset_name/f'split-by-{split_by}'/f'{mf}'
                    assert models_dir.exists(), f"Models directory {models_dir} does not exist."

                    models = get_models(models_dir, device)
                    fusion_config = get_alf_config(models, dataset_name, is_iid, device)

                    print(f'\nFusing {dataset_name} | split={split_by} | iid={is_iid} | model_folder={mf}')
                    fuse_alf(models, fusion_config, models_dir, dataset_name=dataset_name, num_groups=num_groups, num_per_group=group_size)


if __name__ == '__main__':
    add_safe_globals()
    base_dir = Path(__file__).parents[3].resolve().absolute()/'base-models'

    # in practice, use a subset and run on different servers to cover all combinations
    iid_settings = [True]
    dataset_names = ['CIFAR-100']
    split_by = [1]
    model_folders = ["combined_seeds_1_2_3_4_5"]

    fuse_all_vits(base_dir, iid_settings, dataset_names, split_by, model_folders, num_groups=30, group_size=4)
