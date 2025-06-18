import gc
import torch

from pathlib import Path

from models.vgg import VGG
from utils.general_utils import set_seed
from utils.run_utils import add_safe_globals
from utils.fusion_utils import get_num_levels
from utils.dataset_utils import get_fusion_data
from fusion_algorithms.alf import ALFusionConfig
from utils.dataset_utils import get_classifier_head_weights
from main_scripts.experiments_code.fusion.fuse_all_utils import fuse_alf, fuse_hf, fuse_kf, fuse_gr, fuse_otf


SEED = 3409


def get_alf_config(models: list[VGG], dataset_name: str, iid_setting, device):
    num_models = len(models)

    # use different params depending on the dataset
    use_percentages = False
    if dataset_name == 'CIFAR-10':
        fusion_dataset_size = 400
        default_optimizer = 'adam' if (num_models == 2 or iid_setting == 'sharded') else 'sgd'
        num_epochs = 100 if (num_models == 2 or iid_setting == 'sharded') else 50
        default_lr = 1e-3 if (num_models == 2 or iid_setting == 'sharded') else 1e-4
        last_level_optimizer = 'adam'
        last_level_epochs = 150
        use_percentages = True
    elif dataset_name == 'CIFAR-100':
        fusion_dataset_size = 1_000
        default_optimizer = 'adam' if (num_models == 2 or iid_setting == 'sharded') else 'sgd'
        num_epochs = 100 if (num_models == 2 or iid_setting == 'sharded') else 50
        default_lr = 1e-4 if (num_models == 2 or iid_setting == 'sharded') else 1e-3
        last_level_optimizer = 'adam' if (num_models == 2 or iid_setting == 'sharded') else 'sgd'
        last_level_epochs = 100
        use_percentages = False
    elif dataset_name == 'Tiny-ImageNet':
        raise NotImplementedError(f"Not yet implemented for {dataset_name}.")
    elif dataset_name == 'MNIST':
        raise ValueError(f"MNIST not supported for VGGs.")
    elif dataset_name == 'BloodMNIST':
        fusion_dataset_size = 400
        default_optimizer = 'adam'
        num_epochs = 100
        default_lr = 1e-3
        last_level_optimizer = 'adam'
        last_level_epochs = 100
        use_percentages = False
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # get the data
    indices_to_sample_from = models[0].train_indices + models[0].val_indices
    input_samples, labels, dataset, num_classes = get_fusion_data(dataset_name, indices_to_sample_from,
                                                                  batch_size=fusion_dataset_size)
    input_samples, labels = input_samples.to(device), labels.to(device)
    
    # percentages of train samples per class, per model
    percentages = get_classifier_head_weights(models, dataset, num_classes=num_classes)
    percentages = torch.from_numpy(percentages).to(device)
    if not use_percentages:
        percentages = None

    # per-level alf configs
    num_levels = get_num_levels(models)
    epochs_per_level = [num_epochs] * num_levels
    optimizer_per_level = [default_optimizer] * num_levels
    lr_per_level = [default_lr] * num_levels
    weight_decay_per_level = [0.0001] * num_levels

     # fix last levels
    epochs_per_level[-1] = last_level_epochs  # last level
    optimizer_per_level[-1] = last_level_optimizer  # last level

    # create fusion config
    fusion_config = ALFusionConfig(
        X=input_samples.to(device),
        y=labels.to(device),
        optimizer_per_level=optimizer_per_level,
        lr_per_level=lr_per_level,
        weight_decay_per_level=weight_decay_per_level,
        epochs_per_level=epochs_per_level,
        use_scheduler=False,
        neuron_importance_method='uniform',
        train_batch_size=32,
        max_patience_epochs=15,
        use_weighted_loss=False,
        cls_head_weights=percentages,
        verbose=False
    )
    return fusion_config


def fuse_all(
    base_dir: Path,
    iid_settings: list[bool],
    dataset_names: list[str],
    split_by: list[int],
    model_folders: list[str],
    num_groups: int = 0,
    group_size: int = 2
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in dataset_names:
        for iid_setting in iid_settings:
            for split_by in split_by:
                for mf in model_folders:
                    gc.collect()
                    torch.cuda.empty_cache()

                    # set up directory for current experiment
                    models_dir = base_dir/iid_setting/'VGGs'/dataset_name/f'split-by-{split_by}'/mf
                    assert models_dir.exists(), f"Models directory {models_dir} does not exist."

                    print(f'\nFusing {dataset_name} | split={split_by} | iid_setting={iid_setting} | model_folder={mf}')

                    # load pretrained models
                    models = []
                    for model_path in sorted(list(models_dir.glob('model*.pt'))):  # make sure to load models in order (model 0 has to be first!)
                        model = torch.load(model_path, weights_only=True).to(device)
                        models.append(model)
                        print(f'Loaded {model_path} with model class: {model.__class__.__name__}.')

                    alf_config = get_alf_config(models, dataset_name, iid_setting, device)
                    X, y = alf_config.X, alf_config.y
                    fuse_hf(models, models_dir, dataset_name, X, y, num_groups=num_groups, num_per_group=group_size)
                    fuse_kf(models, models_dir, dataset_name, X, y, num_groups=num_groups, num_per_group=group_size)
                    fuse_gr(models, models_dir, dataset_name, X, y, num_groups=num_groups, num_per_group=group_size)
                    fuse_otf(models, models_dir, dataset_name, X, y, num_groups=num_groups, num_per_group=group_size)
                    fuse_alf(models, fusion_config=alf_config, dataset_name=dataset_name, model_savedir=models_dir, num_groups=num_groups, num_per_group=group_size)


if __name__ == '__main__':
    add_safe_globals()
    set_seed(SEED)
    base_dir = Path(__file__).parents[3].resolve().absolute()/'example-base-models'

    # Non IID splits
    iid_settings = ['non-iid']
    dataset_names = ['CIFAR-10']
    split_by = [4]
    model_folders = ["seed1"]

    fuse_all(base_dir, iid_settings, dataset_names, split_by, model_folders, num_groups=0, group_size=2)
