import gc
import torch

from pathlib import Path


from utils.run_utils import add_safe_globals
from utils.dataset_utils import get_fusion_data
from fusion_algorithms.alf import ALFusionConfig
from utils.dataset_utils import get_classifier_head_weights
from utils.fusion_utils import get_num_levels
from models.mlp import MLPNet

from main_scripts.experiments_code.fusion.fuse_all_utils import fuse_alf, fuse_hf, fuse_kf, fuse_gr, fuse_otf

def get_alf_config(models: list[MLPNet], dataset_name: str, device):

    num_classes = models[0].output_size

    # use different params depending on the dataset
    if dataset_name == 'CIFAR-10':
        fusion_dataset_size = 400
        num_epochs = 25
        default_lr = 1e-4
        last_level_optimizer = 'adam'
        last_level_epochs = 150
        use_percentages = True
    elif dataset_name == 'CIFAR-100':
        fusion_dataset_size = 1_000
        num_epochs = 25
        default_lr = 1e-4
        last_level_optimizer = 'adam'
        last_level_epochs = 100
        use_percentages = False
    elif dataset_name == 'Tiny-ImageNet':
        raise ValueError(f"Not yet implemented for {dataset_name}.")
    elif dataset_name == 'MNIST':
        fusion_dataset_size = 400
        num_epochs = 25
        default_lr = 1e-4
        last_level_optimizer = 'adam'
        last_level_epochs = 150
        use_percentages = True
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # get the data
    indices_to_sample_from = models[0].train_indices + models[0].val_indices
    input_samples, labels, dataset, num_classes = get_fusion_data(dataset_name, indices_to_sample_from,
                                                                  batch_size=fusion_dataset_size)

    # percentages of train samples per class, per model
    percentages = None
    if use_percentages:
        percentages = get_classifier_head_weights(models, dataset, num_classes=num_classes)
        percentages = torch.from_numpy(percentages).to(device)

    # per-level alf configs
    num_levels = get_num_levels(models)
    epochs_per_level = [num_epochs] * num_levels
    optimizer_per_level = ['adam'] * num_levels
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
    num_pairs: int = 0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    for dataset_name in dataset_names:
        for is_iid in iid_settings:
            for split_by in split_by:
                for mf in model_folders:
                    gc.collect()
                    torch.cuda.empty_cache()

                    # set up directory for current experiment
                    models_dir = base_dir/('iid' if is_iid else 'non-iid')/'MLPs'/dataset_name/f'split-by-{split_by}'/mf
                    assert models_dir.exists(), f"Models directory {models_dir} does not exist."

                    print(f'\nFusing {dataset_name} | split={split_by} | iid={is_iid} | model_folder={mf}')

                    # load pretrained models
                    models = []
                    for model_path in sorted(list(models_dir.glob('model*.pt'))):  # make sure to load models in order (model 0 has to be first!)
                        model = torch.load(model_path, weights_only=True).to(device)
                        models.append(model)
                        print(f'Loaded {model_path} with model class: {model.__class__.__name__}.')

                    fuse_hf(models, models_dir, dataset_name, num_pairs=num_pairs)
                    fuse_kf(models, models_dir, dataset_name, num_pairs=num_pairs)
                    fuse_gr(models, models_dir, dataset_name, num_pairs=num_pairs)
                    fuse_otf(models, models_dir, dataset_name, num_pairs=num_pairs)
                    
                    alf_config = get_alf_config(models, dataset_name, device)
                    fuse_alf(models, fusion_config=alf_config, dataset_name=dataset_name, model_savedir=models_dir, num_pairs=num_pairs)

if __name__ == '__main__':
    add_safe_globals()
    base_dir = Path(__file__).parents[3].resolve().absolute()/'base-models'

    # Fine tuning
    # iid_settings = [True]
    # dataset_names = ['MNIST']
    # split_by = [1]
    # seeds = [1,2,3,4,5,6,7,8]

    # fuse_all(base_dir, iid_settings, dataset_names, split_by, seeds, num_pairs=False)

    # Non IID splits
    iid_settings = [False]
    dataset_names = ['MNIST']
    split_by = [2, 4, 8]
    model_folders = ["seed1","seed2","seed3","seed4","seed5"]

    fuse_all(base_dir, iid_settings, dataset_names, split_by, model_folders, num_pairs=0)
