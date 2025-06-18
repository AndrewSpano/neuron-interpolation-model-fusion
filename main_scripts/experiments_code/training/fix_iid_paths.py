import torch

from pathlib import Path

from utils.train_utils import ModelPathReorganizer
from utils.run_utils import add_safe_globals, eval_model


def refactor_paths(base_dir: Path, test_dl, device, num_classes):
    add_safe_globals()
    
    seed_dirs = list(base_dir.glob('seed*'))
    all_seed_numbers = [int(seed_dir.name.split('seed')[1]) for seed_dir in seed_dirs]
    all_seed_numbers.sort()

    new_unified_dir = base_dir/f'combined_seeds_{"_".join(map(str, all_seed_numbers))}'
    new_unified_dir.mkdir(parents=True, exist_ok=True)
    reorganizer = ModelPathReorganizer(new_unified_dir, metric_name='accuracy', reverse=True)

    for number in all_seed_numbers:
        seed_dir = base_dir/f'seed{number}'
        model_path = seed_dir/f'model_0.pt'  # always assumes 1 model per seed (in iid setting)
        model = torch.load(model_path, weights_only=True, map_location=device)
        test_results = eval_model(model, test_dl, device, num_classes=num_classes)

        new_model_path = new_unified_dir/f'model_{number}.pt'
        model_path.rename(new_model_path)
        reorganizer.update(new_model_path, test_results['accuracy'])
        print(f"Moved {model_path} to {new_model_path}")

    reorganizer.sort_and_rename()
    print(f'Reorganized models saved in {new_unified_dir}')

    # delete old seed directories
    for number in all_seed_numbers:
        seed_dir = base_dir/f'seed{number}'
        if seed_dir.exists():
            seed_dir.rmdir()
            print(f"Deleted {seed_dir}")


if __name__ == "__main__":
    datasets = ['Tiny-ImageNet']
    from main_scripts.experiments_code.individual_train_scripts.vit_args import get_args
    args = get_args(datasets[0], 1)

    def prepare_data(dataset_name, args, num_splits, is_iid, val_ratio, is_sharded_split=False):
        if dataset_name.startswith('CIFAR'):
            is_cifar_10 = dataset_name == 'CIFAR-10'
            from utils.cifar_train_utils import get_cifar_with_advanced_transforms
            train_dls, val_dls, test_dl = get_cifar_with_advanced_transforms(num_splits, is_cifar_10, args, is_iid, val_ratio=val_ratio, is_sharded_split=is_sharded_split)
        elif dataset_name == 'Tiny-ImageNet':
            from utils.dataset_utils import get_tiny_imagenet_dataset
            train_dls, val_dls, test_dl = get_tiny_imagenet_dataset(num_splits, is_iid, args, val_ratio=val_ratio)
        else:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        
        return train_dls, val_dls, test_dl
    base_dir = Path("base-models/iid/ViTs/Tiny-ImageNet/split-by-1")
    _, _, test_dl = prepare_data(datasets[0], args, num_splits=1, is_iid=True, val_ratio=0.0, is_sharded_split=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 200  # replace with actual number of classes

    refactor_paths(base_dir, test_dl, device, num_classes)
