import torch

from pathlib import Path

from utils.general_utils import set_seed
from utils.train_utils import ModelPathReorganizer
from utils.cifar_train_utils import get_cifar_with_advanced_transforms

def prepare_data(dataset_name, args, num_splits, is_iid, is_sharded, val_ratio):
    if dataset_name.startswith('CIFAR'):
        is_cifar_10 = dataset_name == 'CIFAR-10'
        train_dls, val_dls, test_dl = get_cifar_with_advanced_transforms(num_splits, is_cifar_10, args, is_iid, is_sharded, val_ratio=val_ratio)
    elif dataset_name == 'Tiny-ImageNet':
        from utils.dataset_utils import get_tiny_imagenet_dataset
        train_dls, val_dls, test_dl = get_tiny_imagenet_dataset(
            args=args,
            num_splits=num_splits, 
            sharded_split=is_sharded,
            iid_split=is_iid, 
            train_batch_size=args.batch_size,
            test_batch_size=args.eval_batch_size,
            val_ratio=val_ratio)
    elif dataset_name == 'MNIST':
        from utils.dataset_utils import get_mnist_dataset
        train_dls, val_dls, test_dl = get_mnist_dataset(
            num_splits=num_splits,
            iid_split=is_iid,
            sharded_split=is_sharded,
            train_batch_size=args.batch_size,
            test_batch_size=args.eval_batch_size,
            val_ratio=val_ratio)
    elif dataset_name == 'BloodMNIST':
        from utils.dataset_utils import get_bloodmnist_dataset
        train_dls, val_dls, test_dl = get_bloodmnist_dataset(
            num_splits=num_splits,
            iid_split=is_iid,
            sharded_split=is_sharded,
            train_batch_size=args.batch_size,
            test_batch_size=args.eval_batch_size,
            val_ratio=val_ratio)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return train_dls, val_dls, test_dl



def run_experiment(
    seeds,
    iid_settings,
    datasets,
    splits,
    use_val_for_model_selection,
    create_model_func,
    train_model_func,
    get_args,
    model_name,
    basedir
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for dataset_name in datasets:
        for is_iid, is_sharded in iid_settings:
            for split_by in splits:
                if is_iid and split_by > 1:  # only one split for iid
                    continue
                if not is_iid and split_by == 1:  # at least two splits for non-iid
                    continue
                for seed in seeds:

                    # fix random seed for reproducibility
                    set_seed(seed)
                    args = get_args(dataset_name, split_by)
                    args.seed = seed
                    print(f"Running {dataset_name} | split={split_by} | iid={is_iid} | sharded={is_sharded} | seed={seed}")

                    # set up directory for current experiment
                    iid_name = 'iid' if is_iid else ('sharded' if is_sharded else 'non-iid')
                    models_dir = basedir/iid_name/f'{model_name}s'/dataset_name/f'split-by-{split_by}'/f'seed{seed}'
                    models_dir.mkdir(parents=True, exist_ok=True)

                    # object to rearrange model files based on accuracy
                    reorganizer = ModelPathReorganizer(models_dir, metric_name='accuracy', reverse=True)

                    # prepare data
                    val_ratio = 0.05 if use_val_for_model_selection else 0.0
                    train_dls, val_dls, test_dl = prepare_data(dataset_name, args, num_splits=split_by, is_iid=is_iid, is_sharded=is_sharded, val_ratio=val_ratio)

                    # create/train/select base models for current setup
                    for idx in range(split_by):
                        print(f'Training model {idx} with {len(train_dls[idx].dataset)} training samples and {len(val_dls[idx].dataset)} validation samples')
                        model = create_model_func(args, train_dls[idx], val_dls[idx])
                        model.to(device)
                        savepath = models_dir/f'model_{idx}.pt'
                        results, savepath = train_model_func(
                            model=model,
                            args=args,
                            savepath=savepath,
                            train_dl=train_dls[idx],
                            val_dl=val_dls[idx],
                            test_dl=test_dl,
                            use_val_for_model_selection=use_val_for_model_selection
                        )
                        reorganizer.update(savepath, results["accuracy"])

                    # rename files so that model indices (0, 1, 2, ...) are in order of accuracy
                    reorganizer.sort_and_rename()
                    print(f'Reorganized models saved in {models_dir}')
