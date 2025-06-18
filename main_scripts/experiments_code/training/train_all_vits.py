import torch

from pathlib import Path

from models.vit import ViT
from main_scripts.experiments_code.training.fix_iid_paths import refactor_paths
from main_scripts.experiments_code.individual_train_scripts.vit_args import get_args
from main_scripts.experiments_code.training.model_training_utils import run_experiment, prepare_data
from main_scripts.experiments_code.individual_train_scripts.vit_generic_train import train_vit




def create_vit(args, train_dl, val_dl):
    train_indices = train_dl.dataset.indices
    val_indices = val_dl.dataset.indices
    return ViT(
        in_c=args.in_c,
        num_classes=args.num_classes,
        img_size=args.size,
        patch=args.patch,
        dropout=args.dropout,
        mlp_hidden=args.mlp_hidden,
        depth=args.num_layers,
        hidden=args.hidden,
        head=args.head,
        train_indices=train_indices,
        val_indices=val_indices
    )



if __name__ == "__main__":
    # current setup -- adapt to run parallel on different servers
    base_dir = Path(__file__).parents[3].resolve().absolute()/'base-models'

    seeds = [1, 2, 3, 4, 5]
    iid_settings = [(True, None), (False, True), (False, False)]  # (is_iid, is_sharded)
    datasets = ['CIFAR-10', 'CIFAR-100', 'TinyImageNet']
    splits = [1, 2, 4, 6]
    use_val_for_model_selection = False

    run_experiment(
        seeds=seeds,
        iid_settings=iid_settings,
        datasets=datasets,
        splits=splits,
        use_val_for_model_selection=use_val_for_model_selection,
        create_model_func=create_vit,
        train_model_func=train_vit,
        model_name='ViT',
        get_args=get_args,
        basedir=base_dir
    )

    # refactor paths for iid settings
    if iid_settings[0] is True and splits == [1] and len(seeds) == 8:
        base_dir = base_dir/'iid'/'ViTs'/datasets[0]/f'split-by-1'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args = get_args(datasets[0], 1)
        _, _, test_dl = prepare_data(datasets[0], args, num_splits=1, is_iid=True, val_ratio=0.1)
        refactor_paths(base_dir, test_dl, device, args.num_classes)
