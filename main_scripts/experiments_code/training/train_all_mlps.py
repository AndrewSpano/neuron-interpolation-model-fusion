from pathlib import Path

from models.mlp import MLPNet
from main_scripts.experiments_code.individual_train_scripts.mlp_args import get_args
from main_scripts.experiments_code.training.model_training_utils import run_experiment
from main_scripts.experiments_code.individual_train_scripts.mlp_generic_train import train_mlp

RUN_IID = True
RUN_NON_IID = True


def create_mlp(args, train_dl, val_dl):
    train_indices = train_dl.dataset.indices
    val_indices = val_dl.dataset.indices
    return MLPNet(
        input_size=args.input_size,
        output_size=args.num_classes,
        train_indices=train_indices,
        val_indices=val_indices,
        hidden_size=args.hidden_size,
        use_bias=args.use_bias,
        use_activations=args.use_activations,
    )



if __name__ == "__main__":

    base_dir = Path(__file__).parents[3].resolve().absolute()/'base-models'

    if RUN_IID:
        print("Running IID experiment")

        # IID with 8 seeds and 1 split
        seeds = [1, 2, 3, 4, 5, 6, 7, 8]
        iid_settings = [(True, None)]  # (is_iid, is_sharded)
        datasets = ['MNIST']
        splits = [1]
        use_val_for_model_selection = False  # disable this for non-iid splits to get non-overfitted models

        run_experiment(
            seeds=seeds,
            iid_settings=iid_settings,
            datasets=datasets,
            splits=splits,
            use_val_for_model_selection=use_val_for_model_selection,
            create_model_func=create_mlp,
            train_model_func=train_mlp,
            model_name='MLP',
            get_args=get_args,
            basedir=base_dir
        )

    if RUN_NON_IID:
        print("Running non-IID experiment")

        # non-IID with 5 seeds and 2, 4, 8 splits
        seeds = [1, 2, 3, 4, 5]
        iid_settings = [(False, True), (False, False)]  # (is_iid, is_sharded)
        datasets = ['MNIST']
        # splits = [2, 4, 8]
        splits = [2]
        use_val_for_model_selection = False  # disable this for non-iid splits to get non-overfitted models

        run_experiment(
            seeds=seeds,
            iid_settings=iid_settings,
            datasets=datasets,
            splits=splits,
            use_val_for_model_selection=use_val_for_model_selection,
            create_model_func=create_mlp,
            train_model_func=train_mlp,
            model_name='MLP',
            get_args=get_args,
            basedir=base_dir
        )
