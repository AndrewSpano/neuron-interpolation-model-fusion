from pathlib import Path

from models.vgg import VGG
from main_scripts.experiments_code.training.model_training_utils import run_experiment
from main_scripts.experiments_code.individual_train_scripts.vgg_generic_train import train_vgg
from main_scripts.experiments_code.individual_train_scripts.vgg_args import get_args


RUN_IID = True
RUN_NON_IID = False

BASE_DIR = Path(__file__).parents[3].resolve().absolute()/'base-models'

def create_vgg(args, train_dl, val_dl):
    train_indices = train_dl.dataset.indices
    val_indices = val_dl.dataset.indices
    return VGG(
        vgg_name=args.vgg_name,
        num_classes=args.num_classes,
        input_shape=args.input_shape,
        batch_norm=args.batch_norm,
        use_bias=args.use_bias,
        use_activations=args.use_activations,
        train_indices=train_indices,
        val_indices=val_indices
    )



if __name__ == "__main__":
    
    print("Running BloodMNIST experiment")

    # IID with 8 seeds and 1 split
    seeds = [1, 2, 3, 4, 5]
    iid_settings = [(False, False)]  # (is_iid, is_sharded)
    datasets = ['BloodMNIST']  # still need to run CIFAR-100
    splits = [2]
    use_val_for_model_selection = True  # disable this for non-iid splits to get non-overfitted models

    run_experiment(
        seeds=seeds,
        iid_settings=iid_settings,
        datasets=datasets,
        splits=splits,
        use_val_for_model_selection=use_val_for_model_selection,
        create_model_func=create_vgg,
        train_model_func=train_vgg,
        model_name='VGG',
        get_args=get_args,
        basedir=BASE_DIR
    )
