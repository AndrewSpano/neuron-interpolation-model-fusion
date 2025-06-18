import torch

from pathlib import Path

from models.vit import ViT
from utils.run_utils import add_safe_globals

from utils.general_utils import set_seed
from utils.cifar_train_utils import get_cifar_with_advanced_transforms
from utils.dataset_utils import get_mnist_dataset, get_tiny_imagenet_dataset
from main_scripts.experiments_code.individual_train_scripts.vit_generic_train import train_vit
from main_scripts.experiments_code.individual_train_scripts.vit_args import get_args
# from main_scripts.experiments_code.individual_train_scripts.vit_finetuning_args import get_args



BASE_DIR = Path(__file__).parents[3].resolve().absolute()/'base-models'



def prepare_data(dataset_name, args, num_splits, is_iid):
    if dataset_name.startswith('CIFAR'):
        is_cifar_10 = dataset_name == 'CIFAR-10'
        train_dls, val_dls, test_dl = get_cifar_with_advanced_transforms(num_splits, is_cifar_10, args, is_iid, False, val_ratio=0.1)
    elif dataset_name == 'Tiny-ImageNet':
        train_dls, val_dls, test_dl = get_tiny_imagenet_dataset(num_splits=num_splits, train_batch_size=args.batch_size, test_batch_size=args.eval_batch_size, iid_split=is_iid, val_ratio=0.1)
    elif dataset_name == 'MNIST':
        train_dls, val_dls, test_dl = get_mnist_dataset(num_splits=1, train_batch_size=args.batch_size, test_batch_size=args.eval_batch_size, iid_split=True, val_ratio=0.1)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    return train_dls, val_dls, test_dl

def finetune_merged_model(
    args,
    dataset_name,
    model_path,
    learning_rates,
    epochs_list,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(123)
    
    train_dls, val_dls, test_dl = prepare_data(dataset_name, args, 1, True)
    
    best_acc = 0.0
    best_lr, best_epochs, best_results = None, None, None
    skipped = False

    for lr in learning_rates:
        for epochs in epochs_list:
            model = torch.load(model_path, weights_only=True).to(device)
            savepath = model_path.parent / f'finetuned_model_lr-{lr}_epochs-{epochs}_{model_path.name}'
            
            savepath.parent.mkdir(parents=True, exist_ok=True)

            if savepath.exists():
                skipped = True
                print(f"Skipping {str(savepath)}")
                continue

            args.lr = lr
            args.max_epochs = epochs
            results, savepath = train_vit(
                model=model,
                args=args,
                savepath=savepath,
                train_dl=train_dls[0],
                val_dl=val_dls[0],
                test_dl=test_dl,
                use_val_for_model_selection=False,
                use_train_scheduler=False
            )

            if results['accuracy'] > best_acc:
                best_acc = results['accuracy']
                best_results = results
                best_lr = lr
                best_epochs = epochs
                print(f"Current Best Run LR: {best_lr}\n Current Best Run Epochs: {best_epochs}\n ----------------- \n\n Current Best Run results: {best_results}")

    if not skipped:
        print(f"Best Run LR: {best_lr}\n Best Run Epochs: {best_epochs}\n ----------------- \n\nBest Run results: {best_results}")

    return best_lr, best_epochs, best_results
    

def get_learning_rates_and_epochs(dataset_name):
    if dataset_name == 'CIFAR-10':
        return [1e-4], [200]
    elif dataset_name == 'CIFAR-100':
        return [1e-4], [200]
    elif dataset_name == 'Tiny-ImageNet':
        return [1e-3,1e-4,1e-5], [1, 2, 10, 20]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

if __name__ == "__main__":
    import os
    add_safe_globals()

    dataset_name = 'CIFAR-100'
    models_folder = "example-base-models/full-dataset/ViTs/CIFAR-100"
    model_type = 'ViTs'
    prefix = ''
    exclude_endings = []


    learning_rates, epochs_list = get_learning_rates_and_epochs(dataset_name)

    args = get_args(dataset_name, 1)
    args.max_epochs = 25000
    args.warmup_epoch =  6

    best_lr_counts = {}
    best_epochs_counts = {}
    best_pair = {}

    with open(Path(models_folder) / f"finetune_results.txt", "w") as f:
        f.write(f"Finetuning models in {models_folder}\n")

    for file in os.listdir(models_folder):
        if file.endswith(".pt") and file.startswith(prefix) and not any([file.endswith(ed) for ed in exclude_endings]):
            model_path = Path(models_folder) / file
            print(f"Finetuning {model_path}")
            best_lr, best_epochs, best_results = finetune_merged_model(
                dataset_name=dataset_name,
                model_path=model_path,
                args=args,
                learning_rates=learning_rates,
                epochs_list=epochs_list
            )
            if best_results is None:
                continue

            best_lr_counts[best_lr] = best_lr_counts.get(best_lr, 0) + 1
            best_epochs_counts[best_epochs] = best_epochs_counts.get(best_epochs, 0) + 1
            best_pair[(best_lr, best_epochs)] = best_pair.get((best_lr, best_epochs), 0) + 1


            with open(model_path.parent / f"finetune_results2.txt", "a") as f:
                f.write(f"Finetuning {model_path}\n")
                f.write(f"Best Run LR: {best_lr}\n")
                f.write(f"Best Run Epochs: {best_epochs}\n")
                f.write(f"Best Run results: {best_results}\n")

    with open(f"{models_folder}/{prefix}-finetune_summary2.txt", "w") as f:
        f.write("Best Learning Rates Counts (sorted by count):\n")
        for lr, count in sorted(best_lr_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"LR: {lr}, Count: {count}\n")

        f.write("\nBest Epochs Counts (sorted by count):\n")
        for epochs, count in sorted(best_epochs_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"Epochs: {epochs}, Count: {count}\n")

        f.write("\nBest Pair Counts (sorted by count):\n")
        for pair, count in sorted(best_pair.items(), key=lambda x: x[1], reverse=True):
            f.write(f"Pair: {pair}, Count: {count}\n")
