import json
import torch

from pathlib import Path

from utils.run_utils import eval_model
from utils.run_utils import add_safe_globals
from utils.dataset_utils import get_fusion_data
from utils.dataset_utils import get_test_dataloader
from utils.dataset_utils import get_classifier_head_weights
from models.ensemble_vanilla_averaging import Ensemble, VanillaAveraging
from itertools import combinations


def get_test_results(model, test_dl, device, num_classes=10):
    test_results = eval_model(model, test_dl, device, num_classes=num_classes)
    del test_results['confusion_matrix']  # remove confusion matrix from results
    return test_results


def evaluate_all(
        base_dir: Path,
        iid_settings,
        dataset_names,
        split_by,
        model_types,
        model_folders,
        metrics_list,
        average_over_seeds: bool = True,
        pairwise_ensemble_and_vanilla: bool = False,
        combination_size: int = 2,
        ensemble_ft: bool = False
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_folders = model_folders == "all"
    for dataset_name in dataset_names:
        nc = 10 if (dataset_name == 'MNIST' or dataset_name == "CIFAR-10") else (100 if dataset_name == 'CIFAR-100' else (8 if dataset_name == "BloodMNIST" else 200))
        for iid_setting in iid_settings:
            for model_type in model_types:
                for split_by in split_by:
                    split_by_dir = base_dir/iid_setting/model_type/dataset_name/f'split-by-{split_by}'
                    results_per_model_folder = {}
                    if all_folders:
                        model_folders = [f.name for f in split_by_dir.iterdir() if f.is_dir()]
                    for mf in model_folders:
                        # set up directory for current experiment
                        models_dir = split_by_dir/f'{mf}'
                        assert models_dir.exists(), f"Models directory {models_dir} does not exist."

                        print(f'\nEvaluating {dataset_name} | split={split_by} | iid_setting={iid_setting} | model_folder={mf}')
                        test_dl = get_test_dataloader(dataset_name, batch_size=1_000)

                        # get all model paths
                        model_paths = list(models_dir.glob('*.pt'))
                        models = [torch.load(model_path, weights_only=True).to(device) for model_path in model_paths]

                        base_model_paths = list(models_dir.glob('model*.pt'))
                        base_models = [torch.load(model_path, weights_only=True).to(device) for model_path in base_model_paths]
                        
                        # dict to save results as JSON
                        results = {}

                        # evaluate each model
                        for model_path, model in zip(model_paths, models):
                            print(f'Evaluating {model_path}')
                            results[model_path.stem] = get_test_results(model, test_dl, device, num_classes=nc)

                        # get the percentages for the ensemble
                        _, _, dataset, num_classes = get_fusion_data(dataset_name, list(range(100)), batch_size=100)
                        percentages = get_classifier_head_weights(base_models, dataset, num_classes=num_classes)
                        percentages = torch.from_numpy(percentages).to(device)  # [num_models x num_classes]

                        # create also ensemble + vanilla averaging models, and evaluate them
                        ensemble_model = Ensemble(base_models, percentages=percentages)
                        print(f'Evaluating ensemble model')
                        results['ensemble'] = get_test_results(ensemble_model, test_dl, device, num_classes=nc)
                        vanilla_averaging_model = VanillaAveraging(base_models)
                        print(f'Evaluating vanilla averaging model')
                        results['vanilla_averaging'] = get_test_results(vanilla_averaging_model, test_dl, device, num_classes=nc)

                        if pairwise_ensemble_and_vanilla:
                            for r in [combination_size]:  # r determines the size of the combination
                                for model_combination in combinations(base_models, r):
                                    # create ensemble model and evaluate
                                    ensemble_model = Ensemble(list(model_combination), percentages=percentages)
                                    print(f'Evaluating ensemble model with {r} models')
                                    results[f'ensemble_group-{"-".join([str(base_models.index(m)) for m in model_combination])}'] = get_test_results(ensemble_model, test_dl, device, num_classes=nc)
                                    
                                    # create vanilla averaging model and evaluate
                                    vanilla_averaging_model = VanillaAveraging(list(model_combination))
                                    print(f'Evaluating vanilla averaging model with {r} models')
                                    results[f'vanilla_averaging_group-{"-".join([str(base_models.index(m)) for m in model_combination])}'] = get_test_results(vanilla_averaging_model, test_dl, device, num_classes=nc)

                        # Delete and collect old base models, then load finetuned base models
                        del base_models  # Remove old base models from memory


                        if ensemble_ft:
                            # Load finetuned base models
                            finetuned_model_paths = list(models_dir.glob('finetuned_model_lr-*_epochs-*_model_*.pt'))
                            finetuned_base_models = [torch.load(model_path, weights_only=True).to(device) for model_path in finetuned_model_paths]

                            # Create ensemble + vanilla averaging models for finetuned base models and evaluate them
                            finetuned_ensemble_model = Ensemble(finetuned_base_models, percentages=percentages)
                            print(f'Evaluating finetuned ensemble model')
                            results['finetuned_ensemble'] = get_test_results(finetuned_ensemble_model, test_dl, device, num_classes=nc)

                            finetuned_vanilla_averaging_model = VanillaAveraging(finetuned_base_models)
                            print(f'Evaluating finetuned vanilla averaging model')
                            results['finetuned_vanilla_averaging'] = get_test_results(finetuned_vanilla_averaging_model, test_dl, device, num_classes=nc)

                            if pairwise_ensemble_and_vanilla:
                                for r in [combination_size]:  # r determines the size of the combination
                                    for model_combination in combinations(finetuned_base_models, r):
                                        # create ensemble model and evaluate
                                        ensemble_model = Ensemble(list(model_combination), percentages=percentages)
                                        print(f'Evaluating ensemble model with {r} models')
                                        results[f'finetuned_ensemble_ft_group-{"-".join([str(finetuned_base_models.index(m)) for m in model_combination])}'] = get_test_results(ensemble_model, test_dl, device, num_classes=nc)
                                        
                                        # create vanilla averaging model and evaluate
                                        vanilla_averaging_model = VanillaAveraging(list(model_combination))
                                        print(f'Evaluating vanilla averaging model with {r} models')
                                        results[f'finetuned_vanilla_averaging_ft_group-{"-".join([str(finetuned_base_models.index(m)) for m in model_combination])}'] = get_test_results(vanilla_averaging_model, test_dl, device, num_classes=nc)




                        # save results (and as JSON)
                        results_per_model_folder[mf] = results
                        with open(models_dir/'results.json', 'w') as f:
                            json.dump(results, f, indent=4)

                    # average results over seeds -- compute mean and std
                    if average_over_seeds:
                        averaged_results = {}
                        model_names = sorted(list(results_per_model_folder[model_folders[0]].keys()))
                        for metric in metrics_list:
                            averaged_results[metric] = {}
                            for model_name in model_names:
                                averaged_results[metric][model_name] = {
                                    'mean': torch.mean(torch.tensor([results_per_model_folder[mf][model_name][metric] for mf in model_folders])).item(),
                                    'std': torch.std(torch.tensor([results_per_model_folder[mf][model_name][metric] for mf in model_folders])).item()
                                }
                        # save averaged results
                        with open(split_by_dir/'averaged_results.json', 'w') as f:
                            json.dump(averaged_results, f, indent=4)


if __name__ == "__main__":
    add_safe_globals()
    base_dir = Path(__file__).parents[2].resolve().absolute()/'example-base-models'

    iid_settings = ['sharded']
    model_types = ['ViTs']
    dataset_names = ['CIFAR-100']
    split_by = [2]
    model_folders = ["seed1"]

    metrics_list = ['accuracy', 'f1', 'loss', 'roc_auc']

    evaluate_all(base_dir, iid_settings, dataset_names, split_by, model_types, model_folders, metrics_list, average_over_seeds=False, pairwise_ensemble_and_vanilla=True, combination_size=4, ensemble_ft=True)

