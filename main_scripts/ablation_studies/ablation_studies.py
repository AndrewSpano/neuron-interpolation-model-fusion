import torch
import torch.nn as nn

from pathlib import Path

from utils.general_utils import set_seed
from fusion_algorithms.otfusion import OTFusion
from models.mlp import AblationMLPNet
from main_scripts.experiments_code.training import train_test_model
from fusion_algorithms.git_rebasin import GitRebasin
from fusion_algorithms.hf_kf import LSFusion
from utils.general_utils import get_mnist_dataset, sample_balanced_subset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from models.ensemble_vanilla_averaging import Ensemble, VanillaAveraging



SEEDS = [1,2,3,4,5]
NUM_MODELS = 2
EPOCHS = 200
USE_CNNS = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVEDIR = Path('saved-models')
IID_SPLIT = False
LOAD_TRAINED_MODELS = True
CLEAR_PREVIOUS_SAVED_MODELS_IF_TRAINING = True


def main(experiment, SEED, num_middle_layers=0, num_hidden_units=32, fine_tuning_epochs=0):
    assert experiment in ['num_layers', 'num_hidden_units', 'fine_tuning']
    MODEL_SAVEDIR_EXPERIMENT = Path(str(MODEL_SAVEDIR) + '_' + experiment)

    if experiment == 'num_layers':
        MODEL_SAVEDIR_EXPERIMENT = Path(str(MODEL_SAVEDIR_EXPERIMENT) + '_' + str(num_middle_layers) + '/' + str(SEED))

    if experiment == 'num_hidden_units':
        MODEL_SAVEDIR_EXPERIMENT = Path(str(MODEL_SAVEDIR_EXPERIMENT) + '_' + str(num_hidden_units) + '/' + str(SEED))

    if experiment == 'fine_tuning':
        MODEL_SAVEDIR_EXPERIMENT = Path(str(MODEL_SAVEDIR_EXPERIMENT) + '_' + str(fine_tuning_epochs) + '/' + str(SEED))

    print(f'Using device: {DEVICE}')
    set_seed(SEED)

    # get the data
    train_dataloaders, test_dataloader = get_mnist_dataset(NUM_MODELS, IID_SPLIT)

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    def eval_model(model, test_dataloader):
        model.eval()
        with torch.no_grad():
            # place all predictions and targets in the lists below
            stacked_outputs, staked_targets = [], []
            for i, (data, target) in enumerate(test_dataloader):
                output = model(data.to(DEVICE)).cpu()
                stacked_outputs.append(output)
                staked_targets.append(target)
            stacked_outputs, staked_targets = torch.cat(stacked_outputs), torch.cat(staked_targets)

            # calculate the loss
            loss = criterion(stacked_outputs, staked_targets).item()

            # calculate the accuracy, f1 score
            point_predictions = stacked_outputs.argmax(dim=1)
            acc = accuracy_score(staked_targets, point_predictions)
            f1 = f1_score(staked_targets, point_predictions, average='weighted')

            # calculate the roc_auc score
            staked_targets_onehot = torch.zeros_like(stacked_outputs)  # one hot encode the targets
            staked_targets_onehot.scatter_(1, staked_targets.view(-1, 1), 1)
            roc_auc = roc_auc_score(staked_targets_onehot, stacked_outputs, average='weighted', multi_class='ovr')  # ovr == One vs Rest

            return {
                'loss': loss,
                'accuracy': acc,
                'f1': f1,
                'roc_auc': roc_auc
            }

    # initialize the  models list
    models = []

    # train the models (if not loading)
    if not LOAD_TRAINED_MODELS:
        # create the models
        first_model = AblationMLPNet(28 * 28, 32, num_middle_layers=num_middle_layers).to(DEVICE)
        second_model = AblationMLPNet(28 * 28, num_hidden_units, num_middle_layers=num_middle_layers).to(DEVICE)

        models = [first_model, second_model]

        # dictionary to store the trained models -- make it empty if specified
        MODEL_SAVEDIR_EXPERIMENT.mkdir(exist_ok=True, parents=True)
        if CLEAR_PREVIOUS_SAVED_MODELS_IF_TRAINING:
            for model_path in MODEL_SAVEDIR_EXPERIMENT.glob('*.pth'):
                model_path.unlink()

        # train and save the models
        for model_idx, (model, train_dataloader) in enumerate(zip(models, train_dataloaders), start=1):
            train_test_model(model, train_dataloader, test_dataloader, model_idx, EPOCHS, device=DEVICE)
            weights_path = MODEL_SAVEDIR_EXPERIMENT/f'model{model_idx}.pth'
            torch.save(model, weights_path)
            print(f'Model {model_idx} saved to {MODEL_SAVEDIR_EXPERIMENT}')

    # otherwise load the models
    else:
        # for model_idx, model in enumerate(models, start=1):
        for model_idx in range(1, NUM_MODELS + 1):
            weights_path = MODEL_SAVEDIR_EXPERIMENT/f'model{model_idx}.pth'
            model = torch.load(weights_path, map_location=DEVICE, weights_only=True).eval()
            models.append(model)
            print(f'Model {model_idx} loaded from {MODEL_SAVEDIR_EXPERIMENT}')

    df = pd.DataFrame()

    df[f'model0'] = [eval_model(models[0], test_dataloader)['accuracy']]
    df[f'model1'] = [eval_model(models[1], test_dataloader)['accuracy']]

    ensemble = Ensemble(models).to(DEVICE)
    df[f'ensemble'] = [eval_model(ensemble, test_dataloader)['accuracy']]

    if experiment != "num_hidden_units":
        vanilla_averaging_model = VanillaAveraging(models).to(DEVICE)
        df[f'vanilla_averaging'] = [eval_model(vanilla_averaging_model, test_dataloader)['accuracy']]

    # fuse the models and save the fused model
    print('-' * 50)
    print(f'Fusing the trained models with OTFusion')
    split1_dataset = train_dataloaders[0].dataset
    original_mnist = split1_dataset.dataset
    # note: if we use an unbalanced subset, it might perform poorly

    activation_dataset, activation_labels = sample_balanced_subset(original_mnist, split1_dataset, batch_size=7)
    activation_dataset = activation_dataset.to(DEVICE)
    activation_labels = activation_labels.to(DEVICE)

    models_to_align = models[:-1]
    model_to_align_with = models[-1]

    alignment_strategy = {
        'mode': 'acts',  # 'acts' or 'wts'
        'acts': {
            'activations_dataset': activation_dataset,
            'activation_labels': activation_labels,
            'neuron_importance_method': 'uniform',
            'normalize_activations': True
        },
        'wts': {}
    }

    # it matters wrt to which model we align with
    # models_to_be_aligned = models[1:]
    # model_to_be_aligned_with = models[0]

    # OTFusion
    print("Uniform Scores")
    fusion = OTFusion(models_to_align, model_to_align_with)
    fused_model = fusion.fuse_models(alignment_strategy, handle_skip=True)
    torch.save(fused_model, MODEL_SAVEDIR/f'ot-uniform-{alignment_strategy["mode"]}.pth')
    df[f'ot-uniform-{alignment_strategy["mode"]}'] = [eval_model(fused_model, test_dataloader)['accuracy']]
    if fine_tuning_epochs:
        train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'ot-uniform', fine_tuning_epochs, device=DEVICE)
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-ot-uniform.pth')
        print(f'Fine Tuned ot-uniform model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'finetuned-ot-uniform-{alignment_strategy["mode"]}'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    if experiment != 'num_hidden_units':
        # Hungarian Fusion
        print('-' * 50)
        print(f'Hungarian Algorithm')
        fusion = LSFusion([models[0], models[1]])
        activation_dataset, activation_classes  = sample_balanced_subset(original_mnist, split1_dataset, batch_size=200)
        activation_dataset = activation_dataset.to(DEVICE)
        activation_classes = activation_classes.to(DEVICE)
        fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='hungarian', neuron_importance_method='uniform')
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'hungarian-uniform.pth')
        print(f'Hungarian Uniform Fused model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'hungarian-uniform'] = [eval_model(fused_model, test_dataloader)['accuracy']]

        if fine_tuning_epochs:
            train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'hungarian-uniform', fine_tuning_epochs, device=DEVICE)
            torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-hungarian-uniform.pth')
            print(f'Fine Tuned hungarian-uniform model saved to {MODEL_SAVEDIR_EXPERIMENT}')
            df[f'finetuned-hungarian-uniform'] = [eval_model(fused_model, test_dataloader)['accuracy']]

        print('-' * 50)
        print(f'Hungarian Algorithm - Conductance')
        fusion = LSFusion([models[0], models[1]])
        activation_dataset, activation_classes  = sample_balanced_subset(original_mnist, split1_dataset, batch_size=200)
        activation_dataset = activation_dataset.to(DEVICE)
        activation_classes = activation_classes.to(DEVICE)
        fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='hungarian', neuron_importance_method='conductance')
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'hungarian-conductance.pth')
        print(f'Hungarian Fused model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'hungarian-conductance'] = [eval_model(fused_model, test_dataloader)['accuracy']]

        if fine_tuning_epochs:
            train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'hungarian-conductance', fine_tuning_epochs, device=DEVICE)
            torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-hungarian-conductance.pth')
            print(f'Fine Tuned hungarian-conductance model saved to {MODEL_SAVEDIR_EXPERIMENT}')
            df[f'finetuned-hungarian-conductance'] = [eval_model(fused_model, test_dataloader)['accuracy']]

        print('-' * 50)
        print(f'Hungarian Algorithm - DeepLIFT')
        fusion = LSFusion([models[0], models[1]])
        activation_dataset, activation_classes  = sample_balanced_subset(original_mnist, split1_dataset, batch_size=200)
        activation_dataset = activation_dataset.to(DEVICE)
        activation_classes = activation_classes.to(DEVICE)
        fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='hungarian', neuron_importance_method='deeplift')
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'hungarian-deeplift.pth')
        print(f'Hungarian Fused model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'hungarian-deeplift'] = [eval_model(fused_model, test_dataloader)['accuracy']]

        if fine_tuning_epochs:
            train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'hungarian-deeplift', fine_tuning_epochs, device=DEVICE)
            torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-hungarian-deeplift.pth')
            print(f'Fine Tuned hungarian-deeplift model saved to {MODEL_SAVEDIR_EXPERIMENT}')
            df[f'finetuned-hungarian-deeplift'] = [eval_model(fused_model, test_dataloader)['accuracy']]

        # Git Rebasin Fusion
        print('-' * 50)
        print(f'Git Re-basin Algorithm')
        fusion = GitRebasin(models[0], models[1])
        activation_dataset, activation_classes  = sample_balanced_subset(original_mnist, split1_dataset, batch_size=200)
        activation_dataset = activation_dataset.to(DEVICE)
        activation_classes = activation_classes.to(DEVICE)
        fused_model = fusion.fuse(activation_dataset, activation_classes, neuron_importance_method='uniform')
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'git_rebasin.pth')
        print(f'Git Re-basin Fused model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'git_rebasin'] = [eval_model(fused_model, test_dataloader)['accuracy']]

        if fine_tuning_epochs:
            train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'git_rebasin', fine_tuning_epochs, device=DEVICE)
            torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-git_rebasin.pth')
            print(f'Fine Tuned git_rebasin model saved to {MODEL_SAVEDIR_EXPERIMENT}')
            df[f'finetuned-git_rebasin'] = [eval_model(fused_model, test_dataloader)['accuracy']]


    print('-' * 50)
    print(f'k_means Algorithm')
    fusion = LSFusion([models[0], models[1]])
    activation_dataset, activation_classes  = sample_balanced_subset(original_mnist, split1_dataset, batch_size=200)
    activation_dataset = activation_dataset.to(DEVICE)
    activation_classes = activation_classes.to(DEVICE)
    fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='k_means', neuron_importance_method='uniform')
    torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'k_means-uniform.pth')
    print(f'k_means-uniform Fused model saved to {MODEL_SAVEDIR_EXPERIMENT}')
    df[f'k_means-uniform'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    if fine_tuning_epochs:
        train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'k_means-uniform', fine_tuning_epochs, device=DEVICE)
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-k_means-uniform.pth')
        print(f'Fine Tuned k_means-uniform model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'finetuned-k_means-uniform'] = [eval_model(fused_model, test_dataloader)['accuracy']]


    print('-' * 50)
    print(f'k_means Algorithm - Conductance')
    fusion = LSFusion([models[0], models[1]])
    activation_dataset, activation_classes  = sample_balanced_subset(original_mnist, split1_dataset, batch_size=200)
    activation_dataset = activation_dataset.to(DEVICE)
    activation_classes = activation_classes.to(DEVICE)
    fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='k_means', neuron_importance_method='conductance')
    torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'k_means-conductance.pth')
    print(f'k_means-conductance Fused model saved to {MODEL_SAVEDIR_EXPERIMENT}')
    df[f'k_means-conductance'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    if fine_tuning_epochs:
        train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'k_means-conductance', fine_tuning_epochs, device=DEVICE)
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-k_means-conductance.pth')
        print(f'Fine Tuned k_means-conductance model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'finetuned-k_means-conductance'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print('-' * 50)
    print(f'k_means Algorithm - DeepLIFT')
    fusion = LSFusion([models[0], models[1]])
    activation_dataset, activation_classes  = sample_balanced_subset(original_mnist, split1_dataset, batch_size=200)
    activation_dataset = activation_dataset.to(DEVICE)
    activation_classes = activation_classes.to(DEVICE)
    fused_model = fusion.fuse(activation_dataset, activation_classes, fusion_method='k_means', neuron_importance_method='deeplift')
    torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'k_means-deeplift.pth')
    print(f'k_means-deeplift Fused model saved to {MODEL_SAVEDIR_EXPERIMENT}')
    df[f'k_means-deeplift'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    if fine_tuning_epochs:
        train_test_model(fused_model, train_dataloaders[0], test_dataloader, 'k_means-deeplift', fine_tuning_epochs, device=DEVICE)
        torch.save(fused_model, MODEL_SAVEDIR_EXPERIMENT/f'fine_tuned-k_means-deeplift.pth')
        print(f'Fine Tuned k_means-deeplift model saved to {MODEL_SAVEDIR_EXPERIMENT}')
        df[f'finetuned-k_means-deeplift'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    return df

if __name__ == '__main__':
    torch.serialization.add_safe_globals([
        AblationMLPNet,
        nn.Linear, nn.Dropout, nn.Conv2d, nn.MaxPool2d, nn.Dropout2d,
        set
    ])

    num_hidden_units = [8, 16, 32, 64, 128]
    num_middle_layers = [0, 1, 2, 4, 8]
    fine_tuning_epochs = 50

    dfs = []
    for SEED in SEEDS:
        df = main('fine_tuning', SEED, fine_tuning_epochs=fine_tuning_epochs)
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv("ablation_fine_tuning.csv")

    dfs = []
    for n in num_middle_layers:
        for SEED in SEEDS:
            df = main('num_layers', SEED, num_middle_layers=n)
            df["num_middle_layers"] = n
            dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv("ablation_vary_middle_layers.csv")

    dfs = []
    for n in num_hidden_units:
        for SEED in SEEDS:
            df = main('num_hidden_units', SEED, num_hidden_units=n)
            df["num_hidden_units"] = n
            dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv("ablation_vary_num_hidden_units.csv")
