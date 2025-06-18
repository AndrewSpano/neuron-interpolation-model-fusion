import torch
import torch.nn as nn

from pathlib import Path

from utils.general_utils import set_seed
from fusion_algorithms.otfusion import OTFusion
from models.mlp import MLPNet
from main_scripts.experiments_code.training import train_test_model
from utils.general_utils import get_mnist_dataset, sample_balanced_subset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from models.ensemble_vanilla_averaging import Ensemble, VanillaAveraging

SEEDS = [1,2,3,4,5]
NUM_MODELS = 2
EPOCHS = 200
USE_CNNS = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVEDIR = Path('saved-models-otfusion-with-scores')
IID_SPLIT = False
LOAD_TRAINED_MODELS = True
CLEAR_PREVIOUS_SAVED_MODELS_IF_TRAINING = True


def main(SEED):
    MODEL_SAVEDIR_EXPERIMENT = Path(str(MODEL_SAVEDIR) + '/' + str(SEED))

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
        first_model = MLPNet(28 * 28, 32).to(DEVICE)
        second_model = MLPNet(28 * 28, 32).to(DEVICE)

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

    # fuse the models and save the fused model
    print('-' * 50)
    print(f'Fusing the trained models with OTFusion')
    split1_dataset = train_dataloaders[0].dataset
    original_mnist = split1_dataset.dataset

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

    df = pd.DataFrame()

    df[f'model0'] = [eval_model(models[0], test_dataloader)['accuracy']]
    df[f'model1'] = [eval_model(models[1], test_dataloader)['accuracy']]

    ensemble = Ensemble(models).to(DEVICE)
    df[f'ensemble'] = [eval_model(ensemble, test_dataloader)['accuracy']]

    vanilla_averaging_model = VanillaAveraging(models).to(DEVICE)
    df[f'vanilla_averaging'] = [eval_model(vanilla_averaging_model, test_dataloader)['accuracy']]

    # OTFusion
    print("Uniform Scores")
    fusion = OTFusion(models_to_align, model_to_align_with)
    fused_model = fusion.fuse_models(alignment_strategy, handle_skip=True)
    torch.save(fused_model, MODEL_SAVEDIR/f'ot-uniform-{alignment_strategy["mode"]}.pth')
    df[f'ot-uniform-{alignment_strategy["mode"]}'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print("Conductance-based Scores")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'conductance'
    alignment_strategy['acts']['rescale_min_importance'] = None
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-conductance.pth')
    df[f'ot-conductance'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print("DeepLIFT-based Scores")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'deeplift'
    alignment_strategy['acts']['rescale_min_importance'] = None
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-deeplift.pth')
    print(f'Fused models saved to {MODEL_SAVEDIR}')
    df[f'ot-deeplift'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print("Conductance-based Scores rescaled to [0.1,1]")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'conductance'
    alignment_strategy['acts']['rescale_min_importance'] = 0.1
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-conductance_0.1.pth')
    df[f'ot-conductance_0.1'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print("DeepLIFT-based Scores rescaled to [0.1,1]")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'deeplift'
    alignment_strategy['acts']['rescale_min_importance'] = 0.1
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-deeplift_0.1.pth')
    df[f'ot-deeplift_0.1'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print("Conductance-based Scores rescaled to [0.5,1]")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'conductance'
    alignment_strategy['acts']['rescale_min_importance'] = 0.5
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-conductance_0.5.pth')
    print(f'Fused models saved to {MODEL_SAVEDIR}')
    df[f'ot-conductance_0.5'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print("DeepLIFT-based Scores rescaled to [0.5,1]")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'deeplift'
    alignment_strategy['acts']['rescale_min_importance'] = 0.5
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-deeplift_0.5.pth')
    df[f'ot-deeplift_0.5'] = [eval_model(fused_model, test_dataloader)['accuracy']]


    print("Conductance-based Scores rescaled to [0.9,1]")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'conductance'
    alignment_strategy['acts']['rescale_min_importance'] = 0.9
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-conductance_0.9.pth')
    print(f'Fused models saved to {MODEL_SAVEDIR}')
    df[f'ot-conductance_0.9'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    print("DeepLIFT-based Scores rescaled to [0.9,1]")
    fusion = OTFusion(models_to_align, model_to_align_with)
    alignment_strategy['acts']['neuron_importance_method'] = 'deeplift'
    alignment_strategy['acts']['rescale_min_importance'] = 0.9
    fused_model = fusion.fuse_models(alignment_strategy)
    torch.save(fused_model, MODEL_SAVEDIR/'ot-deeplift_0.9.pth')
    df[f'ot-deeplift_0.9'] = [eval_model(fused_model, test_dataloader)['accuracy']]

    return df

if __name__ == '__main__':
    torch.serialization.add_safe_globals([
        MLPNet,
        nn.Linear, nn.Dropout, nn.Conv2d, nn.MaxPool2d, nn.Dropout2d,
        set
    ])

    dfs = []
    for SEED in SEEDS:
        dfs.append(main(SEED))

    df = pd.concat(dfs)
    df.to_csv("otfusion_with_scores.csv")
