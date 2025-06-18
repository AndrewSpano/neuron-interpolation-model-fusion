import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from models.mlp import MLPNet
from utils.run_utils import eval_model
from utils.general_utils import set_seed
from fusion_algorithms.alf import ALFusion, ALFusionConfig
from utils.fusion_utils import compute_neuron_importance_scores
from utils.dataset_utils import get_mnist_dataset, get_cifar10_dataset, sample_balanced_subset


SEED = 123
HIDDEN_SIZE = 10
set_seed(SEED)


def get_mnist(num_splits):
    """Get the MNIST dataset."""
    return 28*28, get_mnist_dataset(
        num_splits,
        iid_split=False,
        sharded_split=False,
        train_batch_size=128
    )


def get_cifar10(num_splits):
    """Get the CIFAR-10 dataset."""
    return 3 * 32 * 32, get_cifar10_dataset(
        num_splits,
        iid_split=False,
        sharded_split=False,
        train_batch_size=128
    )


def train_ablation_mlps(num_models=2):
    """Train MLPs on the MNIST dataset."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device = }')

    input_dim, (train_dls, val_dls, test_dl) = get_mnist(num_models)
    # input_dim, (train_dls, val_dls, test_dl) = get_cifar10(num_models)
    nets = []
    for idx in range(num_models):
        train_indices = train_dls[idx].dataset.indices
        val_indices = val_dls[idx].dataset.indices
        mlp = MLPNet(input_dim, HIDDEN_SIZE, 10, train_indices=train_indices, val_indices=val_indices,
                     use_bias=True, use_activations=False)
        nets.append(mlp.to(device))
        print(f'For {idx = }: {len(train_indices)} train samples, {len(val_indices)} val samples')

    epochs = 30
    for idx, (net, train_dl, val_dl) in enumerate(zip(nets, train_dls, val_dls)):
        print('\n\n')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        for i in range(epochs):
            train_loss = 0.0
            net.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                y_hat = net(x)
                loss = criterion(y_hat, y)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_dl)

            # evaluate the model + print results
            val_results = eval_model(net, val_dl, device)
            test_results = eval_model(net, test_dl, device)
            print(f'Epoch [{i+1:2}/{epochs}]:')
            print(f'\tTrain loss: {avg_train_loss:.4f}')
            print(f'\tValidation loss: {val_results["loss"]:.4f}   Validation accuracy: {val_results["accuracy"]:.4f}')
            print(f'\tTest loss:       {test_results["loss"]:.4f}   Test accuracy:       {test_results["accuracy"]:.4f}')

    return nets, train_dls, val_dls, test_dl


def compute_importance_scores(nets, val_dls, method):
    """Compute the importance scores for the MLPs."""
    val_indices = []
    for val_dl in val_dls:
        val_indices += val_dl.dataset.indices

    X, y = sample_balanced_subset(val_dls[0].dataset.dataset, val_indices, 100)
    X, y = X.to(nets[0].device), y.to(nets[0].device)

    scores = [[] for _ in range(len(nets))]
    print(f'Computing {method} importance scores')
    for i, net in enumerate(nets):
        scores[i].append(compute_neuron_importance_scores(method, net, 0, X, y, fusion_logic='level'))

    return scores


def get_clustering_and_targets(nets, X, scores):
    """Get the clustering and targets."""
    config = ALFusionConfig(
        X=X,
        y=None,
        optimizer_per_level=None,
        lr_per_level=None,
        weight_decay_per_level=None,
        epochs_per_level=None
    )

    fusion = ALFusion(nets)
    fusion.config = config
    fusion.ni_scores = scores

    T, assignment = fusion._k_means_v2(
        level_idx=0,
        target_width=HIDDEN_SIZE,
        fusion_config=fusion.config,
        level_X=X
    )

    return T.cpu(), assignment.cpu()


def plot_importance_scores(ax, acts, T, assignment, neuron_scores, name):
    colors = ['r', 'g', 'b', 'c']
    for i, (act, col, model_scores) in enumerate(zip(acts, colors, neuron_scores)):
        scores = model_scores[0] / model_scores[0].max()
        scores = torch.clamp(scores, min=0.1)
        multiplier = 100 if name == 'Uniform' else 100
        s = multiplier * scores
        ax.scatter(act[0, :], act[1, :], c=col, alpha=0.5, label=f'activations of model {i+1}', s=s)
    
    # now scatter the T matrix
    ax.scatter(T[0, :], T[1, :], c='k', marker='x', label='Cluster centers')

    acts = torch.cat(acts, dim=1)
    total_acts = acts.shape[1]

    for i in range(total_acts):
        source = acts[:, i]
        target = T[:, assignment[i]]
        ax.plot((source[0], target[0]), (source[1], target[1]), 'k--', linewidth=1, alpha=0.4)

    ax.set_title(f'{name} importance scores', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid()

def main():
    """Main function to train MLPs on MNIST."""
    nets, _, val_dls, _ = train_ablation_mlps(num_models=3)
    device = nets[0].device

    # compute the importance scores
    uniform_scores = compute_importance_scores(nets, val_dls, 'uniform')
    conductance_scores = compute_importance_scores(nets, val_dls, 'conductance')
    deeplift_scores = compute_importance_scores(nets, val_dls, 'deeplift')

    print(f'uniform_scores: {uniform_scores}')
    print(f'conductance_scores: {conductance_scores}')
    print(f'deeplift_scores: {deeplift_scores}')

    # sample 2 images
    dataset = val_dls[-1].dataset.dataset
    indices = nets[-1].val_indices
    X, _ = sample_balanced_subset(dataset, indices, 2)
    X = X.to(device)

    # get the clustering and targets
    uniform_T, uniform_assignment = get_clustering_and_targets(nets, X, uniform_scores)
    conductance_T, conductance_assignment = get_clustering_and_targets(nets, X, conductance_scores)
    deeplift_T, deeplift_assignment = get_clustering_and_targets(nets, X, deeplift_scores)

    print(f'uniform_assignment: {uniform_assignment}')
    print(f'conductance_assignment: {conductance_assignment}')
    print(f'deeplift_assignment: {deeplift_assignment}')

    print(f'uniform_T: {uniform_T}')
    print(f'conductance_T: {conductance_T}')
    print(f'deeplift_T: {deeplift_T}')

    # get the activations per model
    with torch.no_grad():
        acts = [net.eval().level1(X).cpu() for net in nets]

    # initiate the plot
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',  # LaTeX-style math
        'font.size': 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(18, 18))

    # just plot the activations in the first subplot
    ax = axes[0, 0]
    for i, act in enumerate(acts):
        ax.scatter(act[0, :], act[1, :], label=f'activations of model {i+1}', alpha=0.5, s=100)
    ax.set_ylabel('Data Point 2')
    ax.set_title('Activations of the models', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid()

    # plot the uniform importance scores
    ax = axes[0, 1]
    plot_importance_scores(ax, acts, uniform_T, uniform_assignment, uniform_scores, 'Uniform')

    # plot the conductance importance scores
    ax = axes[1, 0]
    ax.set_xlabel('Data Point 1')
    ax.set_ylabel('Data point 2')
    plot_importance_scores(ax, acts, conductance_T, conductance_assignment, conductance_scores, 'Conductance')

    # plot the deeplift importance scores
    ax = axes[1, 1]
    ax.set_xlabel('Data Point 1')
    plot_importance_scores(ax, acts, deeplift_T, deeplift_assignment, deeplift_scores, 'DeepLIFT')

    # save the figure
    fig.tight_layout()
    fig.savefig('importance_scores_mlp.png', dpi=300)



if __name__ == '__main__':
    main()
