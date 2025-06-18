import torch
import random
import numpy as np

from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, Resize
from torchvision.transforms import functional as F  # Ensure all transform utilities are available
from medmnist import BloodMNIST

class TinyImageNetDataset(Dataset):
    def __init__(self,
        dataset_root,
        train=True,
        download=False,
        transform=None
    ):
        if train:
            train_dataset = load_dataset('slegroux/tiny-imagenet-200-clean', split='train')
            val_dataset = load_dataset('slegroux/tiny-imagenet-200-clean', split='validation')
            dataset = concatenate_datasets([train_dataset, val_dataset])
        else:
            dataset = load_dataset('slegroux/tiny-imagenet-200-clean', split='test')
        self.dataset = dataset
        if transform is None:
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
            ])
        else:
            self.transform = transform

        # Extract targets (labels) and other instance variables from the dataset
        self.targets = [item['label'][0] if isinstance(item['label'], list) else item['label'] for item in self.dataset]
        self.classes = list(set(self.targets))  # Unique classes in the dataset
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Class-to-index mapping
        self.samples = [(item['image'], item['label']) for item in self.dataset]  # List of (image, label) pairs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        image = self.transform(image)
        return image, label


DATASETS_DIR = Path(__file__).parent.parent/'datasets'
MNIST_ROOT = DATASETS_DIR/'mnist'
CIFAR10_ROOT = DATASETS_DIR/'CIFAR10'
CIFAR100_ROOT = DATASETS_DIR/'CIFAR100'
BLOODMNIST_ROOT = DATASETS_DIR/'BloodMNIST'


def create_iid_splits(dataset, number_of_splits):
    """Create IID splits of a dataset by evenly distributing data points across splits.
        Parameters:
            dataset (VisionDataset): The dataset to split.
            number_of_splits (int): Number of splits to create.

        Returns:
            list of torch.utils.data.Subset: A list of `Subset` datasets for each split.
            dict: A dictionary mapping a split index to the corresponding data indices of the original dataset.
    """
    # extract targets (labels) from the dataset
    targets = np.array(dataset.targets)
    classes = np.unique(targets).tolist()

    # create a mapping of class indices to the corresponding data indices
    class_to_indices = {cls: np.where(targets == cls)[0].tolist() for cls in classes}

    # initialize a dictionary to hold indices for each split
    split_indices = defaultdict(list)

    # for each class, distribute the indices evenly across splits
    for indices in class_to_indices.values():
        # shuffle the indices for randomness
        np.random.shuffle(indices)

        # calculate the base number of samples per split and the remainder
        base_count = len(indices) // number_of_splits
        remainder = len(indices) % number_of_splits

        # distribute indices to splits
        start = 0
        for split_idx in range(number_of_splits):
            count = base_count + (1 if split_idx < remainder else 0)
            split_indices[split_idx].extend(indices[start:start + count])
            start += count

    for split_ind in split_indices.values():
        random.shuffle(split_ind)

    # return a list of `Subset` Datasets for each split
    return [Subset(dataset, split_indices[i]) for i in range(number_of_splits)], split_indices


def create_non_iid_splits(dataset, number_of_splits, smallest_to_largest_ratio=0.2):
    """Create non-iid splits of a dataset using ordered Dirichlet distributions.
        Parameters:
            dataset (VisionDataset): The dataset to split. Currently, MNIST, CIFAR10, CIFAR100 are supported.
            number_of_splits (int): Number of splits to create.
            smallest_to_largest_ratio (float): Ratio of smallest to largest alpha in Dirichlet distribution.

        Returns:
            list of torch.utils.data.Subset: A list of `Subset` datasets for each split.
            dict: A dictionary mapping a split index to the corresponding data indices of the original daraset.
    """
    # extract targets (labels) from the dataset
    if type(dataset.targets) is list:
        targets = np.array(dataset.targets)
    else:
        targets = dataset.targets.numpy()
    classes = np.unique(targets).tolist()

    # create a mapping of class indices to the corresponding data indices
    class_to_indices = {cls: np.where(targets == cls)[0].tolist() for cls in classes}

    # generate ordered dirichlet alphas based on the smallest_to_largest_ratio
    smallest_alpha = 1.0
    largest_alpha = smallest_alpha / smallest_to_largest_ratio
    alphas = np.linspace(smallest_alpha, largest_alpha, number_of_splits)

    # initialize a dictionary to hold indices for each split
    split_indices = defaultdict(list)

    # for each class, split the indices in a non-iid way across splits
    for indices in class_to_indices.values():
        # shuffle the indices and alphas for randomness, then sample from dirichlet
        np.random.shuffle(indices)
        np.random.shuffle(alphas)
        dirichlet_probs = np.random.dirichlet(alphas)

        # calculate the number of samples per split for this class
        split_counts = (dirichlet_probs * len(indices)).astype(int).tolist()

        # adjust split counts to ensure total matches the number of indices
        split_counts[-1] += len(indices) - sum(split_counts)

        # split the indices according to the calculated counts
        start = 0
        for split_idx, count in enumerate(split_counts):
            split_indices[split_idx].extend(indices[start:start + count])
            start += count

    # return a list of `Subset` Datasets for each split
    return [Subset(dataset, split_indices[i]) for i in range(number_of_splits)], split_indices


def create_sharded_splits(dataset, number_of_splits):
    """Distribute all classes across splits with no class overlap.
    Some splits may have more classes than others if not divisible.

    Parameters:
        dataset (VisionDataset): The dataset to split.
        number_of_splits (int): Number of splits (clients) to create.

    Returns:
        list of torch.utils.data.Subset: Subsets for each client.
        dict: Mapping from split index to original dataset indices.
    """
    print(f'Using sharded split with {number_of_splits} splits.')
    targets = np.array(dataset.targets)
    classes = np.unique(targets).tolist()

    # shuffle and partition classes across splits
    np.random.shuffle(classes)
    classes_per_split = np.array_split(classes, number_of_splits)

    class_to_indices = {cls: np.where(targets == cls)[0].tolist() for cls in classes}

    split_indices = defaultdict(list)
    for split_idx, class_list in enumerate(classes_per_split):
        for cls in class_list:
            split_indices[split_idx].extend(class_to_indices[cls])
        random.shuffle(split_indices[split_idx])

    return [Subset(dataset, split_indices[i]) for i in range(number_of_splits)], split_indices


def create_random_splits(dataset, number_of_splits):
    """Create random splits of a dataset by distributing each sample to a random split.
        Parameters:
            dataset (VisionDataset): The dataset to split. Currently, MNIST, CIFAR10, CIFAR100 are supported.
            number_of_splits (int): Number of splits to create.

        Returns:
            list of torch.utils.data.Subset: A list of `Subset` datasets for each split.
            dict: A dictionary mapping a split index to the corresponding data indices of the original dataset.
    """
    # assign each idx to a random split
    idx_to_split_assignment = np.random.randint(0, number_of_splits, len(dataset))

    # gather each split's indices
    split_indices = defaultdict(list)
    for idx, split_idx in enumerate(idx_to_split_assignment):
        split_indices[split_idx].append(idx)

    # shuffle the indices for each split
    for split_ind in split_indices.values():
        random.shuffle(split_ind)

    # return a list of `Subset` Datasets for each split
    return [Subset(dataset, split_indices[i]) for i in range(number_of_splits)], split_indices


def sample_balanced_subset(original_dataset, indices_to_use=None, batch_size=400):
    """Sample a balanced subset of size `batch_size` from the dataset.
        Parameters:
            original_dataset (VisionDataset): The original dataset.
            indices_to_use (list[int]): Indices to use for sampling. If None, all indices are used.
            batch_size (int): The number of samples in the subset.

        Returns:
            torch.Tensor: A tensor of the sampled subset.
            torch.Tensor: A tensor of the corresponding labels.
    """
    if indices_to_use is None:
        indices_to_use = np.arange(len(original_dataset))

    # check if the batch size is larger than the number of available indices
    if batch_size > len(indices_to_use):
        raise ValueError(f"Batch size {batch_size} is larger than the number of available indices {len(indices_to_use)}.")

    # extract targets (labels) from the subset dataset (using the original dataset's targets)
    if type(original_dataset.targets) is list:
        targets = np.array(original_dataset.targets)[indices_to_use]
    else:
        targets = original_dataset.targets.numpy()[indices_to_use]
    classes = np.unique(targets).tolist()

    # create a mapping of class indices to the corresponding data indices
    class_to_indices = {cls: np.where(targets == cls)[0].tolist() for cls in classes}

    # initialize the subset indices and counts
    balanced_indices = []
    samples_per_class = batch_size // len(classes)

    # first pass: try to take `samples_per_class` from each class
    for cls in classes:
        np.random.shuffle(class_to_indices[cls])
        indices = class_to_indices[cls]
        balanced_indices.extend(indices[:samples_per_class])
        class_to_indices[cls] = indices[samples_per_class:]

    # second pass: distribute remaining samples
    remaining_samples = batch_size - len(balanced_indices)
    while remaining_samples > 0:  # infinite loop when not enough samples
        for cls in classes:
            if remaining_samples == 0:
                break
            if class_to_indices[cls]:
                balanced_indices.append(class_to_indices[cls].pop(0))
                remaining_samples -= 1

    # now materialize the data and return it as a torch tensor (make sure to expand channel dim if needed)
    if len(original_dataset[0][0].shape) == 3:
        x = torch.stack([original_dataset[i][0] for i in balanced_indices])
    else:
        x = torch.cat([original_dataset[i][0] for i in balanced_indices], dim=0)
    if len(x.shape) == 3:
        x = x.unsqueeze(1)
    y = torch.tensor([original_dataset[i][1] for i in balanced_indices], dtype=torch.int64)

    return x, y


def get_classifier_head_weights(models, original_dataset, num_classes):
    """Returns the percentage of samples that each model received, for each class."""
    total_samples_per_class = np.zeros((len(models), num_classes))
    for i, model in enumerate(models):
        model_indices = model.train_indices + model.val_indices
        targets = np.array(original_dataset.targets)[model_indices]
        for cls in range(num_classes):
            total_samples_per_class[i][cls] = np.sum(targets == cls)
    percentages = total_samples_per_class / np.sum(total_samples_per_class, axis=0, keepdims=True)
    return percentages


def get_torchvision_dataset(
        dataset_class,
        dataset_root,
        num_splits,
        iid_split,
        sharded_split,
        train_batch_size,
        test_batch_size,
        train_transform,
        test_transform,
        val_ratio
):
    # base datasets (same data, different transforms for val/test)
    full_train_dataset_with_train_transform = dataset_class(
        dataset_root,
        train=True,
        download=not dataset_root.exists(),
        transform=train_transform
    )
    full_train_dataset_with_test_transform = dataset_class(
        dataset_root,
        train=True,
        download=False,
        transform=test_transform
    )
    test_mnist = dataset_class(
        dataset_root,
        train=False,
        download=False,
        transform=test_transform
    )

    # generate splits using full dataset (indices only)
    data_split_fn = create_iid_splits if iid_split else (create_sharded_splits if sharded_split else create_non_iid_splits)
    _, split_indices_dict = data_split_fn(full_train_dataset_with_train_transform, num_splits)

    train_dataloaders = []
    val_dataloaders = []

    for split_idx in range(num_splits):
        all_indices = split_indices_dict[split_idx]
        np.random.shuffle(all_indices)

        val_size = int(len(all_indices) * val_ratio)
        val_indices = all_indices[:val_size]
        train_indices = all_indices[val_size:]  # maybe instead use a stratified sklearn split

        train_subset = Subset(full_train_dataset_with_train_transform, train_indices)
        val_subset = Subset(full_train_dataset_with_test_transform, val_indices)

        train_dataloaders.append(DataLoader(train_subset, batch_size=train_batch_size, shuffle=True))
        val_dataloaders.append(DataLoader(val_subset, batch_size=test_batch_size, shuffle=False))

    test_dataloader = DataLoader(test_mnist, batch_size=test_batch_size, shuffle=False)

    return train_dataloaders, val_dataloaders, test_dataloader


def get_mnist_dataset(
        num_splits,
        iid_split,
        sharded_split=False,
        train_batch_size=1024,
        test_batch_size=256,
        train_transform=None,
        test_transform=None,
        val_ratio=0.1
):
    """Get the MNIST dataset, split it into `num_splits` splits, and create train/val/test dataloaders."""
    if train_transform is None:
        train_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    if test_transform is None:
        test_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    return get_torchvision_dataset(
        MNIST,
        MNIST_ROOT,
        num_splits,
        iid_split,
        sharded_split,
        train_batch_size,
        test_batch_size,
        train_transform,
        test_transform,
        val_ratio
    )

def get_tiny_imagenet_dataset(
    num_splits,
    iid_split,
    sharded_split=False,
    args=None,
    train_batch_size=1024,
    test_batch_size=256,
    train_transform=None,
    test_transform=None,
    val_ratio=0.1
):
    if args is not None:
        train_transform, test_transform = [], []
        train_transform += [
            RandomCrop(size=args.size, padding=args.padding),
            RandomHorizontalFlip()
        ]

        train_transform += [
            ToTensor(),
            Normalize(mean=args.mean, std=args.std)
        ]
        
        test_transform += [
            ToTensor(),
            Normalize(mean=args.mean, std=args.std)
        ]

        train_transform = Compose(train_transform)
        test_transform = Compose(test_transform)

    return get_torchvision_dataset(
        TinyImageNetDataset,
        MNIST_ROOT,
        num_splits,
        iid_split,
        sharded_split,
        train_batch_size,
        test_batch_size,
        train_transform,
        test_transform,
        val_ratio
    )


def get_cifar10_dataset(
        num_splits,
        iid_split,
        sharded_split=False,
        train_batch_size=1024,
        test_batch_size=256,
        train_transform=None,
        test_transform=None,
        val_ratio=0.1
):
    """Get the CIFAR10 dataset, split it into `num_splits` splits, and create train/val/test dataloaders."""
    if train_transform is None:
        train_transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
    if test_transform is None:
        test_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])

    return get_torchvision_dataset(
        CIFAR10,
        CIFAR10_ROOT,
        num_splits,
        iid_split,
        sharded_split,
        train_batch_size,
        test_batch_size,
        train_transform,
        test_transform,
        val_ratio
)

def create_sharded_splits_by_class(dataset, classes_per_split):
    """Distribute all classes across splits by the list of classes

    Parameters:
        dataset (VisionDataset): The dataset to split.
        number_of_splits (int): Number of splits (clients) to create.

    Returns:
        list of torch.utils.data.Subset: Subsets for each client.
        dict: Mapping from split index to original dataset indices.
    """
    print(f'Using sharded by class split with {len(classes_per_split)} splits.')
    targets = np.array(dataset.targets)
    classes = np.unique(targets).tolist()

    # shuffle and partition classes across splits
    class_to_indices = {cls: np.where(targets == cls)[0].tolist() for cls in classes}

    split_indices = defaultdict(list)
    for split_idx, class_list in enumerate(classes_per_split):
        for cls in class_list:
            split_indices[split_idx].extend(class_to_indices[cls])
        random.shuffle(split_indices[split_idx])

    return [Subset(dataset, split_indices[i]) for i in range(len(classes_per_split))], split_indices


class BloodMNISTDataset(Dataset):
    def __init__(self,
        dataset_root,
        train=True,
        download=False,
        transform=None
    ):
        if train:
            train_dataset = BloodMNIST(
                split='train',
                root=BLOODMNIST_ROOT,
                download=True,
                size=64
            )
            val_dataset = BloodMNIST(
                split='val',
                root=BLOODMNIST_ROOT,
                download=False,
                size=64
            )
            dataset = ConcatDataset([train_dataset, val_dataset])
        else:
            dataset = BloodMNIST(
                split='test',
                root=BLOODMNIST_ROOT,
                download=False,
                size=64
            )
        self.dataset = dataset
        if transform is None:
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            ])
        else:
            self.transform = transform

        # Extract targets (labels) and other instance variables from the dataset
        self.targets = [int(item[1]) for item in self.dataset]
        self.classes = list(set(self.targets))  # Unique classes in the dataset
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # Class-to-index mapping
        self.samples = [(item[0], int(item[1])) for item in self.dataset]  # List of (image, label) pairs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[0]
        label = int(item[1])
        image = self.transform(image)
        return image, label


def get_bloodmnist_dataset(
        num_splits=2,
        iid_split=False,
        sharded_split=False,
        train_batch_size=1024,
        test_batch_size=256,
        train_transform=None,
        test_transform=None,
        val_ratio=0.1
):
    """Get the BloodMNIST dataset, split it into `num_splits` splits, and create train/val/test dataloaders."""

    assert num_splits == 2 and iid_split == False and sharded_split == False # These parameters are ignored

    if train_transform is None:
        train_transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(64, padding=4),
            ToTensor(),
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])
    if test_transform is None:
        test_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
        ])

    # base datasets (same data, different transforms for val/test)
    full_train_dataset_with_train_transform = BloodMNISTDataset(
        BLOODMNIST_ROOT,
        train=True,
        download=not BLOODMNIST_ROOT.exists(),
        transform=train_transform
    )
    full_train_dataset_with_test_transform = BloodMNISTDataset(
        BLOODMNIST_ROOT,
        train=True,
        download=False,
        transform=test_transform
    )
    test_dataset = BloodMNISTDataset(
        BLOODMNIST_ROOT,
        train=False,
        download=False,
        transform=test_transform
    )

    # generate splits using full dataset (indices only)
    classes_per_split = [[0,1,2,3],[4,5,6,7]]
    _, split_indices_dict = create_sharded_splits_by_class(full_train_dataset_with_train_transform, classes_per_split)

    train_dataloaders = []
    val_dataloaders = []

    for split_idx in range(num_splits):
        all_indices = split_indices_dict[split_idx]
        np.random.shuffle(all_indices)

        val_size = int(len(all_indices) * val_ratio)
        val_indices = all_indices[:val_size]
        train_indices = all_indices[val_size:]  # maybe instead use a stratified sklearn split

        train_subset = Subset(full_train_dataset_with_train_transform, train_indices)
        val_subset = Subset(full_train_dataset_with_test_transform, val_indices)

        train_dataloaders.append(DataLoader(train_subset, batch_size=train_batch_size, shuffle=True))
        val_dataloaders.append(DataLoader(val_subset, batch_size=test_batch_size, shuffle=False))

    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_dataloaders, val_dataloaders, test_dataloader

def get_cifar100_dataset(
        num_splits,
        iid_split,
        sharded_split=False,
        train_batch_size=1024,
        test_batch_size=256,
        train_transform=None,
        test_transform=None,
        val_ratio=0.1
):
    """Get the CIFAR100 dataset, split it into `num_splits` splits, and create train/val/test dataloaders."""
    if train_transform is None:
        train_transform = Compose([
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4),
            ToTensor(),
            Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])
    if test_transform is None:
        test_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

    return get_torchvision_dataset(
        CIFAR100,
        CIFAR100_ROOT,
        num_splits,
        iid_split,
        sharded_split,
        train_batch_size,
        test_batch_size,
        train_transform,
        test_transform,
        val_ratio
    )


def get_fusion_data(dataset_name: str, indices_to_sample_from: list[int], batch_size: int):
    cifar_test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    mnist_test_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    tiny_imagenet_test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    ])

    if dataset_name == 'MNIST':
        dataset = MNIST(root=MNIST_ROOT, train=True, download=True, transform=mnist_test_transform)
        num_classes = 10
    elif dataset_name == 'CIFAR-10':
        dataset = CIFAR10(root=CIFAR10_ROOT, train=True, download=True, transform=cifar_test_transform)
        num_classes = 10
    elif dataset_name == 'CIFAR-100':
        dataset = CIFAR100(root=CIFAR100_ROOT, train=True, download=True, transform=cifar_test_transform)
        num_classes = 100
    elif dataset_name == 'Tiny-ImageNet':
        dataset = TinyImageNetDataset(
            dataset_root=MNIST_ROOT,
            train=True,
            download=True,
            transform=tiny_imagenet_test_transform
        )
        num_classes = 200
    elif dataset_name == 'BloodMNIST':
        dataset = BloodMNISTDataset(
            dataset_root=BLOODMNIST_ROOT,
            train=True,
            download=True,
            transform=cifar_test_transform
        )
        num_classes = 8
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
    input_samples, labels = sample_balanced_subset(dataset, indices_to_sample_from, batch_size=batch_size)

    return input_samples, labels, dataset, num_classes


def get_test_dataloader(dataset_name: str, batch_size: int = 32):
    cifar_test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

    mnist_test_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    tiny_imagenet_test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2302, 0.2265, 0.2262])
    ])

    if dataset_name == 'MNIST':
        return MNIST(
            root=MNIST_ROOT,
            train=False,
            download=True,
            transform=mnist_test_transform
        )
    if dataset_name == 'CIFAR-10':
        dataset = CIFAR10(root=CIFAR10_ROOT, train=False, download=True, transform=cifar_test_transform)
    elif dataset_name == 'CIFAR-100':
        dataset = CIFAR100(root=CIFAR100_ROOT, train=False, download=True, transform=cifar_test_transform)
    elif dataset_name == 'Tiny-ImageNet':
        dataset = TinyImageNetDataset(
            dataset_root=MNIST_ROOT,
            train=False,
            download=True,
            transform=tiny_imagenet_test_transform
        )
    elif dataset_name == 'BloodMNIST':
        dataset = BloodMNISTDataset(
            dataset_root=BLOODMNIST_ROOT,
            train=False,
            download=True,
            transform=cifar_test_transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return test_dataloader



if __name__ == "__main__":
    # example usage
    train_dls, val_dls, test_dl = get_cifar10_dataset(
        num_splits=4,
        iid_split=False,
        sharded_split=True,
        train_batch_size=32,
        test_batch_size=32,
        train_transform=None,
        test_transform=None,
        val_ratio=0.1
    )

    # print the number of train samples for every class, per split
    for i, train_dl in enumerate(train_dls):
        train_subset = train_dl.dataset
        original_dataset = train_subset.dataset

        all_targets = np.array(original_dataset.targets)
        subset_targets = np.array(original_dataset.targets)[train_subset.indices]
        class_counts = np.bincount(subset_targets, minlength=10)
        print(f"Split {i}: Class counts: {class_counts}")
