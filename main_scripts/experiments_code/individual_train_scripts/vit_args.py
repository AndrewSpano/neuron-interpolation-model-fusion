from argparse import Namespace


cifar10_args = Namespace(
    dataset='c10',
    num_classes=10,
    patch=8,
    batch_size=128,
    eval_batch_size=1024,
    lr=0.001,
    min_lr=1e-05,
    beta1=0.9,
    beta2=0.999,
    max_epochs=350,
    weight_decay=5e-5,
    warmup_epoch=5,
    precision=16,
    autoaugment=True,
    smoothing=0.1,
    rcpaste=False,
    cutmix=True, # for seed 2
    mixup=False,  # for seed 3
    dropout=0.0,
    head=12,
    num_layers=7,
    hidden=384,
    mlp_hidden=384,
    size=32,
    in_c=3,
    padding=4,
    accumulate_grad_batches=1,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.2435, 0.2616]
)
cifar10_epochs = {
    1: cifar10_args.max_epochs,
    2: 200,
    4: 150,
    6: 125
}


cifar100_args = Namespace(
    dataset='c100',
    num_classes=100,
    patch=8,
    batch_size=128,
    eval_batch_size=1024,
    lr=0.001,
    min_lr=1e-05,
    beta1=0.9,
    beta2=0.999,
    max_epochs=350,
    weight_decay=5e-5,
    warmup_epoch=5,
    precision=16,
    autoaugment=True,
    smoothing=0.1,
    rcpaste=False,
    cutmix=True,
    mixup=False,
    dropout=0.0,
    head=12,
    num_layers=7,
    hidden=384,
    mlp_hidden=384,
    size=32,
    in_c=3,
    padding=4,
    accumulate_grad_batches=1,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.2435, 0.2616]
)
cifar100_epochs = {
    1: cifar100_args.max_epochs,
    2: 250,
    4: 225,
    6: 200
}


tiny_imagenet_args = Namespace(
    dataset='tiny-imagenet',
    num_classes=200,
    patch=8,
    batch_size=128,
    eval_batch_size=1024,
    lr=0.001,
    min_lr=1e-05,
    beta1=0.9,
    beta2=0.999,
    max_epochs=300,
    weight_decay=5e-5,
    warmup_epoch=5,
    precision=16,
    autoaugment=True,
    smoothing=0.1,
    cutmix=True,
    mixup=False,
    dropout=0.1,
    head=12,
    num_layers=7,
    hidden=384,
    mlp_hidden=1536,
    size=64,
    in_c=3,
    padding=4,
    accumulate_grad_batches=8,
    mean=[0.4802, 0.4481, 0.3975], 
    std=[0.2302, 0.2265, 0.2262]
)

tiny_imagenet_epochs = {
    1: tiny_imagenet_args.max_epochs,
    2: tiny_imagenet_args.max_epochs,
    4: 250,
    6: 200
}


def get_args(dataset_name, num_splits):
    if dataset_name == 'CIFAR-10':
        cifar10_args.max_epochs = cifar10_epochs[num_splits]
        return cifar10_args
    elif dataset_name == 'CIFAR-100':
        cifar100_args.max_epochs = cifar100_epochs[num_splits]
        return cifar100_args
    elif dataset_name == 'Tiny-ImageNet':
        return tiny_imagenet_args
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
