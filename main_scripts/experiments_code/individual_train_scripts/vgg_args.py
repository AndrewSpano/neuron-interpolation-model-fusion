from argparse import Namespace


cifar10_args = Namespace(
    dataset='c10',
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
    rcpaste=False,
    cutmix=True,
    mixup=False,
    vgg_name='VGG11',
    num_classes=10,
    size=32,
    padding=4,
    use_activations=False,
    input_shape=(3, 32, 32),
    batch_norm=False,
    use_bias=True,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.2435, 0.2616]
)
cifar10_epochs = {
    1: cifar10_args.max_epochs,
    2: 200,
    4: 150,
    6: 125,
    8: 125
}


cifar100_args = Namespace(
    dataset='c100',
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
    rcpaste=False,
    cutmix=True,
    mixup=False,
    use_activations=False,
    vgg_name='VGG11',
    num_classes=100,
    size=32,
    padding=4,
    input_shape=(3, 32, 32),
    batch_norm=False,
    use_bias=True,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.2435, 0.2616]
)
cifar100_epochs = {
    1: cifar100_args.max_epochs,
    2: 250,
    4: 225,
    6: 200,
    8: 200
}

bloodmnist_args = Namespace(
    dataset='blood',
    batch_size=128,
    eval_batch_size=1024,
    lr=0.001,
    min_lr=1e-05,
    beta1=0.9,
    beta2=0.999,
    max_epochs=100,
    weight_decay=5e-5,
    warmup_epoch=5,
    precision=16,
    autoaugment=True,
    smoothing=0.1,
    rcpaste=False,
    cutmix=True,
    mixup=False,
    vgg_name='VGG11',
    num_classes=8,
    size=64,
    padding=4,
    use_activations=False,
    input_shape=(3, 64, 64),
    batch_norm=False,
    use_bias=True,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.247, 0.2435, 0.2616]
)
bloodmnist_args_epochs = {
    1: bloodmnist_args.max_epochs,
    2: 100
}



def get_args(dataset_name, num_splits):
    if dataset_name == 'CIFAR-10':
        cifar10_args.max_epochs = cifar10_epochs[num_splits]
        return cifar10_args
    elif dataset_name == 'CIFAR-100':
        cifar100_args.max_epochs = cifar100_epochs[num_splits]
        return cifar100_args
    elif dataset_name == 'BloodMNIST':
        bloodmnist_args.max_epochs = bloodmnist_args_epochs[num_splits]
        return bloodmnist_args
    else:
        raise ValueError(f"Unknown dataset name for VGGs: {dataset_name}")
