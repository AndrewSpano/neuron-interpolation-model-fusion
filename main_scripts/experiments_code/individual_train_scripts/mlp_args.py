from argparse import Namespace


mnist_args = Namespace(
    dataset='mnist',
    patch=8,
    batch_size=128,
    eval_batch_size=1024,
    lr=0.001,
    min_lr=1e-05,
    beta1=0.9,
    beta2=0.999,
    max_epochs=25,
    weight_decay=5e-5,
    warmup_epoch=5,
    smoothing=0.0,
    cutmix=True,
    mixup=False,
    num_classes=10,
    size=28,
    padding=4,
    input_size=28 * 28, 
    hidden_size=256, 
    use_bias=True, 
    use_activations=False,
    mean=[0.1307],  # Mean for MNIST
    std=[0.3081]    # Std for MNIST
)
mnist_epochs = {
    1: mnist_args.max_epochs,
    2: mnist_args.max_epochs,
    4: 25,
    6: 25,
    8: 25,
}

def get_args(dataset_name, num_splits):
    if dataset_name == 'MNIST':
        mnist_args.max_epochs = mnist_epochs[num_splits]
        return mnist_args
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
