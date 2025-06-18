import torch
import torch.nn as nn

from pathlib import Path
from dataclasses import dataclass

from utils.run_utils import eval_model
from utils.general_utils import set_seed


@dataclass
class FinetuneConfig:
    """Configuration for finetuning a model."""
    dataset: str
    num_classes: int
    batch_size: int
    eval_batch_size: int = 128
    lr: float = 1e-4
    min_lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 5e-5
    max_epochs: int = 200
    label_smoothing: float = 0.1
    optimizer: torch.optim.Optimizer | None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None


def get_optimizer_and_scheduler(model, finetune_config: FinetuneConfig):
    """Get the optimizer and scheduler for the model."""
    if finetune_config.optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=finetune_config.lr,
            betas=(finetune_config.beta1, finetune_config.beta2),
            weight_decay=finetune_config.weight_decay
        )
    else:
        optimizer = finetune_config.optimizer

    if finetune_config.scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=finetune_config.max_epochs,
            eta_min=finetune_config.min_lr
        )
    else:
        scheduler = finetune_config.scheduler

    return optimizer, scheduler


def get_criterion(finetune_config: FinetuneConfig):
    """Get the loss function for the model."""
    return nn.CrossEntropyLoss(label_smoothing=finetune_config.label_smoothing)


def finetune_model(model, finetune_config: FinetuneConfig, train_dl, val_dl, test_dl):
    """Finetune the given model on the specified dataset."""

    # get optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, finetune_config)

    # get the loss function
    criterion = get_criterion(finetune_config)

    # training loop
    model.train()
    for epoch in range(finetune_config.max_epochs):
        train_loss, train_acc = 0, 0
        for batch in train_dl:
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(-1) == labels).float().mean().item()

        # evaluate the model on the validation set
        val_results = eval_model(model, val_dl)
        test_results = eval_model(model, test_dl)

        print(f'Epoch [{epoch+1}/{finetune_config.max_epochs}]')
        print(f'\tTrain loss: {train_loss/len(train_dl):.4f},  acc: {train_acc/len(train_dl):.4f}')
        print(f'\tVal   loss: {val_results["loss"]:.4f},  acc: {val_results["accuracy"]:.4f}')
        print(f'\tTest  loss: {test_results["loss"]:.4f},  acc: {test_results["accuracy"]:.4f}')


SEED = 4200
# SEED = 13
set_seed(SEED)

NUM_MODELS = 4
TRAIN_IDX = -1


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {DEVICE = }')

TRAIN_ON_CIFAR10 = True
DATASET_NAME = 'CIFAR-10' if TRAIN_ON_CIFAR10 else 'CIFAR-100'
SAVE_DIR = Path(__file__).parent.parent.parent/'base-models'/'ViTs'/DATASET_NAME/f'split-by-{NUM_MODELS}'