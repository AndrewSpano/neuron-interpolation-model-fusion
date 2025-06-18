import torch
import numpy as np

from time import time
from pathlib import Path
from argparse import Namespace
from torch.utils.data import DataLoader

from models.vgg import VGG
from utils.run_utils import eval_model
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.cifar_train_utils import LabelSmoothingCrossEntropyLoss, CutMix, MixUp



def train_vgg(
        model: VGG,
        args: Namespace,
        savepath: Path,
        train_dl: DataLoader,
        val_dl: DataLoader,
        test_dl: DataLoader,
        use_val_for_model_selection: bool = True
):
    device = model.device

    criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.min_lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warmup_epoch, after_scheduler=base_scheduler)

    cutmix = CutMix(args.size, beta=1.) if args.cutmix else None
    mixup = MixUp(alpha=1.) if args.mixup else None

    best_acc = 0.0
    epochs = args.max_epochs
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0

        start = time()
        for img, label in train_dl:
            img, label = img.to(device), label.to(device)

            if args.cutmix or args.mixup:
                if args.cutmix:
                    img, label, rand_label, lambda_ = cutmix((img, label))
                elif args.mixup:
                    if np.random.rand() <= 0.8:
                        img, label, rand_label, lambda_ = mixup((img, label))
                    else:
                        img, label, rand_label, lambda_ = img, label, torch.zeros_like(label), 1.
                out = model(img)
                loss = criterion(out, label) * lambda_ + criterion(out, rand_label) * (1. - lambda_)
            else:
                out = model(img)
                loss = criterion(out, label)

            acc = (out.argmax(-1) == label).float().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        end = time()

        scheduler.step()
        avg_train_loss = train_loss / len(train_dl)
        train_acc = train_acc / len(train_dl)

        if use_val_for_model_selection:
            val_results = eval_model(model, val_dl, device, num_classes=args.num_classes)
        test_results = eval_model(model, test_dl, device, num_classes=args.num_classes)

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1:2}/{epochs}]')
            print(f'\tTrain loss: {avg_train_loss:.4f},  acc: {train_acc:.4f}')
            if use_val_for_model_selection:
                print(f'\tVal   loss: {val_results["loss"]:.4f},  acc: {val_results["accuracy"]:.4f}')
            print(f'\tTest  loss: {test_results["loss"]:.4f},  acc: {test_results["accuracy"]:.4f}')
            print(f'\tTime taken: {end - start:.2f} seconds')

        # save the model if it has improved on the specified (val vs test) set
        if use_val_for_model_selection:
            if val_results["accuracy"] > best_acc:
                best_test_results = test_results
                best_acc = val_results["accuracy"]
                print(f'\tSaving best model to {savepath}, with test accuracy {val_results["accuracy"]:.4f} (at epoch {epoch+1})')
                torch.save(model, savepath)
        else:
            if test_results["accuracy"] > best_acc:
                best_test_results = test_results
                best_acc = test_results["accuracy"]
                print(f'\tSaving best model to {savepath}, with test accuracy {test_results["accuracy"]:.4f} (at epoch {epoch+1})')
                torch.save(model, savepath)

    return best_test_results, savepath
