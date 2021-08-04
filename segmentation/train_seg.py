from torchmetrics import IoU
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import os
import math
import numpy as np
import cv2
from copy import deepcopy
from typing import Tuple

import segmentation.models as models
from segmentation.utils import Config
from .data import Ego2HandsDataset
from segmentation.utils import AverageMeter


def fetch_model(model_name, n_classes, in_channels):

    model = models.__dict__[model_name](n_classes=n_classes, in_channels=in_channels)

    return model


def train_seg(img_dir, bg_dir, config_path, model, device):

    if device == "-1":
        device = torch.device("cpu")
        print("Running on CPU")

    elif not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA device(s) not available. Running on CPU")

    else:
        if device == "all":
            device = torch.device("cuda")
            model = nn.DataParallel(model)

        else:
            device_ids = device.split(",")
            device_ids = [int(id) for id in device_ids]
            cuda_str = "cuda:" + device
            device = torch.device(cuda_str)
            model = nn.DataParallel(model, device_ids=device_ids)
            print(f"Running on CUDA devices {device_ids}")

    config = Config(config_path)

    dataset = Ego2HandsDataset(img_dir, bg_dir, config.with_arms, config.input_edge)

    val_size = math.floor(config.val_split_ratio * len(dataset))
    train_size = math.ceil((1 - config.val_split_ratio) * len(dataset))
    print(
        f"No. of training samples = {train_size}, no. of validation samples = {val_size}"
    )
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    print("Loaded dataset")

    if config.optimizer == "SGD":
        optimizer = getattr(optim, config.optimizer)(
            model.parameters(), lr=config.lr, momentum=config.momentum
        )
    else:
        optimizer = getattr(optim, config.optimizer)(model.parameters(), lr=config.lr)

    lr_scheduler = getattr(optim.lr_scheduler, config.lr_scheduler)(
        optimizer, step_size=config.lr_policy.step_size, gamma=config.lr_policy.gamma
    )

    os.makedirs(config.save_dir, exist_ok=True)

    print("Training!")
    model = train_model(
        model,
        train_loader,
        val_loader,
        config.epochs,
        optimizer,
        lr_scheduler,
        device,
        config.n_classes,
        config.save_dir,
        config.save_interval,
        config.eval_interval,
        log_dir=config.log_dir,
    )

    model_name = model.__class__.__name__.lower()
    torch.save(
        model.state_dict(), os.path.join(config.save_dir, model_name + "_best.pth")
    )


def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs,
    optimizer,
    lr_scheduler,
    device,
    n_classes,
    save_dir,
    save_interval=2,
    eval_interval=2,
    log_dir=None,
):

    writer = SummaryWriter(log_dir)

    loss_fn = nn.CrossEntropyLoss()

    avg_iou = 0
    best_model = deepcopy(model)

    model = model.to(device)
    model.train()

    iter_loss = AverageMeter()

    epochs = 0
    while epochs < n_epochs:

        print(f"Epoch {epochs+1} of {n_epochs} ")

        iter_loss.reset()
        for iteration, (img, mask) in enumerate(train_loader):

            img, mask = img.to(device), mask.to(device)
            out = model(img)
            if isinstance(out, Tuple):
                out = out[0]

            if out.shape[-2:] != mask.shape[-2:]:
                out = F.interpolate(
                    out, mask.shape[-2:], mode="bilinear", align_corners=False
                )

            loss = loss_fn(out, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            iter_loss.update(loss.item(), train_loader.batch_size)

            if iteration % 5000 == 0:
                writer.add_scalar(
                    "avg_training_loss",
                    iter_loss.avg,
                    iteration + (epochs * len(train_loader.dataset)),
                )

        if epochs % eval_interval == 0:

            new_avg_iou = eval_model(model, val_loader, n_classes, device)
            if new_avg_iou > avg_iou:
                best_model = deepcopy(model)
                avg_iou = new_avg_iou

            writer.add_scalar("validation_iou", avg_iou, epochs + 1)

        print(f"Loss = {iter_loss.sum}")
        writer.add_scalar("epochs_training_loss", iter_loss.sum, epochs + 1)

        if epochs % save_interval == 0:
            model_name = model.__class__.__name__.lower()
            torch.save(model.state_dict(), os.path.join(save_dir, model_name + ".pth"))

        epochs += 1

    writer.close()

    return best_model


def eval_model(model, val_loader, n_classes, device):

    model = model.to(device)
    iou_getter = IoU(num_classes=n_classes)
    iou_meter = AverageMeter()
    batch_size = val_loader.batch_size

    for img, mask in val_loader:

        img, mask = img.to(device), mask.to(device)
        out = model(img)
        if isinstance(out, Tuple):
            out = out[0]

        if out.shape[-2:] != mask.shape[-2:]:
            out = F.interpolate(
                out, mask.shape[-2:], mode="bilinear", align_corners=False
            )

        iou = iou_getter(out, mask)
        iou_meter.update(iou.item(), n=batch_size)

    return iou_meter.avg


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="Path to the root directory containing all training images",
    )
    parser.add_argument(
        "--bg_dir",
        type=str,
        required=True,
        help="Path to the root directory containing all background images",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to be trained",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="models_weights/",
        help="Directory where to store models",
    )

    parser.add_argument(
        "--device",
        default=-1,
        help="GPU device ids comma separated with no spaces- (0,1..). Enter 'all' to run on all available GPUs. Use -1 for CPU",
    )

    args = parser.parse_args()

    config = Config(args.config)
    model = fetch_model(args.model, config.n_classes, config.in_channels)

    train_seg(args.img_dir, args.bg_dir, args.config, model, args.device)
