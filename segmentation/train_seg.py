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


def fetch_model(model_name, n_classes):

    model = models.__dict__[model_name](n_classes=n_classes)

    return model


def train_seg(img_dir, bg_dir, config_path, model, device):

    config = Config(config_path)

    dataset = Ego2HandsDataset(img_dir, bg_dir, config.with_arms, config.input_edge)

    val_size = math.floor(config.val_split_ratio * len(dataset))
    train_size = math.ceil((1 - config.val_split_ratio) * len(dataset))
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

    print("Training!")
    model = train_model(
        model,
        train_loader,
        val_loader,
        config.iters,
        optimizer,
        lr_scheduler,
        device,
        config.n_classes,
        config.eval_interval,
        log_dir=config.log_dir,
    )

    os.makedirs(config.save_dir, exist_ok=True)
    model_name = model.__class__.__name__.lower()
    torch.save(model.state_dict(), os.path.join(config.save_dir, model_name + ".pth"))


def train_model(
    model,
    train_loader,
    val_loader,
    n_iters,
    optimizer,
    lr_scheduler,
    device,
    n_classes,
    eval_interval=2,
    log_dir=None,
):

    writer = SummaryWriter(log_dir)

    loss_fn = nn.CrossEntropyLoss()

    avg_iou = 0
    best_model = deepcopy(model)

    model = model.to(device)
    model.train()

    iters = 0
    while iters < n_iters:

        print(f"Iteration {iters} of {n_iters} ")

        iter_loss = 0
        for img, mask in train_loader:

            iters += 1

            img, mask = img.to(device), mask.to(device)
            out = model(img)
            if isinstance(out, Tuple):
                out = out[0]

            if out.shape[-2:] != mask.shape[-2:]:
                pass  # ToDo

            loss = loss_fn(out, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            iter_loss += loss.item()

        if iters % eval_interval == 0:

            new_avg_iou = eval_model(model, val_loader, n_classes, device)
            if new_avg_iou > avg_iou:
                best_model = deepcopy(model)
                avg_iou = new_avg_iou

            writer.add_scalar("Validation IoU", avg_iou, iter)

        writer.add_scalar("Training loss", iter_loss, iter)

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

        iou = iou_getter(out, mask)
        iou_meter.update(iou, n=batch_size)

    return iou_meter.avg


if __name__ == "__main__":

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--imgdir",
    #     type=str,
    #     required=True,
    #     help="Path to the root directory containing all training images",
    # )
    # parser.add_argument(
    #     "--bgdir",
    #     type=str,
    #     required=True,
    #     help="Path to the root directory containing all background images",
    # )
    # parser.add_argument(
    #     "--config", type=str, required=True, help="Path to the config file"
    # )

    # parser.add_argument(
    #     "--model",
    #     type=str,
    #     required=True,
    #     help="Name (in lowercase) of the model to be trained",
    # )

    # parser.add_argument(
    #     "--save_path",
    #     type=str,
    #     default="models_saved/",
    #     help="Directory where to store models",
    # )

    # parser.add_argument(
    #     "--device", type=int, default=0, help="GPU device id (0,1..), -1 for CPU"
    # )

    # args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.CustomUNet(in_channels=1, n_classes=3)

    train_seg(
        "../data/ego2hands/train",
        "../data/ego2hands/background",
        "segmentation/config.yml",
        model,
        device,
    )
