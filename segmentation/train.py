from segmentation.utils.other_utils import Config
from typing import Tuple
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import cv2

import segmentation.models as models
from segmentation.utils import Config
from .data import Ego2HandsDataset


def fetch_model(model_name, n_classes):

    model = models.__dict__[model_name](n_classes=n_classes)

    return model


def train(img_dir, bg_dir, config_path, model):

    config = Config(config_path)

    dataset = Ego2HandsDataset(img_dir, bg_dir, config.with_arms, config.input_edge)

    val_size = config.val_split_ratio * len(dataset)
    train_size = (1 - config.val_split_ratio) * len(dataset)
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    if config.optimizer == "SGD":
        optimizer = getattr(optim, config.optimizer)(
            model.parameters(), lr=config.lr, momentum=config.momentum
        )
    else:
        optimizer = getattr(optim, config.optimizer)(
            model.parameters(), lr=config.lr, momentum=config.momentum
        )

    writer = SummaryWriter(config.logdir)


def train_model(model, train_loader, val_loader, n_iters, optimizer, lr_scheduler, device, save_dir):

    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    iters = 0
    while iters < n_iters:
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

def eval_model(model, val_loader):
    pass


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgdir",
        type=str,
        required=True,
        help="Path to the root directory containing all training images",
    )
    parser.add_argument(
        "--bgdir",
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
        help="Name (in lowercase) of the model to be trained",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="models_saved/",
        help="Directory where to store models",
    )

    parser.add_argument(
        "--device", type=int, default=0, help="GPU device id (0,1..), -1 for CPU"
    )
