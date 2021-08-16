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


class SegTrainer:
    """
    Base Trainer class for hand segmentation models
    """

    def __init__(self, model, config_path, img_dir, bg_dir, device="cpu"):

        self.img_dir = img_dir
        self.bg_dir = bg_dir
        self.model = model
        self.config = Config(config_path)

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

        self.device = device

    def _make_dataloader(self):

        dataset = Ego2HandsDataset(
            self.img_dir, self.bg_dir, self.config.with_arms, self.config.input_edge
        )

        val_size = math.floor(self.config.val_split_ratio * len(dataset))
        train_size = math.ceil((1 - self.config.val_split_ratio) * len(dataset))
        print(
            f"No. of training samples = {train_size}, no. of validation samples = {val_size}"
        )
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=self.config.batch_size, shuffle=True
        )

        print("Loaded dataset")

        self.train_loader = train_loader
        self.val_loader = val_loader

    def _interpolate(self, img, mask, size=None):

        if size is None:
            size = mask.shape[-2:]

        if img.shape[-2:] != size:
            img = F.interpolate(img, size, mode="bilinear", align_corners=False)

        return img, mask

    def _calculate_loss(self, img, mask, loss_fn):

        return loss_fn(img, mask)

    def _train_model(self, n_epochs, loss_fn, optimizer, lr_scheduler):

        writer = SummaryWriter(self.config.log_dir)

        avg_iou = 0
        best_model = deepcopy(self.model)

        model = self.model.to(self.device)
        model.train()

        iter_loss = AverageMeter()

        epochs = 0
        while epochs < n_epochs:

            print(f"Epoch {epochs+1} of {n_epochs} ")

            iter_loss.reset()
            for iteration, (img, mask) in enumerate(self.train_loader):

                img, mask = img.to(self.device), mask.to(self.device)
                out = model(img)
                if isinstance(out, Tuple):
                    out = out[0]

                out, mask = self._interpolate(out, mask)

                loss = self._calculate_loss(out, mask, loss_fn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                iter_loss.update(loss.item(), self.train_loader.batch_size)

                if iteration % 5000 == 0:
                    writer.add_scalar(
                        "avg_training_loss",
                        iter_loss.avg,
                        iteration + (epochs * len(self.train_loader.dataset)),
                    )

            if epochs % self.eval_interval == 0:

                new_avg_iou = self._eval_model(model)
                if new_avg_iou > avg_iou:
                    best_model = deepcopy(model)
                    avg_iou = new_avg_iou

                writer.add_scalar("validation_iou", avg_iou, epochs + 1)

            print(f"Loss = {iter_loss.sum}")
            writer.add_scalar("epochs_training_loss", iter_loss.sum, epochs + 1)

            if epochs % self.config.save_interval == 0:
                model_name = model.__class__.__name__.lower()
                torch.save(
                    model.state_dict(),
                    os.path.join(self.config.save_dir, model_name + ".pth"),
                )

            epochs += 1

        writer.close()

        return best_model

    def _eval_model(self, model):

        model = model.to(self.device)
        iou_getter = IoU(num_classes=self.config.n_classes)
        iou_meter = AverageMeter()
        batch_size = self.val_loader.batch_size

        with torch.no_grad():
            for img, mask in self.val_loader:

                img, mask = img.to(self.device), mask.to(self.device)
                out = model(img)
                if isinstance(out, Tuple):
                    out = out[0]

                if out.shape[-2:] != mask.shape[-2:]:
                    out = self._interpolate(out, mask.shape[-2:])

                iou = iou_getter(out, mask)
                iou_meter.update(iou.item(), n=batch_size)

        return iou_meter.avg

    def train(self, n_epochs=None):

        if n_epochs is None:
            n_epochs = self.config.n_epochs

        self._make_dataloader()
        model_name = self.model.__class__.__name__.lower()

        if config.optimizer == "SGD":
            optimizer = getattr(optim, config.optimizer)(
                self.model.parameters(), lr=config.lr, momentum=config.momentum
            )
        else:
            optimizer = getattr(optim, config.optimizer)(
                self.model.parameters(), lr=config.lr
            )

        lr_scheduler = getattr(optim.lr_scheduler, config.lr_scheduler)(
            optimizer,
            step_size=config.lr_policy.step_size,
            gamma=config.lr_policy.gamma,
        )

        loss_fn = nn.CrossEntropyLoss()

        os.makedirs(config.save_dir, exist_ok=True)

        print(f"Training {model_name} for {n_epochs}")
        model = self._train_model(n_epochs, loss_fn, optimizer, lr_scheduler)
        print("Training complete!")

        torch.save(
            model.state_dict(), os.path.join(config.save_dir, model_name + "_best.pth")
        )
        print("Saved best model!")


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

    trainer = SegTrainer(model, args.config, args.img_dir, args.bg_dir, args.device)
    trainer.train(n_epochs=1)
