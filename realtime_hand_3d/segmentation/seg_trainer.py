import math
import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import IoU

from realtime_hand_3d.segmentation.criterion import (
    SEG_MODEL_CRITERIONS,
    SEG_CRITERION_REGISTRY,
)
from realtime_hand_3d.segmentation.data import Ego2HandsDataset
from realtime_hand_3d.segmentation.models import SEG_MODELS_REGISTRY
from realtime_hand_3d.segmentation.utils import AverageMeter, Config
from realtime_hand_3d.utils import optimizers, schedulers


class SegTrainer:
    """
    Trainer class for hand segmentation models
    """

    def __init__(
        self,
        model,
        cfg,
        img_dir,
        bg_dir,
        device=None,
    ):

        self.cfg = cfg
        if type(cfg) == str:
            self.cfg = Config(cfg)

        self.img_dir = img_dir
        self.bg_dir = bg_dir

        if device is None:
            device = self.cfg.device

        self.model = model
        self.model_name = model.__class__.__name__.lower()
        self._setup_model(model, device)

    def _setup_model(self, model, device):

        if isinstance(device, list) or isinstance(device, tuple):
            device = ",".join(map(str, device))

        print("\n")

        self.model_parallel = False

        if device == "-1" or device == -1 or device == "cpu":
            device = torch.device("cpu")
            print("Running on CPU\n")

        elif not torch.cuda.is_available():
            device = torch.device("cpu")
            print("CUDA device(s) not available. Running on CPU\n")

        else:
            self.model_parallel = True

            if device == "all":
                device = torch.device("cuda")
                if self.cfg.distributed:
                    model = DDP(model)
                else:
                    model = nn.DataParallel(model)
                print(f"Running on all available CUDA devices\n")

            else:

                if type(device) != str:
                    device = str(device)

                device_ids = device.split(",")
                device_ids = [int(id) for id in device_ids]
                cuda_str = "cuda:" + device
                device = torch.device(cuda_str)
                if self.cfg.distributed:
                    model = DDP(model)
                else:
                    model = nn.DataParallel(model, device_ids=device_ids)
                print(f"Running on CUDA devices {device_ids}\n")

        self.device = device
        self.model = model.to(self.device)

    def _make_dataloader(self):

        dataset = Ego2HandsDataset(
            self.img_dir, self.bg_dir, self.cfg.with_arms, self.cfg.input_edge
        )

        val_size = math.floor(self.cfg.val_split_ratio * len(dataset))
        train_size = math.ceil((1 - self.cfg.val_split_ratio) * len(dataset))
        print(
            f"No. of training samples = {train_size}, no. of validation samples = {val_size}"
        )
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_set, batch_size=self.cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=True)

        print("Loaded dataset")

        self.train_loader = train_loader
        self.val_loader = val_loader

    def _setup_training(self, loss_fn=None, optimizer=None, scheduler=None):

        if loss_fn is None:

            if self.model_name in SEG_MODEL_CRITERIONS.keys():
                loss = SEG_CRITERION_REGISTRY[SEG_MODEL_CRITERIONS[self.model_name]]
            else:
                loss = nn.CrossEntropyLoss

            if self.cfg.criterion.params:
                loss_fn = loss(**self.cfg.criterion.params)
            else:
                loss_fn = loss()

        self.loss_fn = loss_fn

        if optimizer is None:

            opt = optimizers.get(self.cfg.optimizer.name)

            if self.cfg.optimizer.params:
                optimizer = opt(
                    self.model.parameters(),
                    lr=self.cfg.optimizer.lr,
                    **self.cfg.optimizer.params,
                )
            else:
                optimizer = opt(self.model.parameters(), lr=self.cfg.optimizer.lr)

        if scheduler is None:

            if self.cfg.scheduler.use:
                sched = schedulers.get(self.cfg.scheduler.name)

                if self.cfg.scheduler.params:
                    scheduler = sched(optimizer, **self.cfg.SCHEDULER.PARAMS)
                else:
                    scheduler = sched(optimizer)

        return loss_fn, optimizer, scheduler

    def _interpolate(self, img, mask, size=None):

        if size is None:
            size = mask.shape[-2:]

        if img.shape[-2:] != size:
            img = F.interpolate(img, size, mode="bilinear", align_corners=True)

        return img, mask

    def _calculate_loss(self, pred, target):

        pred, target = self._interpolate(pred, target)

        return self.loss_fn(pred, target)

    def _train_model(self, loss_fn, optimizer, scheduler, n_epochs, start_epoch=None):

        writer = SummaryWriter(log_dir=self.cfg.log_dir)

        model = self.model
        best_model = deepcopy(model)
        model.train()

        self.loss_fn = loss_fn

        epoch_loss = AverageMeter()
        min_avg_val_loss = float("inf")
        min_avg_val_metric = float("inf")

        if start_epoch is not None:
            print(f"Resuming training from epoch {start_epoch+1}\n")
        else:
            start_epoch = 0

        for epochs in range(start_epoch, start_epoch + n_epochs):

            print(f"Epoch {epochs+1} of {start_epoch+n_epochs}")
            print("-" * 80)

            epoch_loss.reset()
            for iteration, (img, mask) in enumerate(self.train_loader):

                img, mask = (
                    img.to(self.device),
                    mask.to(self.device),
                )

                pred = model(img)

                loss = self._calculate_loss(pred, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                epoch_loss.update(loss.item())

                if iteration % self.cfg.log_iterations_interval == 0:

                    total_iters = iteration + (epochs * len(self.train_loader))
                    writer.add_scalar(
                        "avg_batch_training_loss",
                        epoch_loss.avg,
                        total_iters,
                    )
                    print(
                        f"Epoch iterations: {iteration}, Total iterations: {total_iters}, Average batch training loss: {epoch_loss.avg}"
                    )

                    new_avg_val_loss, new_avg_val_metric = self._validate_model(model)

                    if new_avg_val_loss < min_avg_val_loss:

                        min_avg_val_loss = new_avg_val_loss
                        print("New minimum average validation loss!")

                        if self.cfg.validate_on.lower() == "loss":
                            best_model = deepcopy(model)
                            save_best_model = (
                                best_model.module if self.model_parallel else best_model
                            )
                            torch.save(
                                save_best_model.state_dict(),
                                os.path.join(
                                    self.cfg.ckpt_dir, self.model_name + "_best.pth"
                                ),
                            )
                            print(
                                f"Saved new best model at epoch {epochs+1}, iteration {iteration}!"
                            )

                    if new_avg_val_metric < min_avg_val_metric:

                        min_avg_val_metric = new_avg_val_metric
                        print("New minimum average validation metric!")

                        if self.cfg.validate_on.lower() == "metric":
                            best_model = deepcopy(model)
                            save_best_model = (
                                best_model.module if self.model_parallel else best_model
                            )
                            torch.save(
                                save_best_model.state_dict(),
                                os.path.join(
                                    self.cfg.ckpt_dir, self.model_name + "_best.pth"
                                ),
                            )
                            print(
                                f"Saved new best model at epoch {epochs+1}, iteration {iteration}!"
                            )

            print(f"\nEpoch {epochs+1}: Training loss = {epoch_loss.sum}")
            writer.add_scalar("epochs_training_loss", epoch_loss.sum, epochs + 1)

            if epochs % self.cfg.validate_interval == 0:

                new_avg_val_loss, new_avg_val_metric = self._validate_model(model)

                writer.add_scalar("avg_validation_loss", new_avg_val_loss, epochs + 1)
                print(f"Epoch {epochs+1}: Average validation loss = {new_avg_val_loss}")

                writer.add_scalar(
                    "avg_validation_metric", new_avg_val_metric, epochs + 1
                )
                print(
                    f"Epoch {epochs+1}: Average validation metric = {new_avg_val_metric}"
                )

                if new_avg_val_loss < min_avg_val_loss:

                    min_avg_val_loss = new_avg_val_loss
                    print("New minimum average validation loss!")

                    if self.cfg.validate_on.lower() == "loss":
                        best_model = deepcopy(model)
                        save_best_model = (
                            best_model.module if self.model_parallel else best_model
                        )
                        torch.save(
                            save_best_model.state_dict(),
                            os.path.join(
                                self.cfg.ckpt_dir, self.model_name + "_best.pth"
                            ),
                        )
                        print(f"Saved new best model at epoch {epochs+1}!")

                if new_avg_val_metric < min_avg_val_metric:

                    min_avg_val_metric = new_avg_val_metric
                    print("New minimum average validation metric!")

                    if self.cfg.validate_on.lower() == "metric":
                        best_model = deepcopy(model)
                        save_best_model = (
                            best_model.module if self.model_parallel else best_model
                        )
                        torch.save(
                            save_best_model.state_dict(),
                            os.path.join(
                                self.cfg.ckpt_dir, self.model_name + "_best.pth"
                            ),
                        )
                        print(f"Saved new best model at epoch {epochs+1}!")

            if epochs % self.cfg.ckpt_interval == 0:

                if self.model_parallel:
                    save_model = model.module
                else:
                    save_model = model

                consolidated_save_dict = {
                    "model_state_dict": save_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epochs": epochs,
                }
                if scheduler is not None:
                    consolidated_save_dict[
                        "scheduler_state_dict"
                    ] = scheduler.state_dict()

                torch.save(
                    consolidated_save_dict,
                    os.path.join(
                        self.cfg.ckpt_dir,
                        self.model_name + "_epochs" + str(epochs + 1) + ".pth",
                    ),
                )

            print("\n")

        writer.close()

        return best_model

    def _validate_model(self, model):

        model.eval()

        metric_fn = IoU(num_classes=self.cfg.n_classes)

        metric_meter = AverageMeter()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for img, mask in self.val_loader:

                img, mask = img.to(self.device), mask.to(self.device)
                pred = model(img)

                if isinstance(pred, tuple) or isinstance(pred, list):
                    pred = pred[-1]

                if pred.shape[-2:] != mask.shape[-2:]:
                    pred, mask = self._interpolate(pred, mask, mask.shape[-2:])

                loss = self._calculate_loss(pred, mask)
                loss_meter.update(loss.item())

                metric = metric_fn(pred, mask)
                metric_meter.update(metric.item())

        model.train()

        return loss_meter.avg, metric_meter.avg

    def train(
        self,
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        n_epochs=None,
        start_epoch=None,
    ):

        loss_fn, optimizer, scheduler = self._setup_training(
            loss_fn, optimizer, scheduler
        )

        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self._make_dataloader()

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir, exist_ok=True)

        print("Training config:\n")
        print(self.cfg)
        print("-" * 80)

        print(f"Training {self.model_name.upper()} for {n_epochs} epochs\n")
        best_model = self._train_model(
            loss_fn, optimizer, scheduler, n_epochs, start_epoch
        )
        print("Training complete!")

        if self.model_parallel:
            best_model = best_model.module

        torch.save(
            best_model.state_dict(),
            os.path.join(self.cfg.ckpt_dir, self.model_name + "_best_final.pth"),
        )
        print("Saved best model!\n")

    def resume_training(
        self,
        consolidated_ckpt=None,
        model_ckpt=None,
        optimizer_ckpt=None,
        n_epochs=None,
        start_epoch=None,
        scheduler_ckpt=None,
        use_cfg=False,
    ):

        consolidated_ckpt = (
            self.cfg.resume_training.consolidated_ckpt
            if use_cfg is True
            else consolidated_ckpt
        )

        if consolidated_ckpt is not None:

            ckpt = torch.load(consolidated_ckpt)

            model_state_dict = ckpt["model_state_dict"]
            optimizer_state_dict = ckpt["optimizer_state_dict"]

            if "scheduler_state_dict" in ckpt.keys():
                scheduler_state_dict = ckpt["scheduler_state_dict"]

            if "epochs" in ckpt.keys():
                start_epoch = ckpt["epochs"] + 1

        else:

            assert (
                model_ckpt is not None and optimizer_ckpt is not None
            ), "Must provide a consolidated ckpt or model and optimizer ckpts separately"

            model_state_dict = torch.load(model_ckpt)
            optimizer_state_dict = torch.load(optimizer_ckpt)

            if scheduler_ckpt is not None:
                scheduler_state_dict = torch.load(scheduler_ckpt)

        model = self.model.module
        model.load_state_dict(model_state_dict)
        self._setup_model(model)

        loss_fn, optimizer, scheduler = self._setup_training()
        optimizer.load_state_dict(optimizer_state_dict)

        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state_dict)

        if n_epochs is None and use_cfg:
            n_epochs = self.cfg.resume_training.n_epochs
        if start_epoch is None and use_cfg:
            start_epoch = self.cfg.resume_training.start_epoch

        self.train(loss_fn, optimizer, scheduler, n_epochs, start_epoch)

    def validate(self, model=None):

        if model is None:
            model = self.model

        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            avg_val_loss, avg_val_metric = self._validate_model(model)

        print(f"Average validation loss = {avg_val_loss}")
        print(f"Average validation metric = {avg_val_metric}")

        return avg_val_loss, avg_val_metric


def fetch_model(model_name, n_classes, in_channels, **kwargs):

    model = SEG_MODELS_REGISTRY.get(model_name)(
        n_classes=n_classes, in_channels=in_channels, **kwargs
    )

    return model


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_cfg", type=str, required=True, help="Path to the config file"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model to be trained",
    )
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
        "--log_dir",
        type=str,
        required=True,
        help="Directory where logs are to be written",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory where ckpts are to be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ids comma separated with no spaces- (0,1..). Enter 'all' to run on all available GPUs. Use -1 for CPU",
    )
    parser.add_argument(
        "--distributed",
        type=bool,
        default=False,
        help="Whether to do distributed training",
    )
    parser.add_argument("--lr", required=False, help="Learning rate")
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Whether to resume training from a previous ckpt",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to ckpt for resuming training",
    )
    parser.add_argument(
        "--resume_epochs",
        type=int,
        default=None,
        help="Number of epochs to train after resumption",
    )
    parser.add_argument(
        "--n_classes", type=int, default=3, help="No. of segmentation classes"
    )
    parser.add_argument(
        "--in_channels", type=int, default=1, help="No. of input channels"
    )

    args = parser.parse_args()

    training_cfg = Config(args.train_cfg)
    training_cfg.n_classes = args.n_classes
    training_cfg.in_channels = args.in_channels
    training_cfg.log_dir = args.log_dir
    training_cfg.ckpt_dir = args.ckpt_dir
    training_cfg.device = args.device
    training_cfg.distributed = args.distributed

    if args.lr is not None:
        training_cfg.optimizer.lr = args.lr

    model = fetch_model(
        args.model, n_classes=args.n_classes, in_channels=args.in_channels
    )

    trainer = SegTrainer(model, training_cfg, args.img_dir, args.bg_dir, args.device)

    if args.resume is True:
        assert (
            args.resume_ckpt is not None
        ), "Please provide a ckpt to resume training from"
        trainer.resume_training(args.resume_ckpt, n_epochs=args.resume_epochs)

    else:
        trainer.train(n_epochs=args.epochs)
