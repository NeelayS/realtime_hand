import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import io

from .models import SEG_MODELS_REGISTRY


class SegPredictor:
    def __init__(
        self,
        model_name,
        model_params=None,
        data_transform=None,
        device="cpu",
    ):

        self.model = SEG_MODELS_REGISTRY.get(model_name)(**model_params)
        self._setup_model(self.model, device)
        self.model.eval()

        self.data_transform = data_transform

    def _setup_model(self, model, device):

        if isinstance(device, list) or isinstance(device, tuple):
            device = ",".join(map(str, device))

        print("\n")

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
                if self.cfg.DISTRIBUTED:
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
                if self.cfg.DISTRIBUTED:
                    model = DDP(model)
                else:
                    model = nn.DataParallel(model, device_ids=device_ids)
                print(f"Running on CUDA devices {device_ids}\n")

        self.device = device
        self.model = model.to(self.device)

    def __call__(self, img1):

        if type(img1) == str:
            img1 = io.read_image(img1)

        if self.data_transform is not None:
            img1 = self.data_transform(img1)

        return self.model(img1)
