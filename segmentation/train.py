import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import numpy as np
import cv2

from .models import *
from .data import Ego2HandsDataset


if __name__ == "__main__":

    dataset = Ego2HandsDataset(
        "../data/ego2hands/train", "../data/ego2hands/background"
    )
    trainloader = DataLoader(dataset)

    print(len(dataset))
    a = next(iter(trainloader))
    print(2 in a[3])


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
        "--save_path",
        type=str,
        default="models_saved/",
        help="Directory where to store models",
    )
