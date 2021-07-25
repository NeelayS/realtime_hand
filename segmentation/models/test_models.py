import torch
from .unet import CustomUNet, CustomSmallUNet, UNet, SmallUNet


def test_custom_unet():

    x = torch.randn(1, 3, 572, 572)

    model = CustomUNet()
    _ = model(x)

    model = CustomSmallUNet()
    _ = model(x)


def test_unet():

    x = torch.randn(1, 3, 572, 572)

    model = UNet()
    _ = model(x)

    model = SmallUNet()
    _ = model(x)
