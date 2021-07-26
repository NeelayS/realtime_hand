import torch
from .unet import CustomUNet, CustomSmallUNet, UNet, SmallUNet
from .refinenet import RefineNet, LightWeightRefineNet
from .segnet import SegNet, ModSegNet
from .icnet import ICNet


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


def test_refinenet():

    x = torch.randn(1, 3, 572, 572)

    model = RefineNet()
    _ = model(x)

    model = LightWeightRefineNet()
    _ = model(x)


def test_segnet():

    x = torch.randn(1, 3, 572, 572)

    model = SegNet()
    _ = model(x)

    model = ModSegNet()
    _ = model(x)


def test_icnet():

    x = torch.randn(1, 3, 572, 572)

    model = ICNet()
    _ = model(x)

    model = ICNet()
    _ = model(x)
