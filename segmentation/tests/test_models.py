import torch
from models.unet import CustomUNet, CustomSmallUNet, UNet, SmallUNet
from models.refinenet import RefineNet, LightWeightRefineNet
from models.segnet import SegNet, ModSegNet
from models.icnet import ICNet
from models.fastscnn import FastSCNN


# def test_custom_unet():

x = torch.randn(1, 3, 572, 572)

model = CustomUNet().eval()
out = model(x)
assert out.shape[-2:] == x.shape[-2:]

model = CustomSmallUNet().eval()
out = model(x)
assert out.shape[-2:] == x.shape[-2:]


def test_unet():

    x = torch.randn(1, 3, 572, 572)

    model = UNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]

    model = SmallUNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]


def test_refinenet():

    x = torch.randn(1, 3, 572, 572)

    model = RefineNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]

    model = LightWeightRefineNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]


def test_segnet():

    x = torch.randn(1, 3, 572, 572)

    model = SegNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]

    model = ModSegNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]


def test_icnet():

    x = torch.randn(1, 3, 572, 572)

    model = ICNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]

    model = ICNet().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]


def test_fastscnn():

    x = torch.randn(1, 3, 572, 572)

    model = FastSCNN().eval()
    out = model(x)
    assert out.shape[-2:] == x.shape[-2:]
