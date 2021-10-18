import torch

from realtime_hand_3d.segmentation.models import (
    BiSegNet,
    CustomICNet,
    CustomSmallUNet,
    CustomUNet,
    DFANet,
    DFSegNet,
    DFSegNetV1,
    DFSegNetV2,
    ESPNet,
    FastSCNN,
    ICNet,
    LightWeightRefineNet,
    ModSegNet,
    PSPNet_res50,
    PSPNet_res101,
    RefineNet,
    SegNet,
    SmallUNet,
    SwiftNetRes18,
    SwiftNetResNet,
    UNet,
)

inp = torch.randn(2, 3, 512, 512)

# def test_BiSegNet():

#     model = BiSegNet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


def test_CustomICNet():

    model = CustomICNet(in_channels=3, n_classes=3).eval()
    out = model(inp)
    assert out.shape == inp.shape


def test_CustomUNet():

    model = CustomUNet(in_channels=3, n_classes=3).eval()
    out = model(inp)
    assert out.shape == inp.shape

    model = CustomSmallUNet(in_channels=3, n_classes=3).eval()
    out = model(inp)
    assert out.shape == inp.shape


# def test_DFANet():

#     model = DFANet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


# def test_DFSegNet(): # Final size

#     model = DFSegNet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape

#     model = DFSegNetV1(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape

#     model = DFSegNetV2(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


# def test_ESPNet():

#     model = ESPNet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


# def test_FastSCNN(): # Final size

#     model = FastSCNN(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


# def test_ICNet():

#     model = ICNet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


def test_ModSegNet():

    model = ModSegNet(in_channels=3, n_classes=3).eval()
    out = model(inp)
    assert out.shape == inp.shape


# def test_PSPNet(): # Final size

#     model = PSPNet_res50(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape

#     model = PSPNet_res101(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


# def test_RefineNet(): # Final size

#     model = RefineNet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


# def test_LightWeightRefineNet(): # Final size

#     model = LightWeightRefineNet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


def test_SegNet():

    model = SegNet(in_channels=3, n_classes=3).eval()
    out = model(inp)
    assert out.shape == inp.shape


# def test_SwiftNet(): # Final size

#     model = SwiftNetRes18(in_channels=3, n_classes=3, pretrained=False).eval()
#     out = model(inp)
#     assert out.shape == inp.shape

#     model = SwiftNetResNet(in_channels=3, n_classes=3).eval()
#     out = model(inp)
#     assert out.shape == inp.shape


def test_UNet():

    model = UNet(in_channels=3, n_classes=3).eval()
    out = model(inp)
    assert out.shape == inp.shape

    model = SmallUNet(in_channels=3, n_classes=3).eval()
    out = model(inp)
    assert out.shape == inp.shape
