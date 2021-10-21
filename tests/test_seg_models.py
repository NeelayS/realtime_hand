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

inp = torch.randn(2, 1, 288, 512)

# def test_BiSegNet():

#     model = BiSegNet(in_channels=1, n_classes=3).eval()
#     _ = model(inp)     
#     del model

def test_CustomICNet():

    model = CustomICNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model
    
def test_CustomUNet():

    model = CustomUNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model 

    model = CustomSmallUNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


# def test_DFANet():

#     model = DFANet(in_channels=1, n_classes=3).eval()
#     _ = model(inp)
#     del model    


def test_DFSegNet(): # Final size

    model = DFSegNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model

    model = DFSegNetV1(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model

    model = DFSegNetV2(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


# def test_ESPNet():

#     model = ESPNet(in_channels=1, n_classes=3).eval()
#     _ = model(inp)
#     del model


def test_FastSCNN(): # Final size

    model = FastSCNN(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


def test_ICNet():

    model = ICNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


def test_ModSegNet():

    model = ModSegNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


def test_PSPNet(): # Final size

    model = PSPNet_res50(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model

    model = PSPNet_res101(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


def test_RefineNet(): # Final size

    model = RefineNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


def test_LightWeightRefineNet(): # Final size

    model = LightWeightRefineNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


def test_SegNet():

    model = SegNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model


def test_SwiftNet(): # Final size

    model = SwiftNetRes18(in_channels=1, n_classes=3, pretrained=False).eval()
    _ = model(inp)
    del model



def test_UNet():

    model = UNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model

    model = SmallUNet(in_channels=1, n_classes=3).eval()
    _ = model(inp)
    del model     
