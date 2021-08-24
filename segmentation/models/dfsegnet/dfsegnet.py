import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import conv3x3, dsn
from .blocks import dfnetv1, dfnetv2, PSPModule


class FusionNode(nn.Module):
    def __init__(self, inplane):
        super(FusionNode, self).__init__()
        self.fusion = conv3x3(inplane * 2, inplane)

    def forward(self, x):
        x_h, x_l = x
        size = x_l.size()[2:]
        x_h = F.interpolate(x_h, size, mode="bilinear", align_corners=True)
        res = self.fusion(torch.cat([x_h, x_l], dim=1))

        return res


class DFSegNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, type="dfv1"):
        super(DFSegNet, self).__init__()

        if type == "dfv1":
            self.backbone = dfnetv1(in_channels=in_channels)
        else:
            self.backbone = dfnetv2(in_channels=in_channels)

        self.cc5 = nn.Conv2d(128, 128, 1)
        self.cc4 = nn.Conv2d(256, 128, 1)
        self.cc3 = nn.Conv2d(128, 128, 1)

        self.ppm = PSPModule(512, 128)

        self.fn4 = FusionNode(128)
        self.fn3 = FusionNode(128)

        self.fc = dsn(128, n_classes=n_classes)

    def forward(self, x):

        x3, x4, x5 = self.backbone(x)
        x5 = self.ppm(x5)
        x5 = self.cc5(x5)
        x4 = self.cc4(x4)
        f4 = self.fn4([x5, x4])
        x3 = self.cc3(x3)
        out = self.fn3([f4, x3])
        out = self.fc(out)

        return [out]


def DFSegNetV1(n_classes=3):
    return DFSegNet(n_classes=n_classes, type="dfv1")


def DFSegNetV2(n_classes=3):
    return DFSegNet(n_classes, type="dfv2")
