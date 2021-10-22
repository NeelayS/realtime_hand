import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrieve import SEG_MODELS_REGISTRY


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=False, stride=1, padding=1):
        super().__init__()

        self.batchnorm = batchnorm

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
            ),
            self._get_bn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            ),
            self._get_bn(out_channels),
            nn.ReLU(inplace=True),
        )

    def _get_bn(self, channels):
        if self.batchnorm:
            return nn.BatchNorm2d(channels)
        else:
            return nn.Sequential()

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=False, stride=1, padding=1):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            Block(in_channels, out_channels, batchnorm, stride, padding),
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        batchnorm=False,
        stride=1,
        padding=1,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = Block(in_channels, out_channels, batchnorm, stride, padding)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = Block(in_channels, out_channels, batchnorm, stride, padding)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


@SEG_MODELS_REGISTRY.register()
class CustomUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        n_classes=3,
        bilinear=True,
        channels=(64, 128, 256, 512, 1024),
    ):
        super(CustomUNet, self).__init__()

        factor = 2 if bilinear else 1

        self.in_conv = Block(in_channels, channels[0])

        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        self.down4 = Down(channels[3], channels[4] // factor)

        self.up1 = Up(channels[4], channels[3] // factor, bilinear)
        self.up2 = Up(channels[3], channels[2] // factor, bilinear)
        self.up3 = Up(channels[2], channels[1] // factor, bilinear)
        self.up4 = Up(channels[1], channels[0], bilinear)

        self.out_conv = OutConv(channels[0], n_classes)

    def forward(self, x):

        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits


@SEG_MODELS_REGISTRY.register()
def CustomSmallUNet(in_channels=1, n_classes=3, bilinear=True):
    return CustomUNet(in_channels, n_classes, bilinear, (8, 16, 32, 64, 128))
