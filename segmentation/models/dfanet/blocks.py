import torch.nn as nn
import torch.nn.functional as F
from ..blocks import SeparableConv2d


class BlockA(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
        start_with_relu=True,
    ):
        super(BlockA, self).__init__()

        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU()
        rep = list()
        inter_channels = out_channels // 4

        if start_with_relu:
            rep.append(self.relu)
        rep.append(
            SeparableConv2d(
                in_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer
            )
        )
        rep.append(norm_layer(inter_channels))

        rep.append(self.relu)
        rep.append(
            SeparableConv2d(
                inter_channels, inter_channels, 3, 1, dilation, norm_layer=norm_layer
            )
        )
        rep.append(norm_layer(inter_channels))

        if stride != 1:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    inter_channels, out_channels, 3, stride, norm_layer=norm_layer
                )
            )
            rep.append(norm_layer(out_channels))
        else:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(
                    inter_channels, out_channels, 3, 1, norm_layer=norm_layer
                )
            )
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip

        return out


class Enc(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, norm_layer=nn.BatchNorm2d):
        super(Enc, self).__init__()
        block = list()
        block.append(BlockA(in_channels, out_channels, 2, norm_layer=norm_layer))
        for i in range(blocks - 1):
            block.append(BlockA(out_channels, out_channels, 1, norm_layer=norm_layer))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class FCAttention(nn.Module):
    def __init__(self, in_channels, norm_layer=nn.BatchNorm2d):
        super(FCAttention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1000)
        self.conv = nn.Sequential(
            nn.Conv2d(1000, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        att = self.avgpool(x).view(n, c)
        att = self.fc(att).view(n, 1000, 1, 1)
        att = self.conv(att)
        return x * att.expand_as(x)


class XceptionA(nn.Module):
    def __init__(self, in_channels=1, n_classes=1000, norm_layer=nn.BatchNorm2d):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 2, 1, bias=False), norm_layer(8), nn.ReLU()
        )

        self.enc2 = Enc(8, 48, 4, norm_layer=norm_layer)
        self.enc3 = Enc(48, 96, 6, norm_layer=norm_layer)
        self.enc4 = Enc(96, 192, 4, norm_layer=norm_layer)

        self.fca = FCAttention(192, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, n_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.fca(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
