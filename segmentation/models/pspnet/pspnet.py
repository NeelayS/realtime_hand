import torch.nn as nn
from torch.nn import functional as F
import torch

from .blocks import conv3x3, ResNet, Bottleneck, PSPModule


class PSPHead(nn.Module):
    def __init__(self, block, layers):

        self.inplanes = 128
        super(PSPHead, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True
        )  # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1)
        )
        self.head = PSPModule(2048, 512)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, affine=True),
            )

        layers = []
        generate_multi_grid = (
            lambda index, grids: grids[index % len(grids)]
            if isinstance(grids, tuple)
            else 1
        )
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                dilation=dilation,
                downsample=downsample,
                multi_grid=generate_multi_grid(0, multi_grid),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation,
                    multi_grid=generate_multi_grid(i, multi_grid),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)

        return x


def PSPNet_res101(n_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], n_classes)
    return model


def PSPNet_res50(n_classes=21):
    model = ResNet(Bottleneck, [3, 4, 6, 3], n_classes)
    return model


def PSPHead_res50():
    model = PSPHead(Bottleneck, [3, 4, 6, 3])
    return model