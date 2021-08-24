import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import BasicBlock


class dfnetv1(nn.Module):
    def __init__(self, in_channels=1, n_classes=1000):
        super(dfnetv1, self).__init__()
        self.inplanes = 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage2 = self._make_layer(64, 3, stride=2)
        self.stage3 = self._make_layer(128, 3, stride=2)
        self.stage4 = self._make_layer(256, 3, stride=2)
        self.stage5 = self._make_layer(512, 1, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)  # 4x32
        x = self.stage2(x)  # 8x64
        x3 = self.stage3(x)  # 16x128
        x4 = self.stage4(x3)  # 32x256
        x5 = self.stage5(x4)  # 32x512

        return x3, x4, x5


class dfnetv2(nn.Module):
    def __init__(self, in_channels=1, n_classes=1000):
        super(dfnetv2, self).__init__()
        self.inplanes = 64
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage2_1 = self._make_layer(64, 2, stride=2)
        self.stage2_2 = self._make_layer(128, 1, stride=1)
        self.stage3_1 = self._make_layer(128, 10, stride=2)
        self.stage3_2 = self._make_layer(256, 1, stride=1)
        self.stage4_1 = self._make_layer(256, 4, stride=2)
        self.stage4_2 = self._make_layer(512, 2, stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)  # 4x32
        x = self.stage2_1(x)  # 8x64
        x3 = self.stage2_2(x)  # 8x64
        x4 = self.stage3_1(x3)  # 16x128
        x4 = self.stage3_2(x4)  # 16x128
        x5 = self.stage4_1(x4)  # 32x256
        x5 = self.stage4_2(x5)  # 32x256

        return x3, x4, x5


class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size) for size in sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                features + len(sizes) * out_features,
                out_features,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=stage(feats), size=(h, w), mode="bilinear", align_corners=True
            )
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
