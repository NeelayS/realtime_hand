import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .blocks import ConvBlock, get_resnet


class PyramidPoolingModule(nn.Module):
    def __init__(self, pyramids=[1, 2, 3, 6]):
        super(PyramidPoolingModule, self).__init__()

        self.pyramids = pyramids

    def forward(self, input):

        features = input
        height, width = input.shape[2:]
        for bin_size in self.pyramids:
            x = F.adaptive_avg_pool2d(input, output_size=bin_size)
            x = F.interpolate(
                x, size=(height, width), mode="bilinear", align_corners=True
            )
            features = features + x

        return features


class CascadeFeatFusion(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, n_classes):
        super(CascadeFeatFusion, self).__init__()

        self.conv_low = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            low_channels,
                            out_channels,
                            kernel_size=3,
                            dilation=2,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )
        self.conv_high = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            high_channels, out_channels, kernel_size=1, bias=False
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(out_channels)),
                ]
            )
        )
        self.conv_low_cls = nn.Conv2d(
            out_channels, n_classes, kernel_size=1, bias=False
        )

    def forward(self, input_low, input_high):

        input_low = F.interpolate(
            input_low, size=input_high.shape[2:], mode="bilinear", align_corners=True
        )
        x_low = self.conv_low(input_low)
        x_high = self.conv_high(input_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)

        if self.training:
            x_low_cls = self.conv_low_cls(input_low)
            return x, x_low_cls
        else:
            return x


class ICNet(nn.Module):
    pyramids = [1, 2, 3, 6]
    backbone_os = 8

    def __init__(
        self, in_channels=3, backbone="resnet18", n_classes=3, pretrained_backbone=None
    ):
        super(ICNet, self).__init__()
        if "resnet" in backbone:
            if backbone == "resnet18":
                n_layers = 18
                stage5_channels = 512
            elif backbone == "resnet34":
                n_layers = 34
                stage5_channels = 512
            elif backbone == "resnet50":
                n_layers = 50
                stage5_channels = 2048
            elif backbone == "resnet101":
                n_layers = 101
                stage5_channels = 2048
            else:
                raise NotImplementedError

            self.conv_sub1 = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv1",
                            ConvBlock(
                                in_channels=in_channels,
                                out_channels=32,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                        ),
                        (
                            "conv2",
                            ConvBlock(
                                in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                        ),
                        (
                            "conv3",
                            ConvBlock(
                                in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                        ),
                    ]
                )
            )

            self.backbone = get_resnet(
                n_layers, output_stride=self.backbone_os, n_classes=n_classes
            )
            self.ppm = PyramidPoolingModule(pyramids=self.pyramids)
            self.conv_sub4_reduce = ConvBlock(
                stage5_channels, stage5_channels // 4, kernel_size=1, bias=False
            )

            self.cff_24 = CascadeFeatFusion(
                low_channels=stage5_channels // 4,
                high_channels=128,
                out_channels=128,
                n_classes=n_classes,
            )
            self.cff_12 = CascadeFeatFusion(
                low_channels=128,
                high_channels=64,
                out_channels=128,
                n_classes=n_classes,
            )

            self.conv_cls = nn.Conv2d(
                in_channels=128, out_channels=n_classes, kernel_size=1, bias=False
            )

        else:
            raise NotImplementedError

        self._init_weights()
        if pretrained_backbone is not None:
            self.backbone._load_pretrained_model(pretrained_backbone)

    def forward(self, input):

        x_sub1 = self.conv_sub1(input)

        x_sub2 = F.interpolate(
            input, scale_factor=0.5, mode="bilinear", align_corners=True
        )
        x_sub2 = self._run_backbone_sub2(x_sub2)

        x_sub4 = F.interpolate(
            x_sub2, scale_factor=0.5, mode="bilinear", align_corners=True
        )
        x_sub4 = self._run_backbone_sub4(x_sub4)
        x_sub4 = self.ppm(x_sub4)
        x_sub4 = self.conv_sub4_reduce(x_sub4)

        if self.training:

            x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
            x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)

            x_cff_12 = F.interpolate(
                x_cff_12, scale_factor=2, mode="bilinear", align_corners=True
            )
            x_124_cls = self.conv_cls(x_cff_12)
            return x_124_cls, x_12_cls, x_24_cls

        else:
            x_cff_24 = self.cff_24(x_sub4, x_sub2)
            x_cff_12 = self.cff_12(x_cff_24, x_sub1)

            x_cff_12 = F.interpolate(
                x_cff_12, scale_factor=2, mode="bilinear", align_corners=True
            )
            x_124_cls = self.conv_cls(x_cff_12)
            x_124_cls = F.interpolate(
                x_124_cls, scale_factor=4, mode="bilinear", align_corners=True
            )
            return x_124_cls

    def _run_backbone_sub2(self, input):

        x = self.backbone.conv1(input)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        return x

    def _run_backbone_sub4(self, input):

        x = self.backbone.layer3(input)
        x = self.backbone.layer4(x)

        return x

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained_model(self, pretrained):
        if isinstance(pretrained, str):
            pretrain_dict = torch.load(pretrained, map_location="cpu")
            if "state_dict" in pretrain_dict:
                pretrain_dict = pretrain_dict["state_dict"]
        elif isinstance(pretrained, dict):
            pretrain_dict = pretrained

        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    model_dict[k] = v
                else:
                    print(
                        "[%s]" % (self.__class__.__name__),
                        k,
                        "is ignored due to not matching shape",
                    )
            else:
                print(
                    "[%s]" % (self.__class__.__name__),
                    k,
                    "is ignored due to not matching key",
                )
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


if __name__ == "__main__":

    x = torch.Tensor(1, 1, 512, 512)
    m = ICNet(n_classes=3, in_channels=1).eval()
    out = m(x)
    print(out.shape)
