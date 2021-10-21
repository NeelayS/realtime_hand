from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrieve import SEG_MODELS_REGISTRY


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
    def __init__(
        self,
        low_channels,
        high_channels,
        out_channels,
        n_classes,
    ):
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


@SEG_MODELS_REGISTRY.register()
class ICNet(nn.Module):
    pyramids = [1, 2, 3, 6]
    backbone_os = 8

    def __init__(
        self,
        in_channels=3,
        backbone="resnet18",
        n_classes=3,
        pretrained_backbone=None,
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
                n_layers, output_stride=self.backbone_os, n_classes=n_classes, in_channels=in_channels
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


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None, dilation=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(in_planes, out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

        self.conv2 = conv3x3(out_planes, out_planes, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = conv1x1(out_planes, out_planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    basic_inplanes = 64

    def __init__(self, block, layers, in_channels=1, output_stride=32, n_classes=1000):
        super(ResNet, self).__init__()

        self.inplanes = self.basic_inplanes
        self.output_stride = output_stride
        self.n_classes = n_classes

        if output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        elif output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 32:
            strides = [1, 2, 2, 2]
            dilations = [1, 1, 1, 1]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(
            in_channels, self.basic_inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.basic_inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(
            block,
            1 * self.basic_inplanes,
            num_layers=layers[0],
            stride=strides[0],
            dilation=dilations[0],
        )
        self.layer2 = self._make_layer(
            block,
            2 * self.basic_inplanes,
            num_layers=layers[1],
            stride=strides[1],
            dilation=dilations[1],
        )
        self.layer3 = self._make_layer(
            block,
            4 * self.basic_inplanes,
            num_layers=layers[2],
            stride=strides[2],
            dilation=dilations[2],
        )
        self.layer4 = self._make_layer(
            block,
            8 * self.basic_inplanes,
            num_layers=layers[3],
            stride=strides[3],
            dilation=dilations[3],
        )

        if self.n_classes is not None:
            self.fc = nn.Linear(8 * self.basic_inplanes * block.expansion, n_classes)

        self._init_weights()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.n_classes is not None:
            x = x.mean(dim=(2, 3))
            x = self.fc(x)

        return x

    def _make_layer(self, block, planes, num_layers, stride=1, dilation=1, grids=None):

        downsample = None
        if (stride != 1) or (self.inplanes != planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        if dilation != 1:
            dilations = [dilation * (2 ** layer_idx) for layer_idx in range(num_layers)]
        else:
            dilations = num_layers * [dilation]

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilations[0]))
        self.inplanes = planes * block.expansion

        for i in range(1, num_layers):
            layers.append(block(self.inplanes, planes, dilation=dilations[i]))

        return nn.Sequential(*layers)

    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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


def resnet18(in_channels=1, pretrained=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet34(pretrained=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet50(pretrained=None, **kwargs):
    model = ResNet(BottleneckBlock, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet101(pretrained=None, **kwargs):
    model = ResNet(BottleneckBlock, [3, 4, 23, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def resnet152(pretrained=None, **kwargs):
    model = ResNet(BottleneckBlock, [3, 8, 36, 3], **kwargs)
    if pretrained is not None:
        model._load_pretrained_model(pretrained)
    return model


def get_resnet(num_layers, **kwargs):
    if num_layers == 18:
        return resnet18(**kwargs)
    elif num_layers == 34:
        return resnet34(**kwargs)
    elif num_layers == 50:
        return resnet50(**kwargs)
    elif num_layers == 101:
        return resnet101(**kwargs)
    elif num_layers == 152:
        return resnet152(**kwargs)
    else:
        raise NotImplementedError
