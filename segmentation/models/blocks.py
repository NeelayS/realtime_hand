import torch.nn as nn
import torch.nn.functional as F
from .common import load_model


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        ksize,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        has_bn=True,
        norm_layer=nn.BatchNorm2d,
        bn_eps=1e-5,
        has_relu=True,
        inplace=True,
        has_bias=False,
    ):

        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=has_bias,
        )
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        inplace=True,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        out = self.relu_inplace(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        inplace=True,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=inplace)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inplace = inplace

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

        if self.inplace:
            out += residual
        else:
            out = out + residual
        out = self.relu_inplace(out)

        return out


def dsn(in_channels, n_classes, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        norm_layer(in_channels),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(
            in_channels, n_classes, kernel_size=1, stride=1, padding=0, bias=True
        ),
    )


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=False,
        norm_layer=None,
    ):
        super(SeparableConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            0,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)

        return x

    def fix_padding(self, x, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))

        return padded_inputs


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channels=1,
        norm_layer=nn.BatchNorm2d,
        bn_eps=1e-5,
        bn_momentum=0.1,
        deep_stem=False,
        stem_width=32,
        inplace=True,
    ):
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNet, self).__init__()

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    stem_width,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(
                    stem_width,
                    stem_width,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                norm_layer(stem_width, eps=bn_eps, momentum=bn_momentum),
                nn.ReLU(inplace=inplace),
                nn.Conv2d(
                    stem_width,
                    stem_width * 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.bn1 = norm_layer(
            stem_width * 2 if deep_stem else 64, eps=bn_eps, momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block,
            norm_layer,
            64,
            layers[0],
            inplace,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.layer2 = self._make_layer(
            block,
            norm_layer,
            128,
            layers[1],
            inplace,
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.layer3 = self._make_layer(
            block,
            norm_layer,
            256,
            layers[2],
            inplace,
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
        self.layer4 = self._make_layer(
            block,
            norm_layer,
            512,
            layers[3],
            inplace,
            stride=2,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )

    def _make_layer(
        self,
        block,
        norm_layer,
        planes,
        blocks,
        inplace=True,
        stride=1,
        bn_eps=1e-5,
        bn_momentum=0.1,
    ):
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
                norm_layer(planes * block.expansion, eps=bn_eps, momentum=bn_momentum),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                norm_layer,
                bn_eps,
                bn_momentum,
                downsample,
                inplace,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    bn_eps=bn_eps,
                    bn_momentum=bn_momentum,
                    inplace=inplace,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layers = []
        x = self.layer1(x)
        layers.append(x)
        x = self.layer2(x)
        layers.append(x)
        x = self.layer3(x)
        layers.append(x)
        x = self.layer4(x)
        layers.append(x)

        return layers


def resnet18(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet34(pretrained_model=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet50(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model


def resnet101(pretrained_model=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained_model is not None:
        model = load_model(model, pretrained_model)
    return model
