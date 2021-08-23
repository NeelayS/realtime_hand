import torch.nn as nn
import torch.nn.functional as F


def conv3x3(
    in_planes, out_planes, kernel=3, stride=1, padding=0, dilation=1, bias=True
):
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        fist_dilation=1,
        multi_grid=1,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation * multi_grid,
            dilation=dilation * multi_grid,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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

        out = out + residual
        out = self.relu_inplace(out)

        return out


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
