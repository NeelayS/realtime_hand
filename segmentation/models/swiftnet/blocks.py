import torch.nn as nn
import torch.nn.functional as F

from ..blocks import conv3x3


bilinear_upsample = lambda x, size: F.interpolate(
    x, size, mode="bilinear", align_corners=True
)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, efficient=True, use_bn=True
    ):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if self.use_bn else None
        self.downsample = downsample
        self.stride = stride
        self.efficient = efficient

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
        relu = self.relu(out)

        return relu, out


class BNReluConv(nn.Sequential):
    def __init__(
        self,
        num_maps_in,
        num_maps_out,
        k=3,
        batch_norm=True,
        bn_momentum=0.1,
        bias=False,
        dilation=1,
    ):
        super(BNReluConv, self).__init__()

        if batch_norm:
            self.add_module("norm", nn.BatchNorm2d(num_maps_in, momentum=bn_momentum))
        self.add_module("relu", nn.ReLU(inplace=batch_norm is True))
        padding = k // 2
        self.add_module(
            "conv",
            nn.Conv2d(
                num_maps_in,
                num_maps_out,
                kernel_size=k,
                padding=padding,
                bias=bias,
                dilation=dilation,
            ),
        )


class Upsample(nn.Module):
    def __init__(self, num_maps_in, skip_maps_in, num_maps_out, use_bn=True, k=3):
        super(Upsample, self).__init__()

        print(
            f"Upsample layer: in = {num_maps_in}, skip = {skip_maps_in}, out = {num_maps_out}"
        )
        self.bottleneck = BNReluConv(skip_maps_in, num_maps_in, k=1, batch_norm=use_bn)
        self.blend_conv = BNReluConv(num_maps_in, num_maps_out, k=k, batch_norm=use_bn)

    def forward(self, x, skip):

        skip = self.bottleneck.forward(skip)
        skip_size = skip.size()[2:4]
        x = bilinear_upsample(x, skip_size)
        x = x + skip
        x = self.blend_conv.forward(x)

        return x


def dsn(in_channels, nclass, norm_layer=nn.BatchNorm2d):

    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        norm_layer(in_channels),
        nn.ReLU(),
        nn.Dropout2d(0.1),
        nn.Conv2d(in_channels, nclass, kernel_size=1, stride=1, padding=0, bias=True),
    )
