import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .blocks import conv3x3
from .common import model_urls
from .retrieve import SEG_MODELS_REGISTRY

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


class SpatialPyramidPooling(nn.Module):
    def __init__(
        self,
        num_maps_in,
        num_levels,
        bt_size=512,
        level_size=128,
        out_size=128,
        grids=(6, 3, 2, 1),
        square_grid=False,
        bn_momentum=0.1,
        use_bn=True,
    ):
        super(SpatialPyramidPooling, self).__init__()

        self.grids = grids
        self.square_grid = square_grid
        self.spp = nn.Sequential()
        self.spp.add_module(
            "spp_bn",
            BNReluConv(
                num_maps_in, bt_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn
            ),
        )
        n_classes = bt_size
        final_size = n_classes
        for i in range(num_levels):
            final_size += level_size
            self.spp.add_module(
                "spp" + str(i),
                BNReluConv(
                    n_classes,
                    level_size,
                    k=1,
                    bn_momentum=bn_momentum,
                    batch_norm=use_bn,
                ),
            )
        self.spp.add_module(
            "spp_fuse",
            BNReluConv(
                final_size, out_size, k=1, bn_momentum=bn_momentum, batch_norm=use_bn
            ),
        )

    def forward(self, x):

        levels = []
        target_size = x.size()[2:4]

        ar = target_size[1] / target_size[0]

        x = self.spp[0].forward(x)
        levels.append(x)
        num = len(self.spp) - 1

        for i in range(1, num):
            if not self.square_grid:
                grid_size = (self.grids[i - 1], max(1, round(ar * self.grids[i - 1])))
                x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            else:
                x_pooled = F.adaptive_avg_pool2d(x, self.grids[i - 1])
            level = self.spp[i].forward(x_pooled)

            level = bilinear_upsample(level, target_size)
            levels.append(level)
        x = torch.cat(levels, 1)
        x = self.spp[-1].forward(x)

        return x


@SEG_MODELS_REGISTRY.register()
class SwiftNetResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        in_channels=1,
        n_classes=3,
        k_up=3,
        efficient=True,
        use_bn=True,
        spp_grids=(8, 4, 2, 1),
        spp_square_grid=False,
    ):
        super(SwiftNetResNet, self).__init__()

        self.inplanes = 64
        self.efficient = efficient
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64) if self.use_bn else lambda x: x
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        upsamples = []
        self.layer1 = self._make_layer(block, 64, layers[0])
        upsamples += [
            Upsample(n_classes, self.inplanes, n_classes, use_bn=self.use_bn, k=k_up)
        ]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        upsamples += [
            Upsample(n_classes, self.inplanes, n_classes, use_bn=self.use_bn, k=k_up)
        ]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        upsamples += [
            Upsample(n_classes, self.inplanes, n_classes, use_bn=self.use_bn, k=k_up)
        ]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fine_tune = [
            self.conv1,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        if self.use_bn:
            self.fine_tune += [self.bn1]

        num_levels = 3
        self.spp_size = n_classes
        bt_size = self.spp_size

        level_size = self.spp_size // num_levels

        self.dsn = dsn(256, self.n_classes)

        self.spp = SpatialPyramidPooling(
            self.inplanes,
            num_levels,
            bt_size=bt_size,
            level_size=level_size,
            out_size=self.spp_size,
            grids=spp_grids,
            square_grid=spp_square_grid,
            bn_momentum=0.01 / 2,
            use_bn=self.use_bn,
        )
        self.upsample = nn.ModuleList(list(reversed(upsamples)))

        self.random_init = [self.spp, self.upsample]

        self.n_classes = n_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = [
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            ]
            if self.use_bn:
                layers += [nn.BatchNorm2d(planes * block.expansion)]
            downsample = nn.Sequential(*layers)
        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                efficient=self.efficient,
                use_bn=self.use_bn,
            )
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers += [
                block(
                    self.inplanes, planes, efficient=self.efficient, use_bn=self.use_bn
                )
            ]

        return nn.Sequential(*layers)

    def forward_resblock(self, x, layers):
        skip = None
        for l in layers:
            x = l(x)
            if isinstance(x, tuple):
                x, skip = x
        return x, skip

    def forward_down(self, image):
        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = []
        x, skip = self.forward_resblock(x, self.layer1)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer2)
        features += [skip]
        x, skip = self.forward_resblock(x, self.layer3)
        features += [skip]

        dsn = None
        if self.training:
            dsn = self.dsn(x)
        x, skip = self.forward_resblock(x, self.layer4)

        features += [self.spp.forward(skip)]
        if self.training:
            return features, dsn
        else:
            return features

    def forward_up(self, features):
        features = features[::-1]

        x = features[0]

        upsamples = []
        for skip, up in zip(features[1:], self.upsample):
            x = up(x, skip)
            upsamples += [x]

        return [x]

    def forward(self, x):
        dsn = None
        if self.training:
            features, dsn = self.forward_down(x)
        else:
            features = self.forward_down(x)

        res = self.forward_up(features)

        if self.training:
            res.append(dsn)
            return res

        return res[-1]


@SEG_MODELS_REGISTRY.register()
def SwiftNetRes18(in_channels=1, n_classes=3, pretrained=True, **kwargs):

    model = SwiftNetResNet(
        BasicBlock, [2, 2, 2, 2], in_channels=in_channels, n_classes=n_classes, **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)

    return model
