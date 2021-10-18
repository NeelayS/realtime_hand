import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBnRelu, SeparableConv2d, dsn
from .retrieve import SEG_MODELS_REGISTRY


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


@SEG_MODELS_REGISTRY.register()
class DFANet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, **kwargs):
        super(DFANet, self).__init__()

        self.backbone = XceptionA(in_channels=in_channels)

        self.enc2_2 = Enc(240, 48, 4, **kwargs)
        self.enc3_2 = Enc(144, 96, 6, **kwargs)
        self.enc4_2 = Enc(288, 192, 4, **kwargs)
        self.fca_2 = FCAttention(192, **kwargs)

        self.enc2_3 = Enc(240, 48, 4, **kwargs)
        self.enc3_3 = Enc(144, 96, 6, **kwargs)
        self.enc3_4 = Enc(288, 192, 4, **kwargs)
        self.fca_3 = FCAttention(192, **kwargs)

        self.enc2_1_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.enc2_2_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.enc2_3_reduce = ConvBnRelu(48, 32, 1, **kwargs)
        self.conv_fusion = ConvBnRelu(32, 32, 1, **kwargs)

        self.fca_1_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.fca_2_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.fca_3_reduce = ConvBnRelu(192, 32, 1, **kwargs)
        self.conv_out = nn.Conv2d(32, n_classes, 1)

        self.dsn1 = dsn(192, n_classes)
        self.dsn2 = dsn(192, n_classes)

        self.__setattr__(
            "exclusive",
            [
                "enc2_2",
                "enc3_2",
                "enc4_2",
                "fca_2",
                "enc2_3",
                "enc3_3",
                "enc3_4",
                "fca_3",
                "enc2_1_reduce",
                "enc2_2_reduce",
                "enc2_3_reduce",
                "conv_fusion",
                "fca_1_reduce",
                "fca_2_reduce",
                "fca_3_reduce",
                "conv_out",
            ],
        )

    def forward(self, x):
        # backbone
        stage1_conv1 = self.backbone.conv1(x)
        stage1_enc2 = self.backbone.enc2(stage1_conv1)
        stage1_enc3 = self.backbone.enc3(stage1_enc2)
        stage1_enc4 = self.backbone.enc4(stage1_enc3)
        stage1_fca = self.backbone.fca(stage1_enc4)
        stage1_out = F.interpolate(
            stage1_fca, scale_factor=4, mode="bilinear", align_corners=True
        )

        dsn1 = self.dsn1(stage1_out)
        # stage2
        stage2_enc2 = self.enc2_2(torch.cat([stage1_enc2, stage1_out], dim=1))
        stage2_enc3 = self.enc3_2(torch.cat([stage1_enc3, stage2_enc2], dim=1))
        stage2_enc4 = self.enc4_2(torch.cat([stage1_enc4, stage2_enc3], dim=1))
        stage2_fca = self.fca_2(stage2_enc4)
        stage2_out = F.interpolate(
            stage2_fca, scale_factor=4, mode="bilinear", align_corners=True
        )

        dsn2 = self.dsn2(stage2_out)

        # stage3
        stage3_enc2 = self.enc2_3(torch.cat([stage2_enc2, stage2_out], dim=1))
        stage3_enc3 = self.enc3_3(torch.cat([stage2_enc3, stage3_enc2], dim=1))
        stage3_enc4 = self.enc3_4(torch.cat([stage2_enc4, stage3_enc3], dim=1))
        stage3_fca = self.fca_3(stage3_enc4)

        stage1_enc2_decoder = self.enc2_1_reduce(stage1_enc2)
        stage2_enc2_docoder = F.interpolate(
            self.enc2_2_reduce(stage2_enc2),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        stage3_enc2_decoder = F.interpolate(
            self.enc2_3_reduce(stage3_enc2),
            scale_factor=4,
            mode="bilinear",
            align_corners=True,
        )
        fusion = stage1_enc2_decoder + stage2_enc2_docoder + stage3_enc2_decoder
        fusion = self.conv_fusion(fusion)

        stage1_fca_decoder = F.interpolate(
            self.fca_1_reduce(stage1_fca),
            scale_factor=4,
            mode="bilinear",
            align_corners=True,
        )
        stage2_fca_decoder = F.interpolate(
            self.fca_2_reduce(stage2_fca),
            scale_factor=8,
            mode="bilinear",
            align_corners=True,
        )
        stage3_fca_decoder = F.interpolate(
            self.fca_3_reduce(stage3_fca),
            scale_factor=16,
            mode="bilinear",
            align_corners=True,
        )
        fusion = fusion + stage1_fca_decoder + stage2_fca_decoder + stage3_fca_decoder

        outputs = list()
        out = self.conv_out(fusion)
        outputs.append(out)
        outputs.append(dsn1)
        outputs.append(dsn2)

        return outputs
