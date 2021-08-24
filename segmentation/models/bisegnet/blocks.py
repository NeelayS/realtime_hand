import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import ConvBnRelu


class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):

        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(
            in_planes,
            out_planes,
            3,
            1,
            1,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                out_planes,
                out_planes,
                1,
                1,
                0,
                has_bn=True,
                norm_layer=norm_layer,
                has_relu=False,
                has_bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):

        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()

        self.conv_1x1 = ConvBnRelu(
            in_planes,
            out_planes,
            1,
            1,
            0,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                out_planes,
                out_planes // reduction,
                1,
                1,
                0,
                has_bn=False,
                norm_layer=norm_layer,
                has_relu=True,
                has_bias=False,
            ),
            ConvBnRelu(
                out_planes // reduction,
                out_planes,
                1,
                1,
                0,
                has_bn=False,
                norm_layer=norm_layer,
                has_relu=False,
                has_bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):

        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se

        return output
