import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=False):
        super().__init__()

        self.batchnorm = batchnorm

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            self._get_bn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            self._get_bn(out_channels),
            nn.ReLU(inplace=True),
        )

    def _get_bn(self, channels):
        if self.batchnorm:
            return nn.BatchNorm2d(channels)
        else:
            return nn.Sequential()

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()

        self.encoder = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        features = []
        for block in self.encoder:
            x = block(x)
            features.append(x)
            x = self.pool(x)

        return features


class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()

        self.channels = channels
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
                for i in range(len(channels) - 1)
            ]
        )
        self.decoder = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, encoder_features):

        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self._crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.decoder[i](x)

        return x

    def _crop(self, enc_ftrs, x):

        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)

        return enc_ftrs


class UNet(nn.Module):
    def __init__(
        self,
        encoder_channels=(3, 64, 128, 256, 512, 1024),
        decoder_channels=(1024, 512, 256, 128, 64),
        n_classes=2,
        retain_dim=False,
        out_size=(572, 572),
    ):
        super().__init__()

        self.encoder = Encoder(encoder_channels)
        self.decoder = Decoder(decoder_channels)
        self.head = nn.Conv2d(decoder_channels[-1], n_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):

        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)

        if self.retain_dim:
            out = F.interpolate(out, self.out_size)

        return out


def SmallUNet(n_classes=2, retain_dim=False, out_size=(572, 572)):
    return UNet(
        (3, 8, 16, 32, 64, 128), (128, 64, 32, 16, 8), n_classes, retain_dim, out_size
    )
