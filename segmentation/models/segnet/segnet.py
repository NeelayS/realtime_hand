import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=2, p_dropout=0.5):
        super(Encoder, self).__init__()

        encoder = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )

        if n_blocks > 1:
            encoder.extend(
                [
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if n_blocks == 3:
                encoder.append(nn.Dropout(p_dropout))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):

        out = self.encoder(x)

        return F.max_pool2d(out, 2, 2, return_indices=True), out.size()


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=2, p_dropout=0.5):
        super(Decoder, self).__init__()

        decoder = nn.ModuleList(
            [
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ]
        )

        if n_blocks > 1:
            decoder.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            if n_blocks == 3:
                decoder.append(nn.Dropout(p_dropout))

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x, indices, size):

        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)

        return self.decoder(unpooled)


class SegNet(nn.Module):
    def __init__(
        self,
        n_classes=3,
        in_channels=1,
        p_dropout=0.5,
        filter_config=(64, 128, 256, 512, 512),
    ):
        super(SegNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        n_encoder_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (in_channels,) + filter_config

        n_decoder_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):

            self.encoder.append(
                Encoder(
                    encoder_filter_config[i],
                    encoder_filter_config[i + 1],
                    n_encoder_layers[i],
                    p_dropout,
                )
            )

            self.decoder.append(
                Decoder(
                    decoder_filter_config[i],
                    decoder_filter_config[i + 1],
                    n_decoder_layers[i],
                    p_dropout,
                )
            )

        self.classifier = nn.Conv2d(filter_config[0], n_classes, 3, 1, 1)

    def forward(self, x):

        indices = []
        unpool_sizes = []
        features = x

        for i in range(0, 5):
            (features, ind), size = self.encoder[i](features)
            indices.append(ind)
            unpool_sizes.append(size)

        for i in range(0, 5):
            features = self.decoder[i](features, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(features)
