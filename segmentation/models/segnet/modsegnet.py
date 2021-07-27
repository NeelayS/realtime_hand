import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout=0):
        super(Encoder, self).__init__()

        encoder = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.ReLU(inplace=True),
            ]
        )

        if p_dropout > 0:
            encoder.append(nn.Dropout(p_dropout))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.encoder = Encoder(in_channels * 2, out_channels)

    def forward(self, x, encoder_features, indices, size):

        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        features = torch.cat([unpooled, encoder_features], 1)
        features = self.encoder(features)

        return features


class ModSegNet(nn.Module):
    def __init__(
        self,
        n_classes=2,
        in_channels=3,
        p_dropout=0.5,
        filter_config=(32, 64, 128, 256),
    ):
        super(ModSegNet, self).__init__()

        self.encoder1 = Encoder(in_channels, filter_config[0])
        self.encoder2 = Encoder(filter_config[0], filter_config[1])
        self.encoder3 = Encoder(filter_config[1], filter_config[2], p_dropout)
        self.encoder4 = Encoder(filter_config[2], filter_config[3], p_dropout)

        self.decoder1 = Decoder(filter_config[3], filter_config[2])
        self.decoder2 = Decoder(filter_config[2], filter_config[1])
        self.decoder3 = Decoder(filter_config[1], filter_config[0])
        self.decoder4 = Decoder(filter_config[0], filter_config[0])

        self.classifier = nn.Conv2d(filter_config[0], n_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, x):

        encoder_features_1 = self.encoder1(x)
        size_2 = encoder_features_1.size()
        encoder_features_2, ind_2 = F.max_pool2d(
            encoder_features_1, 2, 2, return_indices=True
        )

        encoder_features_2 = self.encoder2(encoder_features_2)
        size_3 = encoder_features_2.size()
        encoder_features_3, ind_3 = F.max_pool2d(
            encoder_features_2, 2, 2, return_indices=True
        )

        encoder_features_3 = self.encoder3(encoder_features_3)
        size_4 = encoder_features_3.size()
        encoder_features_4, ind_4 = F.max_pool2d(
            encoder_features_3, 2, 2, return_indices=True
        )

        encoder_features_4 = self.encoder4(encoder_features_4)
        size_5 = encoder_features_4.size()
        encoder_features_5, ind_5 = F.max_pool2d(
            encoder_features_4, 2, 2, return_indices=True
        )

        decoder_features = self.decoder1(
            encoder_features_5, encoder_features_4, ind_5, size_5
        )
        decoder_features = self.decoder2(
            decoder_features, encoder_features_3, ind_4, size_4
        )
        decoder_features = self.decoder3(
            decoder_features, encoder_features_2, ind_3, size_3
        )
        decoder_features = self.decoder4(
            decoder_features, encoder_features_1, ind_2, size_2
        )

        return self.classifier(decoder_features)
