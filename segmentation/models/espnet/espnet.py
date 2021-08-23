import torch
import torch.nn as nn

from .blocks import (
    CBR,
    BR,
    InputProjectionA,
    DownSamplerB,
    DilatedParllelResidualBlockB,
    C,
)


class ESPNet_Encoder(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, p=5, q=3):
        super().__init__()

        self.level1 = CBR(in_channels, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = DownSamplerB(16 + 3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64, 64))
        self.b2 = BR(128 + 3)

        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128, 128))
        self.b3 = BR(256)

        self.classifier = C(256, n_classes, 1, 1)

    def forward(self, input):

        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        return classifier


class ESPNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=3, p=2, q=3, encoderFile=None):
        super().__init__()

        self.encoder = ESPNet_Encoder(in_channels, n_classes, p, q)
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print("Encoder loaded!")
        # load the encoder modules
        self.modules = []
        for i, m in enumerate(self.encoder.children()):
            self.modules.append(m)

        # light-weight decoder
        self.level3_C = C(128 + 3, n_classes, 1, 1)
        self.br = nn.BatchNorm2d(n_classes, eps=1e-03)
        self.conv = CBR(19 + n_classes, n_classes, 3, 1)

        self.up_l3 = nn.Sequential(
            nn.ConvTranspose2d(
                n_classes,
                n_classes,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            )
        )
        self.combine_l2_l3 = nn.Sequential(
            BR(2 * n_classes),
            DilatedParllelResidualBlockB(2 * n_classes, n_classes, add=False),
        )

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(
                n_classes,
                n_classes,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            ),
            BR(n_classes),
        )

        self.classifier = nn.ConvTranspose2d(
            n_classes, n_classes, 2, stride=2, padding=0, output_padding=0, bias=False
        )

    def forward(self, input):

        output0 = self.modules[0](input)
        inp1 = self.modules[1](input)
        inp2 = self.modules[2](input)

        output0_cat = self.modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.modules[6](torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.modules[9](
            torch.cat([output2_0, output2], 1)
        )  # concatenate for feature map width expansion

        output2_c = self.up_l3(self.br(self.modules[10](output2_cat)))  # RUM

        output1_C = self.level3_C(output1_cat)  # project to C-dimensional space
        comb_l2_l3 = self.up_l2(
            self.combine_l2_l3(torch.cat([output1_C, output2_c], 1))
        )  # RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.classifier(concat_features)

        out = []
        out.append(classifier)

        return out
