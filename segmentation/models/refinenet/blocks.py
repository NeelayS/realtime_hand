import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, bias=False):

    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):

    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias
    )


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_stages):
        super(CRPBlock, self).__init__()

        for i in range(n_stages):
            if i == 0:
                setattr(
                    self,
                    f'{i+1}_{"outvar_dimred"}',
                    conv3x3(
                        in_planes,
                        out_planes,
                        stride=1,
                        bias=False,
                    ),
                )
            else:
                setattr(
                    self,
                    f'{i+1}_{"outvar_dimred"}',
                    conv3x3(
                        out_planes,
                        out_planes,
                        stride=1,
                        bias=False,
                    ),
                )

        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):

        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, f'{i+1}_{"outvar_dimred"}')(top)
            x = top + x

        return x


stages_suffixes = {0: "_conv", 1: "_conv_relu_varout_dimred"}


class RCUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()

        for i in range(n_blocks):
            for j in range(n_stages):

                if i == 0 and j == 0:
                    setattr(
                        self,
                        f"{i+1}{stages_suffixes[j]}",
                        conv3x3(in_planes, out_planes, stride=1, bias=(j == 0)),
                    )

                else:
                    setattr(
                        self,
                        f"{i+1}{stages_suffixes[j]}",
                        conv3x3(out_planes, out_planes, stride=1, bias=(j == 0)),
                    )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, f"{i+1}{stages_suffixes[j]}")(x)
            x += residual
        return x
