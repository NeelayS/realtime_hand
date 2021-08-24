import torch
import torch.nn as nn
import torch.nn.functional as F

from ..criterion import OhemCrossEntropy2dTensor


class CriterionDFANet(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionDFANet, self).__init__()

        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh=thresh, min_kept=min_kept
        )
        self.criterion2 = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduce=reduce
        )

        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.interpolate(
            input=preds[0], size=(h, w), mode="bilinear", align_corners=True
        )
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.interpolate(
            input=preds[1], size=(h, w), mode="bilinear", align_corners=True
        )
        loss2 = self.criterion1(scale_pred, target)

        scale_pred = F.interpolate(
            input=preds[2], size=(h, w), mode="bilinear", align_corners=True
        )
        loss3 = self.criterion1(scale_pred, target)

        return loss1 + 0.4 * loss2 + 0.4 * loss3
