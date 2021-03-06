import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import Registry

SEG_CRITERION_REGISTRY = Registry("CRITERION")


@SEG_CRITERION_REGISTRY.register()
class OhemCrossEntropy2dTensor(nn.Module):
    def __init__(
        self,
        ignore_label,
        reduction="elementwise_mean",
        thresh=0.6,
        min_kept=256,
        down_ratio=1,
        use_weight=False,
    ):
        super(OhemCrossEntropy2dTensor, self).__init__()

        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio

        if use_weight:

            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            )

            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=reduction, weight=weight, ignore_index=ignore_label
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction=reduction, ignore_index=ignore_label
            )

    def forward(self, pred, target):

        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Labels: {}".format(num_valid))

        elif num_valid > 0:

            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]

            threshold = self.thresh

            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]

                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]

                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


@SEG_CRITERION_REGISTRY.register()
class CriterionDSN(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255, reduce=True):
        super(CriterionDSN, self).__init__()

        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, preds, target):

        H, W = target.shape[-2:]

        scale_pred = F.interpolate(
            input=preds[0], size=(H, W), mode="bilinear", align_corners=True
        )
        loss1 = super(CriterionDSN, self).forward(scale_pred, target)

        scale_pred = F.interpolate(
            input=preds[-1], size=(H, W), mode="bilinear", align_corners=True
        )
        loss2 = super(CriterionDSN, self).forward(scale_pred, target)

        return loss1 + loss2 * 0.4


@SEG_CRITERION_REGISTRY.register()
class CriterionOhemDSN(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionOhemDSN, self).__init__()

        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh=thresh, min_kept=min_kept
        )
        self.criterion2 = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduce=reduce
        )

    def forward(self, preds, target):

        H, W = target.shape[-2:]

        scale_pred = F.interpolate(
            input=preds[0], size=(H, W), mode="bilinear", align_corners=True
        )
        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.interpolate(
            input=preds[1], size=(H, W), mode="bilinear", align_corners=True
        )
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2 * 0.4


@SEG_CRITERION_REGISTRY.register()
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


@SEG_CRITERION_REGISTRY.register()
class CriterionICNet(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionICNet, self).__init__()

        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh=thresh, min_kept=min_kept
        )

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

        scale_pred = F.interpolate(
            input=preds[3], size=(h, w), mode="bilinear", align_corners=True
        )
        loss4 = self.criterion1(scale_pred, target)

        return loss1 + 0.4 * loss2 + 0.4 * loss3 + 0.4 * loss4


@SEG_CRITERION_REGISTRY.register()
class ModCriterionICNet(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(ModCriterionICNet, self).__init__()

        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh=thresh, min_kept=min_kept
        )

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


SEG_MODEL_CRITERIONS = {
    "BiSeNet": "CriterionDFANet",
    "DFANet": "CriterionDFANet",
    "DFSegNet": "CriterionDSN",
    "DFSegNetV1": "CriterionDSN",
    "DFSegNetV2": "CriterionDSN",
    "ESPNet": "CriterionDSN",
    "FastSCNN": "CriterionDSN",
    "ICNet": "ModCriterionICNet",
    "CustomICNet": "CriterionICNet",
    "SwiftNetRes18": "CriterionDSN",
    "SwiftNetResNet": "CriterionDSN",
}
