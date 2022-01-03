import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import cv2 as cv

from ..utils import Registry, mask_to_onehot, onehot_to_binary_edges

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


@SEG_CRITERION_REGISTRY.register()
class CriterionModBiSeNet(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super().__init__()

        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(
            ignore_index, thresh=thresh, min_kept=min_kept
        )
        self.criterion2 = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduce=reduce
        )

    def bce2d(self, input, target):
        n, c, h, w = input.size()

        print(input.shape, target.shape)

        log_p = input.contiguous().view(1, -1)
        target_t = target.contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = target_t == 1
        neg_index = target_t == 0
        ignore_index = target_t > 1

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(target_t.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num

        print(weight.shape, pos_index.shape, neg_index.shape, ignore_index.shape)

        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, reduction="mean"
        )

        print("done")

        return loss

    def mask_to_onehot(self, mask, num_classes):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector
        """
        mask = [mask == i for i in range(num_classes)]
        mask = torch.stack(mask, dim=1)
        return mask.numpy()

    def onehot_to_binary_edges(self, mask, radius, num_classes):
        """
        Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
        """

        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        print(mask.shape)
        mask_pad = np.pad(
            mask, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="constant", constant_values=0
        )
        edgemap = np.zeros(mask.shape[2:])

        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(
                1.0 - mask_pad[i, :]
            )
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            print(edgemap.shape, dist.shape)
            edgemap += dist[0]

        # edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)

        return torch.from_numpy(edgemap)

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

        scale_edge = F.interpolate(
            input=preds[3], size=(h, w), mode="bilinear", align_corners=True
        )
        print(scale_edge.shape)

        edge_target = target.clone()
        print(edge_target.shape)
        edge_target = self.mask_to_onehot(edge_target, 3)
        print(edge_target.shape)
        edge_target = self.onehot_to_binary_edges(edge_target, 2, 3)
        print(edge_target.shape)

        loss4 = self.bce2d(scale_edge, edge_target)

        return loss1 + 0.4 * loss2 + 0.4 * loss3 + loss4


SEG_MODEL_CRITERIONS = {
    "BiSeNet": "CriterionDFANet",
    "ModBiSeNet": "CriterionModBiSeNet",
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
