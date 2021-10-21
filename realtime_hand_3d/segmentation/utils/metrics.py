import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou(pred, target, is_idx=None):

    n_classes = len(np.unique(target.cpu().data.numpy()))

    if n_classes == 1:
        pred_unique_np = np.unique(pred.cpu().data.numpy())
        if len(pred_unique_np) == 1 and pred_unique_np[0] == 0:
            return np.array([1.0])
        else:
            return np.array([0.0])

    ious = []
    if not pred.shape[2] == target.shape[1]:
        pred = nn.functional.interpolate(
            pred,
            size=(target.shape[1], target.shape[2]),
            mode="bilinear",
            align_corners=True,
        )

    if not is_idx:
        pred = torch.argmax(pred, dim=1)

    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(1, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = (
            pred_inds.long().sum().data.cpu().item()
            + target_inds.long().sum().data.cpu().item()
            - intersection
        )
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(float(intersection) / float(union))

    return np.array(ious)
