import os
import numpy as np
from PIL import Image
from scipy.ndimage.morphology import distance_transform_edt
import cv2 as cv
import torch


def normalize_tensor(tensor, mean, std):

    for t in tensor:
        t.sub_(mean).div_(std)

    return tensor


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_frames_as_images(video_path, save_dir):

    video_name = video_path.split("/")[-1][:-4]

    cap = cv.VideoCapture(video_path)

    frame_i = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break

        cv.imwrite(os.path.join(save_dir, video_name + "_" + str(frame_i)), frame)
        frame_i += 1

    print(f"{frame_i+1} images generated")
    cap.release()


def create_video_from_images(img_dir, save_path, fps=25, img_extension="png"):

    if img_extension == "png":
        command = f"ffmpeg -r {fps} -i {img_dir}/*.png -y {save_path}"
    else:
        command = f"ffmpeg -r {fps} -i {img_dir}/*.jpg -y {save_path}"

    # cat *.png | ffmpeg -f image2pipe -i - output.mp4

    os.system(command)


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    mask = [mask == i for i in range(num_classes)]
    mask = torch.stack(mask, dim=1)
    return mask.numpy()


def onehot_to_mask(mask):
    """
    Converts a mask (K,H,W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0)
    _mask[_mask != 0] += 1
    return _mask


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)
    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    print(mask.shape)
    mask_pad = np.pad(
        mask, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0
    )

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(
            1.0 - mask_pad[i, :]
        )
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    print(mask.shape)
    mask_pad = np.pad(
        mask[0], ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=0
    )
    edgemap = np.zeros(mask.shape[2:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(
            1.0 - mask_pad[i, :]
        )
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        print(edgemap.shape, dist.shape)
        edgemap += dist

    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)

    return torch.from_numpy(edgemap)
