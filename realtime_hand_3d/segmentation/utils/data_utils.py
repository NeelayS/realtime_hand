import cv2 as cv
import numpy as np
import os
import shutil
import torch
from torchvision import io


def resize_image(
    image, expected_size, pad_value, ret_params=True, mode=cv.INTER_LINEAR
):

    """
    image (ndarray) with either shape of [H,W,3] for RGB or [H,W] for grayscale.
    Padding is added so that the content of image is in the center.
    """

    h, w = image.shape[:2]
    if w > h:
        w_new = int(expected_size)
        h_new = int(h * w_new / w)
        image = cv.resize(image, (w_new, h_new), interpolation=mode)

        pad_up = (w_new - h_new) // 2
        pad_down = w_new - h_new - pad_up
        if len(image.shape) == 3:
            pad_width = ((pad_up, pad_down), (0, 0), (0, 0))
            constant_values = ((pad_value, pad_value), (0, 0), (0, 0))
        elif len(image.shape) == 2:
            pad_width = ((pad_up, pad_down), (0, 0))
            constant_values = ((pad_value, pad_value), (0, 0))

        image = np.pad(
            image, pad_width=pad_width, mode="constant", constant_values=constant_values
        )
        if ret_params:
            return image, pad_up, 0, h_new, w_new
        else:
            return image

    elif w < h:
        h_new = int(expected_size)
        w_new = int(w * h_new / h)
        image = cv.resize(image, (w_new, h_new), interpolation=mode)

        pad_left = (h_new - w_new) // 2
        pad_right = h_new - w_new - pad_left
        if len(image.shape) == 3:
            pad_width = ((0, 0), (pad_left, pad_right), (0, 0))
            constant_values = ((0, 0), (pad_value, pad_value), (0, 0))
        elif len(image.shape) == 2:
            pad_width = ((0, 0), (pad_left, pad_right))
            constant_values = ((0, 0), (pad_value, pad_value))

        image = np.pad(
            image, pad_width=pad_width, mode="constant", constant_values=constant_values
        )
        if ret_params:
            return image, 0, pad_left, h_new, w_new
        else:
            return image

    else:
        image = cv.resize(image, (expected_size, expected_size), interpolation=mode)
        if ret_params:
            return image, 0, 0, expected_size, expected_size
        else:
            return image


def preprocessing(image, expected_size=224, pad_value=0):

    """
    Pre-processing steps to use pre-trained model on images
    """

    imgnet_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
    imgnet_std = np.array([0.229, 0.224, 0.225])[None, None, :]

    image, pad_up, pad_left, h_new, w_new = resize_image(
        image, expected_size, pad_value, ret_params=True
    )

    image = image.astype(np.float32) / 255.0
    image = (image - imgnet_mean) / imgnet_std

    X = np.transpose(image, axes=(2, 0, 1))
    X = np.expand_dims(X, axis=0)
    X = torch.tensor(X, dtype=torch.float32)

    return X, pad_up, pad_left, h_new, w_new


def gen_e2h_eval_masks(root_dir):

    for seq_dir in os.listdir(root_dir):

        if seq_dir[:4] != "eval":
            continue

        print("Processing directory", seq_dir)

        seq = seq_dir.split("_")[1][3:]
        os.makedirs(os.path.join("imgs", seq), exist_ok=True)
        os.makedirs(os.path.join("masks", seq), exist_ok=True)

        for img_path in os.listdir(seq_dir):

            if (
                img_path.split(".")[0][-4:] == "_e_l"
                or img_path.split(".")[0][-4:] == "_e_r"
                or img_path.split(".")[0][-4:] == "_seg"
            ):
                continue

            img = io.read_image(os.path.join(seq_dir, img_path))
            shutil.copy(
                os.path.join(seq_dir, img_path), os.path.join("imgs", seq, img_path)
            )

            l_mask = io.read_image(
                os.path.join(seq_dir, img_path.replace(".png", "_e_l.png"))
            )
            r_mask = io.read_image(
                os.path.join(seq_dir, img_path.replace(".png", "_e_r.png"))
            )

            mask = torch.zeros_like(l_mask)
            mask[l_mask == 255] = 127
            mask[r_mask == 255] = 255

            io.write_png(mask, os.path.join("masks", seq, img_path))
