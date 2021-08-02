import numpy as np
import cv2


def index_to_color(img_idx, unsure_idx, palette):

    img_color = np.zeros(img_idx.shape + (3,), dtype=np.uint8)
    idx_list = np.unique(img_idx).astype(np.uint8)
    for idx in idx_list:
        if idx != unsure_idx:
            img_color[img_idx == idx] = palette[idx][::-1]
        else:
            img_color[img_idx == idx] = palette[-1][::-1]

    return img_color


def visualize_seg_detection(img, seg):

    palette = [(0, 0, 0), (128, 0, 0), (0, 128, 0)]
    alpha = 0.5
    img_vis = img.copy()

    seg_vis = index_to_color(seg, 255, palette)
    seg_positives = seg > 0
    img_vis[seg_positives] = (
        img[seg_positives] * alpha + seg_vis[seg_positives] * (1 - alpha)
    ).astype("uint8")

    return img_vis


def draw_matting(image, mask):

    mask = 255 * (1.0 - mask)
    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, 3))
    mask = mask.astype(np.uint8)
    image_matting = cv2.add(image, mask)

    return image_matting


def draw_fore_to_back(image, mask, background, kernel_sz=13, sigma=0):

    mask_filtered = cv2.GaussianBlur(mask, (kernel_sz, kernel_sz), sigma)
    mask_filtered = np.expand_dims(mask_filtered, axis=2)
    mask_filtered = np.tile(mask_filtered, (1, 1, 3))
    image_alpha = image * mask_filtered + background * (1 - mask_filtered)

    return image_alpha.astype(np.uint8)
