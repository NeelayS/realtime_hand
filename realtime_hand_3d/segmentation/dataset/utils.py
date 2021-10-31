import math
import random

import cv2
import numpy as np

from ..utils import *


def seg_augmentation_wo_kpts(img, seg):

    img_h, img_w = img.shape[:2]
    fg_mask = seg.copy()

    coords1 = np.where(fg_mask)
    img_top, img_bot = np.min(coords1[0]), np.max(coords1[0])

    shift_range_ratio = 0.2

    down_shift = True if not fg_mask[0, :].any() else False
    if down_shift:
        down_space = int((img_h - img_top) * shift_range_ratio)
        old_bot = img_h
        down_offset = random.randint(0, down_space)
        old_bot -= down_offset

        old_top = 0
        cut_height = old_bot - old_top

        new_bot = img_h
        new_top = new_bot - cut_height
    else:
        old_bot, old_top = img_h, 0
        new_bot, new_top = old_bot, old_top

    coords2 = np.where(fg_mask[old_top:old_bot, :])
    img_left, img_right = np.min(coords2[1]), np.max(coords2[1])

    left_shift = True if not fg_mask[old_top:old_bot, -1].any() else False
    right_shift = True if not fg_mask[old_top:old_bot, 0].any() else False
    if left_shift and right_shift:
        if random.random() > 0.5:
            right_shift = False
        else:
            left_shift = False

    if left_shift:
        left_space = int(img_right * shift_range_ratio)
        old_left = 0
        left_offset = random.randint(0, left_space)
        old_left += left_offset

        old_right = img_w
        cut_width = old_right - old_left

        new_left = 0
        new_right = new_left + cut_width

    if right_shift:
        right_space = int((img_w - img_left) * shift_range_ratio)
        old_right = img_w
        right_offset = random.randint(0, right_space)
        old_right -= right_offset

        old_left = 0
        cut_width = old_right - old_left

        new_right = img_w
        new_left = new_right - cut_width

    if not (left_shift or right_shift):
        old_left, old_right = 0, img_w
        new_left, new_right = old_left, old_right

    img_new = np.zeros_like(img)
    seg_new = np.zeros_like(seg)

    img_new[new_top:new_bot, new_left:new_right] = img[
        old_top:old_bot, old_left:old_right
    ]
    seg_new[new_top:new_bot, new_left:new_right] = seg[
        old_top:old_bot, old_left:old_right
    ]

    return img_new, seg_new


def random_bg_augment(img, img_path="", brightness_aug=True, flip_aug=True):

    if brightness_aug:
        brightness_val = random.randint(50, 225)
        img = change_mean_brightness(img, None, brightness_val, 20, img_path)

    img = img.astype("uint8")

    if flip_aug:
        do_flip = bool(random.getrandbits(1))
        if do_flip:
            img = cv2.flip(img, 1)

    return img


def resize_bg(fg_shape, bg_img):

    fg_h, fg_w = fg_shape[:2]
    bg_h, bg_w = bg_img.shape[:2]

    if bg_h < fg_h or bg_w < fg_w:
        fb_h_ratio = float(fg_h) / bg_h
        fb_w_ratio = float(fg_w) / bg_w
        bg_resize_ratio = max(fb_h_ratio, fb_w_ratio)
        bg_img = cv2.resize(
            bg_img,
            (
                int(math.ceil(bg_img.shape[1] * bg_resize_ratio)),
                int(math.ceil(bg_img.shape[0] * bg_resize_ratio)),
            ),
        )
    bg_h, bg_w = bg_img.shape[:2]

    bg_h_offset_range = max(bg_h - fg_h, 0)
    bg_w_offset_range = max(bg_w - fg_w, 0)

    bg_h_offset = random.randint(0, bg_h_offset_range)
    bg_w_offset = random.randint(0, bg_w_offset_range)
    bg_img = bg_img[
        bg_h_offset : bg_h_offset + fg_h, bg_w_offset : bg_w_offset + fg_w, :3
    ]

    return bg_img


def add_alpha_image_to_bg(alpha_img, bg_img):

    alpha_s = np.repeat((alpha_img[:, :, 3] / 255.0)[:, :, np.newaxis], 3, axis=2)
    alpha_l = 1.0 - alpha_s
    combined_img = np.multiply(alpha_s, alpha_img[:, :, :3]) + np.multiply(
        alpha_l, bg_img
    )

    return combined_img


def add_alpha_border(hand_img):

    fg_mask = (hand_img[:, :, -1] == 0).astype(np.uint8)
    fg_mask = cv2.dilate(fg_mask, np.ones((3, 3)))
    alpha_mask = fg_mask * 255
    alpha_mask = 255 - cv2.GaussianBlur(alpha_mask, (7, 7), 0)
    hand_img[:, :, -1] = alpha_mask
    hand_seg = alpha_mask > 200
    hand_all_seg = alpha_mask > 0

    return hand_img, hand_seg, hand_all_seg


def merge_hands(top_hand_img, bot_hand_img, bg_img, bg_resize=True):

    if top_hand_img is not None and bot_hand_img is not None:
        bot_hand_img, _, _ = add_alpha_border(bot_hand_img)
        top_hand_img, _, _ = add_alpha_border(top_hand_img)
        bg_img_resized = resize_bg(bot_hand_img.shape, bg_img) if bg_resize else bg_img
        combined_hand_img = add_alpha_image_to_bg(bot_hand_img, bg_img_resized)
        combined_hand_img = add_alpha_image_to_bg(top_hand_img, combined_hand_img)
    else:
        top_hand_img, _, _ = add_alpha_border(top_hand_img)
        bg_img_resized = resize_bg(top_hand_img.shape, bg_img) if bg_resize else bg_img
        combined_hand_img = add_alpha_image_to_bg(top_hand_img, bg_img_resized)

    return combined_hand_img, bg_img_resized


def change_mean_brightness(img, seg, brightness_val, jitter_range=20, img_path=""):

    if seg is not None:
        old_mean_val = np.mean(img[seg])
    else:
        old_mean_val = np.mean(img)

    assert old_mean_val != 0, f"ERROR: {img_path} has mean of 0"

    new_mean_val = brightness_val + random.uniform(-jitter_range / 2, jitter_range / 2)
    img *= new_mean_val / old_mean_val
    img = np.clip(img, 0, 255)

    return img


def random_smoothness(img, smooth_rate=0.3):

    smooth_rate_tick = smooth_rate / 5
    rand_val = random.random()
    if rand_val < smooth_rate:
        if rand_val < smooth_rate_tick:
            kernel_size = 3
        elif rand_val < smooth_rate_tick * 2:
            kernel_size = 5
        elif rand_val < smooth_rate_tick * 3:
            kernel_size = 7
        elif rand_val < smooth_rate_tick * 4:
            kernel_size = 9
        else:
            kernel_size = 11
        img[:, :, :3] = cv2.blur(img[:, :, :3], (kernel_size, kernel_size))

    return img


def normalize_tensor(tensor, mean, std):

    for t in tensor:
        t.sub_(mean).div_(std)

    return tensor


def gen_e2h_mask(left_seg_path, right_seg_path):

    left_seg = cv2.imread(left_seg_path, cv2.IMREAD_UNCHANGED)
    right_seg = cv2.imread(right_seg_path, cv2.IMREAD_UNCHANGED)

    left_seg = np.expand_dims(left_seg, axis=2)
    right_seg = np.expand_dims(right_seg, axis=2)

    left_seg = np.concatenate((left_seg, left_seg, left_seg), axis=2)
    right_seg = np.concatenate((right_seg, right_seg, right_seg), axis=2)

    return left_seg, right_seg
