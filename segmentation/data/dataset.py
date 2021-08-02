import torch
from torch.utils.data import Dataset
import os
import sys

from .utils import *


class Ego2HandsDataset(Dataset):

    LEFT_IDX = 1
    RIGHT_IDX = 2

    IMG_H = 288
    IMG_W = 512

    def __init__(self, args, config, with_arms=False, seq_i=-1):

        self.args = args
        self.config = config
        self.bg_adapt = args.adapt
        self.seq_i = seq_i
        self.input_edge = args.input_edge
        self.with_arms = with_arms

        self.bg_list = self.read_bg_imgs(self.args, self.config, self.bg_adapt, seq_i)

        self.img_path_list, self.energy_path_list = self.read_hand_imgs(
            self.config.dataset_train_dir
        )

        self.valid_hand_seg_th = 5000
        self.EMPTY_IMG_ARRAY = np.zeros((1, 1))
        self.EMPTY_BOX_ARRAY = np.zeros([0, 0, 0, 0])

    def _read_hand_imgs(self, root_dir):

        img_path_list = []
        energy_path_list = []

        for root, dirs, files in os.walk(root_dir):
            for file_name in files:
                if (
                    file_name.endswith(".png")
                    and "energy" not in file_name
                    and "vis" not in file_name
                ):
                    img_path = os.path.join(root, file_name)
                    img_path_list.append(img_path)
                    energy_path_list.append(img_path.replace(".png", "_energy.png"))

        return img_path_list, energy_path_list

    def _read_bg_imgs(self, args, config, bg_adapt, seq_i):

        if not args.custom:
            if not bg_adapt:
                root_bg_dir = config.bg_all_dir
            else:
                root_bg_dir = os.path.join(
                    config.dataset_eval_dir, f"eval_seq{seq_i}_bg"
                )
        else:
            assert (
                config.custom_bg_dir != ""
            ), 'Error: custom bg dir not set. Please set "custom_bg_dir" in the config file.'
            root_bg_dir = config.custom_bg_dir

        bg_path_list = []
        for root, dirs, files in os.walk(root_bg_dir):
            for file_name in files:
                if (
                    file_name.endswith(".jpg")
                    or file_name.endswith(".png")
                    or file_name.endswith(".jpeg")
                ):
                    bg_path_list.append(os.path.join(root, file_name))

        return bg_path_list

    def __getitem__(self, index):

        if self.mode == "train":

            left_i = random.randint(0, self.__len__() - 1)
            left_img = cv2.imread(self.img_path_list[left_i], cv2.IMREAD_UNCHANGED)
            assert (
                left_img is not None
            ), f"Error, image not found: {self.img_path_list[left_i]}"

            left_img = left_img.astype(np.float32)
            left_img = cv2.resize(
                left_img, (Ego2HandsDataset.IMG_W, Ego2HandsDataset.IMG_H)
            )
            left_img = cv2.flip(left_img, 1)

            left_energy = cv2.imread(self.energy_path_list[left_i], 0)
            left_energy = (
                cv2.resize(
                    left_energy, (Ego2HandsDataset.IMG_W, Ego2HandsDataset.IMG_H)
                ).astype(np.float32)
                / 255.0
            )
            left_energy = cv2.flip(left_energy, 1)

            if self.with_arms:
                left_seg = left_img[:, :, -1] > 128
            else:
                left_seg = left_energy > 128

            left_img_orig = left_img.copy()
            left_img, left_seg = seg_augmentation_wo_kpts(left_img, left_seg)

            brightness_val = get_random_brightness_for_scene(
                self.args, self.config, self.bg_adapt, self.seq_i
            )
            left_img = change_mean_brightness(
                left_img, left_seg, brightness_val, 20, self.img_path_list[left_i]
            )
            left_img = random_smoothness(left_img)

            right_i = random.randint(0, self.__len__() - 1)
            right_img = cv2.imread(self.img_path_list[right_i], cv2.IMREAD_UNCHANGED)
            assert right_img is not None, "Error, image not found: {}".format(
                self.img_path_list[right_i]
            )

            right_img = right_img.astype(np.float32)
            right_img = cv2.resize(
                right_img, (Ego2HandsDataset.IMG_W, Ego2HandsDataset.IMG_H)
            )

            right_energy = cv2.imread(self.energy_path_list[right_i], 0)
            right_energy = (
                cv2.resize(
                    right_energy, (Ego2HandsDataset.IMG_W, Ego2HandsDataset.IMG_H)
                ).astype(np.float32)
                / 255.0
            )

            if self.with_arms:
                right_seg = right_img[:, :, -1] > 128
            else:
                right_seg = right_energy > 128

            right_img_orig = right_img.copy()

            right_img, right_seg = seg_augmentation_wo_kpts(right_img, right_seg)

            right_img = change_mean_brightness(
                right_img, right_seg, brightness_val, 20, self.img_path_list[right_i]
            )
            right_img = random_smoothness(right_img)

            bg_img = None
            while bg_img is None:
                bg_i = random.randint(0, len(self.bg_list) - 1)
                bg_img = cv2.imread(self.bg_list[bg_i]).astype(np.float32)

                if not self.bg_adapt:
                    bg_img = random_bg_augment(
                        bg_img, self.bg_list[bg_i], bg_adapt=self.bg_adapt
                    )
                else:
                    bg_img = random_bg_augment(
                        bg_img,
                        self.bg_list[bg_i],
                        bg_adapt=self.bg_adapt,
                        flip_aug=False,
                    )

                bg_img = random_smoothness(bg_img)

            merge_mode = random.randint(0, 9)
            if merge_mode < 8:
                if np.sum(left_energy) > np.sum(right_energy):
                    merge_mode = 0
                else:
                    merge_mode = 4

            if merge_mode < 4:

                img_real, bg_img_resized = merge_hands(
                    left_img, right_img, bg_img, self.bg_adapt
                )
                img_real_orig, _ = merge_hands(
                    left_img_orig, right_img_orig, bg_img_resized, self.bg_adapt, False
                )

                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[right_seg] = Ego2HandsDataset.RIGHT_IDX
                seg_real[left_seg] = Ego2HandsDataset.LEFT_IDX

                right_mask = seg_real == Ego2HandsDataset.RIGHT_IDX
                if right_mask.sum() < self.valid_hand_seg_th:
                    seg_real[right_mask] = 0
                    right_energy.fill(0.0)

            elif merge_mode >= 4 and merge_mode < 8:

                img_real, bg_img_resized = merge_hands(
                    right_img, left_img, bg_img, self.bg_adapt
                )
                img_real_orig, _ = merge_hands(
                    right_img_orig, left_img_orig, bg_img_resized, self.bg_adapt, False
                )
                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[left_seg] = Ego2HandsDataset.LEFT_IDX
                seg_real[right_seg] = Ego2HandsDataset.RIGHT_IDX

                left_mask = seg_real == Ego2HandsDataset.LEFT_IDX
                if left_mask.sum() < self.valid_hand_seg_th:
                    seg_real[left_mask] = 0
                    left_energy.fill(0.0)

            elif merge_mode == 8:

                img_real, bg_img_resized = merge_hands(
                    right_img, None, bg_img, self.bg_adapt
                )
                img_real_orig, _ = merge_hands(
                    right_img_orig, None, bg_img_resized, self.bg_adapt, False
                )
                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[right_seg] = Ego2HandsDataset.RIGHT_IDX
                left_energy.fill(0.0)

            elif merge_mode == 9:

                img_real, bg_img_resized = merge_hands(
                    left_img, None, bg_img, self.bg_adapt
                )
                img_real_orig, _ = merge_hands(
                    left_img_orig, None, bg_img_resized, self.bg_adapt, False
                )

                seg_real = np.zeros((img_real.shape[:2]), dtype=np.uint8)
                seg_real[left_seg] = Ego2HandsDataset.LEFT_IDX
                right_energy.fill(0.0)

            seg_real2 = cv2.resize(
                seg_real,
                (Ego2HandsDataset.IMG_W // 2, Ego2HandsDataset.IMG_H // 2),
                interpolation=cv2.INTER_NEAREST,
            )
            seg_real4 = cv2.resize(
                seg_real,
                (Ego2HandsDataset.IMG_W // 4, Ego2HandsDataset.IMG_H // 4),
                interpolation=cv2.INTER_NEAREST,
            )

            img_real_orig_tensor = torch.from_numpy(img_real_orig)
            img_real = cv2.cvtColor(img_real, cv2.COLOR_RGB2GRAY)

            if self.input_edge:
                img_edge = cv2.Canny(img_real.astype(np.uint8), 25, 100).astype(
                    np.float32
                )
                img_real = np.stack((img_real, img_edge), -1)
            else:
                img_real = np.expand_dims(img_real, -1)

            img_id = torch.from_numpy(np.array([0]))
            img_real_tensor = normalize_tensor(
                torch.from_numpy(img_real.transpose(2, 0, 1)), 128.0, 256.0
            )
            seg_real_tensor = torch.from_numpy(seg_real).long()
            seg_real2_tensor = torch.from_numpy(seg_real2).long()
            seg_real4_tensor = torch.from_numpy(seg_real4).long()

            return (
                img_id,
                img_real_orig_tensor,
                img_real_tensor,
                seg_real_tensor,
                seg_real2_tensor,
                seg_real4_tensor,
            )

    def __len__(self):
        return len(self.img_path_list)
