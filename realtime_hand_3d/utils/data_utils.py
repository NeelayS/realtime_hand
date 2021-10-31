import os

import cv2 as cv


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

    os.system(command)
