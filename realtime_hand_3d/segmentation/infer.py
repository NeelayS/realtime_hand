import os
from glob import glob

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import IoU

from ..utils import AverageMeter
from .dataset import normalize_tensor
from .models import SEG_MODELS_REGISTRY


def preprocess(img, size=(512, 288), grayscale=False, input_edge=False):

    if grayscale:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if input_edge:
        edge = cv.Canny(img.astype(np.uint8), 25, 100)
        img = np.stack((img, edge), -1)

    img = cv.resize(img, size)

    if not grayscale:

        img = img / 255.0
        IMG_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        IMG_VARS = np.array((0.229, 0.224, 0.225), dtype=np.float32)

        img -= IMG_MEAN
        img /= IMG_VARS

        img = np.transpose(img, (2, 0, 1))

    if input_edge:
        img = np.transpose(img, (2, 0, 1))

    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)

    if grayscale:
        img = normalize_tensor(img, 128.0, 256.0)

    if grayscale and not input_edge:
        img = img.unsqueeze(0)

    return img


def infer_image(
    image_path,
    model,
    out_dir,
    device="cpu",
    grayscale=False,
    input_edge=False,
    size=(512, 288),
):

    device = torch.device(device)

    model = model.to(device)
    model.eval()

    img = cv.imread(image_path)
    H, W, _ = img.shape
    img_r = img.copy()  # img_r = cv.resize(img, size)

    img = preprocess(img, size=size, grayscale=grayscale, input_edge=input_edge)

    with torch.no_grad():
        pred = model(img.to(device))

    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[0]

    pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)[0]
    # pred = F.interpolate(
    #     pred, size=img.shape[-2:], mode="bilinear", align_corners=True
    # )[0]
    pred = F.softmax(pred, dim=0)
    mask = torch.argmax(pred, dim=0).cpu().numpy()

    img_r[mask == 1] = [127, 127, 255]
    img_r[mask == 2] = [255, 127, 127]

    img_r = cv.resize(img_r, (W, H))
    cv.imwrite(
        os.path.join(
            out_dir, model.__class__.__name__ + "_" + image_path.split("/")[-1]
        ),
        img_r,
    )


def infer_video(
    video_path,
    model,
    out_dir,
    device="cpu",
    grayscale=False,
    input_edge=False,
    size=(512, 288),
):

    device = torch.device(device)

    model = model.to(device)
    model.eval()

    video_name = video_path.split("/")[-1]
    cap = cv.VideoCapture(os.path.join(video_path))

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out_video = cv.VideoWriter(
        os.path.join(out_dir, model.__class__.__name__ + "_" + video_name),
        fourcc,
        cap.get(cv.CAP_PROP_FPS),  # *'XVID' .avi
        (int(cap.get(3)), int(cap.get(4))),
    )

    while cap.isOpened():

        ret, frame = cap.read()
        if ret is False:
            break

        H, W = frame.shape[:2]

        img = preprocess(frame, size=size, grayscale=grayscale, input_edge=input_edge)
        img = img.to(device)

        with torch.no_grad():
            pred = model(img)

        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]

        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)[0]
        pred = F.softmax(pred, dim=0)
        mask = torch.argmax(pred, dim=0).cpu().numpy()

        new_frame = frame
        new_frame[mask == 1] = [127, 127, 255]
        new_frame[mask == 2] = [255, 127, 127]
        new_frame = cv.resize(new_frame, (W, H))

        out_video.write(new_frame)

    cap.release()
    out_video.release()


def eval_imgs(
    model, img_dir, target_dir, grayscale, input_edge, device="cpu", size=(512, 288)
):

    device = torch.device(device)

    model = model.to(device)
    model.eval()

    metric_fn = IoU(num_classes=3).to(device)
    metric_meter = AverageMeter()

    img_paths = sorted(
        glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png"))
    )
    target_paths = sorted(
        glob(os.path.join(target_dir, "*.jpg"))
        + glob(os.path.join(target_dir, "*.png"))
    )

    for img_path, target_path in zip(img_paths, target_paths):

        img = cv.imread(img_path)
        H, W, _ = img.shape
        img = preprocess(img, size=size, grayscale=grayscale, input_edge=input_edge)

        target = cv.imread(target_path, cv.IMREAD_GRAYSCALE) // 127
        target = torch.from_numpy(target).long()

        with torch.no_grad():
            pred = model(img.to(device))

        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]

        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=True)
        pred = F.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)

        target = target.unsqueeze(0)

        metric = metric_fn(pred, target).item()
        print(metric)
        metric_meter.update(metric)

    print(f"The average evaluation metric for the images is {metric_meter.avg}")

    return metric_meter.avg


def setup_model(model_name, weights_path=None, grayscale=False, input_edge=False):

    if grayscale:
        in_channels = 1
    if input_edge:
        in_channels = 2
    else:
        in_channels = 3

    model = SEG_MODELS_REGISTRY.get(model_name)(in_channels=in_channels, n_classes=3)

    if weights_path is not None:
        model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu")),
        )

    model.eval()

    return model


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--video", type=str, default=None, required=False)
    parser.add_argument("--img", type=str, default=None, required=False)
    parser.add_argument("--img_dir", type=str, default=None, required=False)
    parser.add_argument("--target_dir", type=str, default=None, required=False)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grayscale", action="store_true", default=False)
    parser.add_argument("--input_edge", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--size", type=str, default="512,288")
    args = parser.parse_args()

    size = tuple(map(int, args.size.split(",")))

    model = setup_model(args.model, args.weights, args.grayscale, args.input_edge)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    if args.eval and args.target_dir is not None:
        eval_imgs(
            model,
            args.img_dir,
            args.target_dir,
            args.grayscale,
            args.input_edge,
            args.device,
            size=size,
        )

    elif args.video is not None:
        infer_video(
            args.video,
            model,
            args.out_dir,
            args.device,
            args.grayscale,
            args.input_edge,
        )

    elif args.img is not None:
        infer_image(
            args.img,
            model,
            args.out_dir,
            args.device,
            args.grayscale,
            args.input_edge,
            size=size,
        )

    if args.save is True and args.img_dir is not None:
        for image_path in sorted(glob(os.path.join(args.img_dir, "*.png"))):

            if not image_path.split("/")[-1][:-4].isnumeric():
                continue

            print(f"Inferring {image_path}")
            infer_image(
                image_path,
                model,
                args.out_dir,
                args.device,
                args.grayscale,
                args.input_edge,
                size=size,
            )
